use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::iter::repeat;
use std::marker::PhantomData;
use std::ops::Range;
use std::os::raw::c_int;

use itertools::Itertools;
use nauty_Traces_sys::{FALSE, nauty_check, NAUTYVERSIONID, optionblk, ran_init, SETWORDSNEEDED, SparseGraph, sparsegraph, sparsenauty, statsblk, Traces, TracesOptions, TracesStats, TRUE, WORDSIZE};
use petgraph::{Directed, Undirected};
use petgraph;

use crate::ans::Symbol;
use crate::permutable::{Automorphisms, GroupPermutable, GroupPermutableFromFused, Partition, Permutable, Permutation, PermutationGroup};
use crate::permutable::Partial;

pub type EdgeIndex = (usize, usize);
pub type Edge<E = ()> = (EdgeIndex, E);
/// We need deterministic results for getting the automorphism group generators from nauty.
/// For this reason, the list of neighbors we pass to nauty needs to be canonized, i. e. sorted.
/// Using BTreeMap instead of a HashMap avoids the cost for sorting within the codecs.
pub type NeighborMap<E> = BTreeMap<usize, E>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AutomorphismsBackend { SparseNauty, Traces }

pub trait EdgeType: petgraph::EdgeType + Clone + Debug {}

impl<T: petgraph::EdgeType + Clone + Debug> EdgeType for T {}

/// Ordered graph with optional node and edge labels.
#[derive(Clone)]
pub struct Graph<N: Symbol = (), E: Symbol = (), Ty: EdgeType = Directed> where Graph<N, E, Ty>: AsEdges<N, E> {
    pub nodes: Vec<(N, NeighborMap<E>)>,
    preferred_aut_backend: AutomorphismsBackend,
    phantom: PhantomData<Ty>,
}

pub type DiGraph<N = (), E = ()> = Graph<N, E, Directed>;
pub type UnGraph<N = (), E = ()> = Graph<N, E, Undirected>;
pub type PlainGraph<Ty = Directed> = Graph<(), (), Ty>;

impl<N: Symbol, E: Symbol, Ty: EdgeType> PartialEq for Graph<N, E, Ty> where Graph<N, E, Ty>: AsEdges<N, E> {
    fn eq(&self, other: &Self) -> bool {
        self.nodes == other.nodes
    }
}

impl<N: Symbol, E: Symbol, Ty: EdgeType> Eq for Graph<N, E, Ty> where Graph<N, E, Ty>: AsEdges<N, E> {}

pub trait AsEdges<N: Symbol = (), E: Symbol = ()>: Sized {
    fn from_neighbors(neighbors: impl IntoIterator<Item=(N, NeighborMap<E>)>) -> Self;

    fn empty(nodes: impl IntoIterator<Item=N>) -> Self {
        Self::from_neighbors(nodes.into_iter().map(|n| (n, NeighborMap::new())))
    }

    fn new(nodes: impl IntoIterator<Item=N>, edges: impl IntoIterator<Item=Edge<E>>) -> Self {
        let mut x = Self::empty(nodes);
        for (e, l) in edges {
            assert!(x.insert_edge(e, l).is_none(), "Duplicate edge {:?}", e);
        }
        x
    }

    fn edges(&self) -> Vec<Edge<E>>;

    fn insert_edge(&mut self, i: EdgeIndex, e: E) -> Option<E>;

    fn remove_edge(&mut self, i: EdgeIndex) -> Option<E>;

    #[allow(unused)]
    fn edge_indices(&self) -> Vec<EdgeIndex> {
        self.edges().into_iter().map(|(i, _)| i).collect()
    }

    fn edge_labels(&self) -> Vec<E> {
        self.edges().into_iter().map(|(_, e)| e).collect()
    }
}

impl<N: Symbol, E: Symbol> DiGraph<N, E> {
    pub fn into_undirected(self) -> UnGraph<N, E> {
        let Self { nodes, preferred_aut_backend, .. } = self;
        let out = UnGraph { nodes, preferred_aut_backend, phantom: Default::default() };
        out.verify_is_undirected();
        out
    }
}

impl<N: Symbol, E: Symbol> AsEdges<N, E> for DiGraph<N, E> {
    fn from_neighbors(neighbors: impl IntoIterator<Item=(N, NeighborMap<E>)>) -> Self {
        Self::directed_from_neighbors(neighbors)
    }

    fn edges(&self) -> Vec<Edge<E>> {
        self.directed_edges().collect()
    }

    fn insert_edge(&mut self, i: EdgeIndex, e: E) -> Option<E> {
        self.insert_directed_edge(i, e)
    }

    fn remove_edge(&mut self, i: EdgeIndex) -> Option<E> {
        self.remove_directed_edge(i)
    }
}

impl<N: Symbol, E: Symbol> AsEdges<N, E> for UnGraph<N, E> {
    fn from_neighbors(neighbors: impl IntoIterator<Item=(N, NeighborMap<E>)>) -> Self {
        let out = Self::directed_from_neighbors(neighbors);
        out.verify_is_undirected();
        out
    }

    fn edges(&self) -> Vec<Edge<E>> {
        self.directed_edges().filter(|&((i, j), _)| i <= j).collect()
    }

    fn insert_edge(&mut self, (i, j): EdgeIndex, e: E) -> Option<E> {
        let out = self.insert_directed_edge((i, j), e.clone());
        if i != j {
            assert_eq!(out, self.insert_directed_edge((j, i), e));
        }
        out
    }

    fn remove_edge(&mut self, (i, j): EdgeIndex) -> Option<E> {
        let out = self.remove_directed_edge((i, j));
        if i != j {
            assert_eq!(out, self.remove_directed_edge((j, i)));
        }
        out
    }
}

impl<N: Symbol, E: Symbol, Ty: EdgeType> Graph<N, E, Ty> where Graph<N, E, Ty>: AsEdges<N, E> {
    pub fn directed_from_neighbors(neighbors: impl IntoIterator<Item=(N, NeighborMap<E>)>) -> Self {
        Self { nodes: neighbors.into_iter().collect(), preferred_aut_backend: AutomorphismsBackend::SparseNauty, phantom: Default::default() }
    }

    pub fn edge(&self, (i, j): EdgeIndex) -> Option<E> {
        self.nodes[i].1.get(&j).cloned()
    }

    fn insert_directed_edge(&mut self, (i, j): EdgeIndex, e: E) -> Option<E> {
        assert!(j < self.len());
        self.nodes[i].1.insert(j, e)
    }

    fn remove_directed_edge(&mut self, (i, j): EdgeIndex) -> Option<E> {
        self.nodes[i].1.remove(&j)
    }

    fn directed_edges(&self) -> impl Iterator<Item=Edge<E>> + '_ {
        self.nodes.iter().enumerate().flat_map(|(i, (_, ne))|
            ne.iter().map(move |(&j, l)| ((i, j), l.clone())))
    }

    pub fn node_labels(&self) -> impl Iterator<Item=N> + '_ {
        self.nodes.iter().map(|(n, _)| n.clone())
    }

    pub fn has_selfloops(&self) -> bool {
        self.nodes.iter().enumerate().any(|(i, (_, ne))| ne.contains_key(&i))
    }

    /// Automorphisms of the corresponding unlabelled graph (ignoring node and edge labels),
    /// respecting the given partitions.
    pub fn unlabelled_automorphisms(&self, partitions: Option<impl IntoIterator<Item=Partition>>) -> Automorphisms {
        if self.len() == 0 { // Traces breaks on graph size 0.
            return Automorphisms {
                group: PermutationGroup::new(0, vec![]),
                canon: Permutation::identity(0),
                decanon: Permutation::identity(0),
                orbits: vec![],
                bits: 0.,
            };
        }

        let defaultptn = if partitions.is_none() { TRUE } else { FALSE };

        let (mut lab, mut ptn) = if let Some(partitions) = partitions {
            partitions.into_iter().flat_map(|p| {
                p.into_iter().enumerate().rev().
                    map(|(i, x)| (x as i32, if i == 0 { 0 } else { 1 }))
            }).unzip()
        } else { (vec![0; self.len()], vec![0; self.len()]) };
        let mut orbs = vec![0; self.len()];

        unsafe {
            nauty_check(WORDSIZE as c_int, SETWORDSNEEDED(self.len()) as c_int,
                        self.len() as c_int, NAUTYVERSIONID as c_int);
        }

        let sg = &mut self.to_nauty();
        let lab_ptr = lab.as_mut_ptr();
        let ptn_ptr = ptn.as_mut_ptr();
        let orbs_ptr = orbs.as_mut_ptr();

        thread_local! {
            /// Collect generators via static C callback function:
            static GENERATORS: RefCell<Vec<Permutation>> = RefCell::new(vec![]);
        }
        extern "C" fn push_generator(ordinal: c_int, perm: *mut c_int, n: c_int) {
            let generator = Permutation::from((0..n).map(
                |i| unsafe { *perm.offset(i as isize) } as usize).collect());
            GENERATORS.with(|g| {
                let mut generators = g.borrow_mut();
                generators.push(generator);
                assert_eq!(ordinal as usize, generators.len());
            });
        }
        extern "C" fn push_generator_from_nauty(ordinal: c_int, perm: *mut c_int, _orbits: *mut c_int,
                                                _numorbits: c_int, _stabnode: c_int, n: c_int) {
            push_generator(ordinal, perm, n);
        }

        // OPTIMIZE: precompute undirectedness:
        let (grpsize1, grpsize2) = if
        self.preferred_aut_backend == AutomorphismsBackend::Traces && !Ty::is_directed() {
            let options = &mut TracesOptions::default();
            options.getcanon = TRUE;
            options.userautomproc = Some(push_generator);
            options.defaultptn = defaultptn;
            let stats = &mut TracesStats::default();
            unsafe {
                ran_init(0);
                Traces(&mut sg.into(), lab_ptr, ptn_ptr, orbs_ptr,
                       options, stats, std::ptr::null_mut());
            }
            (stats.grpsize1, stats.grpsize2)
        } else {
            let options = &mut if Ty::is_directed() {
                optionblk::default_sparse_digraph()
            } else {
                optionblk::default_sparse()
            };
            options.getcanon = TRUE;
            options.userautomproc = Some(push_generator_from_nauty);
            options.defaultptn = defaultptn;
            let stats = &mut statsblk::default();
            thread_local! {
                /// Avoid canonized graph allocation for every call, nauty allows reuse:
                static CG: RefCell<sparsegraph> = RefCell::new(sparsegraph::default());
            }
            CG.with(|cg| unsafe {
                sparsenauty(&mut sg.into(), lab_ptr, ptn_ptr, orbs_ptr,
                            options, stats, &mut *cg.borrow_mut())
            });
            (stats.grpsize1, stats.grpsize2)
        };

        let generators = GENERATORS.with(|g| {
            let mut gens = g.borrow_mut();
            let out = gens.clone();
            gens.clear();
            out
        });

        let decanon = Permutation::from(lab.into_iter().map(|x| x as usize).collect());
        let canon = decanon.inverse();
        let orbits = orbs.orbits();
        let grpsize2: f64 = grpsize2.try_into().unwrap();
        let bits: f64 = grpsize1.log2() + grpsize2 * (10f64).log2();
        let group = PermutationGroup::new(self.len(), generators);
        Automorphisms { group, canon, decanon, orbits, bits }
    }

    fn to_nauty(&self) -> SparseGraph {
        let d = self.degrees().map(|x| x as i32).collect_vec();
        let v = d.iter().map(|d| *d as usize).scan(0, |acc, d| {
            let out = Some(*acc);
            *acc += d;
            out
        }).collect();
        let e = self.nodes.iter().map(|(_, ne)| ne.iter().map(
            |(i, _)| *i as i32).collect()).concat();
        SparseGraph { v, d, e }
    }

    pub fn degrees(&self) -> impl Iterator<Item=usize> + '_ {
        self.nodes.iter().map(|(_, ne)| ne.len())
    }

    pub fn num_edges(&self) -> usize {
        let e = self.degrees().sum();
        if Ty::is_directed() { e } else { (e + self.num_selfloops()) / 2 }
    }

    pub fn num_selfloops(&self) -> usize {
        self.nodes.iter().enumerate().filter(|(i, (_, ne))| ne.get(&i).is_some()).count()
    }
}

impl<N: Symbol, E: Symbol, Ty: EdgeType> Permutable for Graph<N, E, Ty> where Graph<N, E, Ty>: AsEdges<N, E> {
    fn len(&self) -> usize { self.nodes.len() }

    fn swap(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }

        if Ty::is_directed() {
            *self = self.permuted(&Permutation::create_swap(self.len(), i, j));
            return;
        }

        self.nodes.swap(i, j);
        let nei = self.nodes[i].1.clone();
        let nej = self.nodes[j].1.clone();
        let mut neighbors = nei.keys().chain(nej.keys()).collect::<BTreeSet<_>>();
        neighbors.insert(&i);
        neighbors.insert(&j);
        for n in neighbors {
            let ne = &mut self.nodes[*n].1;
            let ei = ne.remove(&j);
            let ej = ne.remove(&i);
            if let Some(ei) = ei {
                ne.insert(i, ei);
            }
            if let Some(ej) = ej {
                ne.insert(j, ej);
            }
        }
    }

    fn permuted(&self, p: &Permutation) -> Self {
        assert_eq!(self.len(), p.len);
        return Graph::from_neighbors((p * &self.nodes).into_iter().
            map(|(n, ne)| (n, ne.into_iter().
                map(|(j, l)| (p * j, l)).collect())));
    }
}

impl<Ty: EdgeType> GroupPermutableFromFused for PlainGraph<Ty> where PlainGraph<Ty>: AsEdges {
    fn auts(&self) -> Automorphisms { self.unlabelled_automorphisms(None::<Vec<_>>) }
}

impl<Ty: EdgeType> PlainGraph<Ty> where PlainGraph<Ty>: AsEdges {
    pub fn plain_empty(len: usize) -> Self {
        Self::empty(repeat(()).take(len))
    }
}

impl PlainGraph {
    pub fn plain(len: usize, edges: impl IntoIterator<Item=EdgeIndex>) -> Self {
        Graph::new(repeat(()).take(len), edges.into_iter().map(|i| (i, ())))
    }
}

impl<N: Symbol, Ty: EdgeType> Graph<N, (), Ty> where Graph<N, (), Ty>: AsEdges<N> {
    /// Returns true if edge was inserted, false if it already existed.
    pub fn insert_plain_edge(&mut self, e: (usize, usize)) -> bool {
        self.insert_edge(e, ()).is_none()
    }

    /// Returns true if edge was removed, false if it didn't exist.
    pub fn remove_plain_edge(&mut self, e: (usize, usize)) -> bool {
        self.remove_edge(e).is_some()
    }
}

impl<N: Symbol, E: Symbol> UnGraph<N, E> {
    fn verify_is_undirected(&self) {
        assert!(self.nodes.iter().enumerate().all(|(i, (_, ne))|
            ne.iter().all(|(j, l)| self.nodes[*j].1.get(&i) == Some(l))));
    }
}

impl UnGraph {
    pub fn plain(len: usize, undirected_edges: impl IntoIterator<Item=EdgeIndex>) -> Self {
        DiGraph::plain(len, undirected_edges.into_iter().flat_map(|(i, j)| [(i, j), (j, i)])).into_undirected()
    }
}

impl<N: Symbol, E: Symbol, Ty: EdgeType> Debug for Graph<N, E, Ty> where Graph<N, E, Ty>: AsEdges<N, E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let edges = self.edges();

        fn universal_symbol<S: Symbol>(items: impl IntoIterator<Item=S>) -> Option<S> {
            let mut iter = items.into_iter();
            let first = iter.next()?;
            iter.all(|x| x == first).then(|| first)
        }

        let universal_node_label = universal_symbol(self.node_labels());
        let universal_edge_label = universal_symbol(edges.iter().map(|(_, l)| l.clone()));
        let nodes_str = if let Some(universal_node_label) = universal_node_label {
            format!("all {:?}", universal_node_label)
        } else {
            format!("{:?}", self.node_labels().collect_vec())
        };
        let edges_str = if let Some(universal_edge_label) = universal_edge_label {
            format!("all {:?} at {:?}", universal_edge_label, edges.iter().map(|(e, _)| e.clone()).collect_vec())
        } else {
            format!("{:?}", edges)
        };

        write!(f, "N={}: {nodes_str}, E={}{}: {edges_str}", self.len(), edges.len(), if Ty::is_directed() { "" } else { "u" })
    }
}


/// Graph where the edge structure between the last num_hidden_nodes nodes is unknown.
#[derive(Clone, Debug)]
pub struct PartialGraph<N: Symbol = (), E: Symbol = (), Ty: EdgeType = Directed> where Graph<Option<N>, E, Ty>: AsEdges<Option<N>, E> {
    pub graph: Graph<Option<N>, E, Ty>,
    pub num_unknown_nodes: usize,
}

/// petgraph::Directed/Undirected doesn't impl Eq, workaround:
impl<N: Symbol, E: Symbol, Ty: EdgeType> PartialEq for PartialGraph<N, E, Ty> where Graph<Option<N>, E, Ty>: AsEdges<Option<N>, E> {
    fn eq(&self, other: &Self) -> bool {
        self.graph == other.graph
    }
}

impl<N: Symbol, E: Symbol, Ty: EdgeType> Eq for PartialGraph<N, E, Ty> where Graph<Option<N>, E, Ty>: AsEdges<Option<N>, E> {}

impl<N: Symbol, E: Symbol, Ty: EdgeType> Permutable for PartialGraph<N, E, Ty> where Graph<Option<N>, E, Ty>: AsEdges<Option<N>, E> {
    fn len(&self) -> usize {
        self.graph.len() - self.num_unknown_nodes
    }

    fn swap(&mut self, i: usize, j: usize) {
        assert!(i < self.len());
        assert!(j < self.len());
        self.graph.swap(i, j);
    }
}

impl<N: Symbol + Ord + Hash, Ty: EdgeType> GroupPermutableFromFused for PartialGraph<N, (), Ty> where Graph<Option<N>, (), Ty>: AsEdges<Option<N>, ()> {
    fn auts(&self) -> Automorphisms {
        let known_partitions = self.graph.node_labels().
            take(self.len()).map(|x| x.unwrap()).collect_vec().
            orbits().into_iter();
        let unknown_partitions = self.unknown_nodes().
            map(|i| Partition::from_iter(vec![i]));
        let partitions = known_partitions.chain(unknown_partitions);
        let mut a = self.graph.unlabelled_automorphisms(Some(partitions));
        a.group.adjust_len(self.len());
        a
    }
}

impl<N: Symbol, E: Symbol, Ty: EdgeType> Partial for PartialGraph<N, E, Ty> where Graph<N, E, Ty>: AsEdges<N, E>, Graph<Option<N>, E, Ty>: AsEdges<Option<N>, E> {
    type Complete = Graph<N, E, Ty>;
    type Slice = (N, (Vec<Option<E>>, Option<E>));

    fn pop_slice(&mut self) -> Self::Slice {
        let new_unknown_node = self.last_();
        let mut edges = self.unknown_nodes().
            map(|i| self.graph.remove_edge((i, new_unknown_node))).collect_vec();
        if Ty::is_directed() {
            edges.extend(self.unknown_nodes().map(|j|
                self.graph.remove_edge((new_unknown_node, j))));
        }

        self.num_unknown_nodes += 1;
        let x = &mut self.graph.nodes[new_unknown_node];
        let node = x.0.clone().unwrap();
        x.0 = None;
        let self_loop = self.graph.remove_edge((new_unknown_node, new_unknown_node));
        (node, (edges, self_loop))
    }

    fn push_slice(&mut self, (node, (edges, selfloop)): &Self::Slice) {
        let new_known_node = self.len();
        let (n, _) = &mut self.graph.nodes[new_known_node];
        assert!(n.is_none());
        *n = Some(node.clone());

        if let Some(selfloop) = selfloop {
            assert!(self.graph.insert_edge((new_known_node, new_known_node), selfloop.clone()).is_none());
        }

        self.num_unknown_nodes -= 1;
        assert_eq!(self.num_unknown_nodes * if Ty::is_directed() { 2 } else { 1 }, edges.len());
        for (i, edge) in self.unknown_nodes().zip_eq(edges.iter().take(self.num_unknown_nodes).cloned()) {
            if edge.is_some() {
                assert!(self.graph.insert_edge((i, new_known_node), edge.unwrap()).is_none());
            }
        }
        if Ty::is_directed() {
            for (j, edge) in self.unknown_nodes().zip_eq(edges.iter().skip(self.num_unknown_nodes).cloned()) {
                if edge.is_some() {
                    assert!(self.graph.insert_edge((new_known_node, j), edge.unwrap()).is_none());
                }
            }
        }
    }

    fn empty(len: usize) -> Self {
        PartialGraph { graph: Graph::empty(repeat(None).take(len)), num_unknown_nodes: len }
    }
    fn from_complete(graph: Self::Complete) -> Self {
        let graph = Graph::from_neighbors(graph.nodes.into_iter().
            map(|(n, ne)| (Some(n), ne)));
        Self { graph, num_unknown_nodes: 0 }
    }
    fn into_complete(self) -> Self::Complete {
        assert_eq!(0, self.num_unknown_nodes);
        Graph::from_neighbors(self.graph.nodes.into_iter().map(|(n, ne)| (n.unwrap(), ne)))
    }
}


impl<N: Symbol, E: Symbol, Ty: EdgeType> PartialGraph<N, E, Ty> where Graph<Option<N>, E, Ty>: AsEdges<Option<N>, E> {
    pub fn unknown_nodes(&self) -> Range<usize> {
        self.len()..self.graph.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivial_automorphism() {
        assert_eq!(DiGraph::plain(0, []).automorphisms().group.generators, vec![]);
        assert_eq!(DiGraph::plain(1, []).automorphisms().group.generators, vec![]);
    }

    #[test]
    fn tiny_directed_automorphism() {
        let g1 = DiGraph::plain(3, [(0, 1)]);
        let g2 = DiGraph::plain(3, [(2, 1)]);
        assert!(g1.is_isomorphic(&g2));
    }
}
