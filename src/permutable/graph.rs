use crate::autoregressive::graph::GraphPrefix;
use crate::codec::{Codec, EqSymbol, Message, OrdSymbol, IID};
use crate::joint::Canonizable;
use crate::multiset::Multiset;
use crate::permutable::{FBuildHasher, FHashMap, FHashSet, Hashing, Perm, Permutable, Permutation, Unordered};
use itertools::Itertools;
use rayon::prelude::*;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::repeat_n;
use std::marker::PhantomData;

pub type EdgeIndex = (usize, usize);
pub type Edge<E = ()> = (EdgeIndex, E);
pub type NeighborMap<E> = FHashMap<usize, E>;

/// Marker type for a directed graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Directed {}

/// Marker type for an undirected graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Undirected {}

/// A graph's edge type determines whether it has directed edges or not.
pub trait EdgeType: EqSymbol {
    fn is_directed() -> bool;
}

impl EdgeType for Directed {
    #[inline]
    fn is_directed() -> bool {
        true
    }
}

impl EdgeType for Undirected {
    #[inline]
    fn is_directed() -> bool {
        false
    }
}

/// Ordered graph with optional node and edge labels.
#[derive(Clone, Eq, PartialEq)]
pub struct Graph<N: EqSymbol = (), E: EqSymbol = (), Ty: EdgeType = Directed> {
    pub nodes: Vec<(N, NeighborMap<E>)>,
    pub phantom: PhantomData<Ty>,
}

pub type DiGraph<N = (), E = ()> = Graph<N, E, Directed>;
pub type UnGraph<N = (), E = ()> = Graph<N, E, Undirected>;
pub type PlainGraph<Ty = Directed> = Graph<(), (), Ty>;

impl<N: EqSymbol, E: EqSymbol> DiGraph<N, E> {
    pub fn into_undirected(self) -> UnGraph<N, E> {
        self.nodes.into()
    }
}

impl<N: EqSymbol, E: EqSymbol, Ty: EdgeType> From<Vec<(N, NeighborMap<E>)>> for Graph<N, E, Ty> {
    fn from(nodes: Vec<(N, NeighborMap<E>)>) -> Self {
        let out = Self { nodes, phantom: PhantomData };
        if !Ty::is_directed() {
            out.verify_is_undirected();
        }
        out
    }
}

impl<N: EqSymbol, E: EqSymbol, Ty: EdgeType> Graph<N, E, Ty> {
    /// Only in debug build since profiling on large graphs showed a significant performance impact.
    fn verify_is_undirected(&self) {
        debug_assert!(self.nodes.iter().enumerate().all(|(i, (_, ne))|
            ne.iter().all(|(j, l)| self.nodes[*j].1.get(&i) == Some(l))));
    }

    pub fn edges(&self) -> Vec<Edge<E>> {
        if Ty::is_directed() {
            self.directed_edges().collect()
        } else {
            self.directed_edges().filter(|&((i, j), _)| i <= j).collect()
        }
    }

    pub fn insert_edge(&mut self, (i, j): EdgeIndex, e: E) -> Option<E> {
        let out = self.insert_directed_edge((i, j), e.clone());
        if !Ty::is_directed() && i != j {
            assert_eq!(out, self.insert_directed_edge((j, i), e));
        }
        out
    }

    pub fn remove_edge(&mut self, (i, j): EdgeIndex) -> Option<E> {
        let out = self.remove_directed_edge((i, j));
        if !Ty::is_directed() && i != j {
            assert_eq!(out, self.remove_directed_edge((j, i)));
        }
        out
    }

    pub fn empty(nodes: impl IntoIterator<Item=N>) -> Self {
        nodes.into_iter().map(|n| (n, NeighborMap::default())).collect_vec().into()
    }

    pub fn new(nodes: impl IntoIterator<Item=N>, edges: impl IntoIterator<Item=Edge<E>>) -> Self {
        let mut x = Self::empty(nodes);
        for (e, l) in edges {
            assert!(x.insert_edge(e, l).is_none(), "Duplicate edge {:?}", e);
        }
        x
    }

    #[allow(unused)]
    pub fn edge_indices(&self) -> Vec<EdgeIndex> {
        self.edges().into_iter().map(|(i, _)| i).collect()
    }

    pub fn edge_labels(&self) -> Vec<E> {
        self.edges().into_iter().map(|(_, e)| e).collect()
    }

    pub fn edge(&self, (i, j): EdgeIndex) -> Option<E> {
        self.nodes[i].1.get(&j).cloned()
    }

    pub fn has_edge(&self, (i, j): &EdgeIndex) -> bool {
        self.nodes[*i].1.contains_key(j)
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

    pub fn node_labels(&self) -> impl ExactSizeIterator<Item=N> + '_ {
        self.nodes.iter().map(|(n, _)| n.clone())
    }

    pub fn has_selfloops(&self) -> bool {
        self.nodes.iter().enumerate().any(|(i, (_, ne))| ne.contains_key(&i))
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

impl<N: EqSymbol, E: EqSymbol, Ty: EdgeType> Permutable for Graph<N, E, Ty> {
    fn len(&self) -> usize { self.nodes.len() }

    fn swap(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }

        if Ty::is_directed() {
            *self = self.permuted(&Perm::create_swap(self.len(), i, j));
            return;
        }

        self.nodes.swap(i, j);
        let nei = self.nodes[i].1.clone();
        let nej = self.nodes[j].1.clone();
        let mut neighbors = nei.keys().chain(nej.keys()).collect::<FHashSet<_>>();
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

    fn permuted(&self, p: &impl Permutation) -> Self {
        assert_eq!(self.len(), p.len());
        self.nodes.permuted(p).into_par_iter().
            map(|(n, ne)| (n, ne.into_iter().
                map(|(j, l)| (p.apply(j), l)).collect())).collect::<Vec<_>>().into()
    }
}

impl<Ty: EdgeType> PlainGraph<Ty> {
    pub fn plain_empty(len: usize) -> Self {
        Self::empty(repeat_n((), len))
    }
}

impl PlainGraph {
    pub fn plain(len: usize, edges: impl IntoIterator<Item=EdgeIndex>) -> Self {
        Graph::new(repeat_n((), len), edges.into_iter().map(|i| (i, ())))
    }
}

impl<N: EqSymbol, Ty: EdgeType> Graph<N, (), Ty> {
    /// Returns true if edge was inserted, false if it already existed.
    pub fn insert_plain_edge(&mut self, e: (usize, usize)) -> bool {
        self.insert_edge(e, ()).is_none()
    }

    /// Returns true if edge was removed, false if it didn't exist.
    pub fn remove_plain_edge(&mut self, e: (usize, usize)) -> bool {
        self.remove_edge(e).is_some()
    }
}

impl UnGraph {
    #[allow(unused)]
    pub fn plain(len: usize, undirected_edges: impl IntoIterator<Item=EdgeIndex>) -> Self {
        DiGraph::plain(len, undirected_edges.into_iter().flat_map(|(i, j)| [(i, j), (j, i)])).into_undirected()
    }
}

impl<N: EqSymbol, E: EqSymbol, Ty: EdgeType> Debug for Graph<N, E, Ty> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let edges = self.edges();

        fn universal_symbol<S: EqSymbol>(items: impl IntoIterator<Item=S>) -> Option<S> {
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

#[cfg(test)]
mod tests {
    use crate::experimental::complete_joint::aut::AutCanonizable;
    use crate::joint::Canonizable;

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

/// Color refinement algorithm for isomorphism testing.
/// Also known as the 1-dimensional version of the Weisfeiler-Leman algorithm.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColorRefinement {
    pub convs: usize,
    /// Experimental. If true, initialize color refinement hashes with the multiset of neighboring 
    /// edge attributes, if available. Maximizes available information without requiring additional 
    /// update steps when caching on prefixes with full autoregressive shuffle coding.
    pub edge_label_init: bool,
}

impl<N: OrdSymbol, E: OrdSymbol, Ty: EdgeType> Hashing<Graph<N, E, Ty>> for ColorRefinement {
    type Hash = Vec<usize>;

    /// Node hashes of a graph, obtained using the color refinement algorithm.
    fn apply(&self, x: &Graph<N, E, Ty>) -> Self::Hash {
        self.convolutions(x, &|i| &x.nodes[i].0)
    }
}

impl<N: OrdSymbol + Default, E: OrdSymbol, Ty: EdgeType> Hashing<GraphPrefix<N, E, Ty>> for ColorRefinement {
    type Hash = Vec<usize>;

    /// Node hashes of a graph prefix, obtained using the color refinement algorithm.
    fn apply(&self, x: &GraphPrefix<N, E, Ty>) -> Self::Hash {
        let mut hashes = self.convolutions(&x.graph, &|i| x.node_label_or_index(i));
        hashes.truncate(x.len);
        hashes
    }
}

impl ColorRefinement {
    pub fn new(convs: usize, edge_label_init: bool) -> Self {
        Self { convs, edge_label_init }
    }

    pub fn convolutions<N: OrdSymbol, E: OrdSymbol, Ty: EdgeType, I: Hash>(&self, x: &Graph<N, E, Ty>, initial: &(impl Fn(usize) -> I + Sync + Send)) -> Vec<usize> {
        assert!(!Ty::is_directed());

        let mut hashes = self.init(x, &initial);
        for _ in 0..self.convs.min(x.len()) {
            hashes = Self::convolve(x, &hashes);
        }
        hashes
    }

    /// Color refinement hash of the graph.
    #[allow(unused)]
    pub fn hash<N: OrdSymbol, E: OrdSymbol, Ty: EdgeType>(&self, x: &Graph<N, E, Ty>) -> usize {
        Self::get_hash(self.apply(x).canonized())
    }

    /// 1D-Weisfeiler-Leman graph isomorphism test.
    /// If false, the graphs are guaranteed to be non-isomorphic.
    /// If true, the graphs may still be non-isomorphic.
    #[allow(unused)]
    pub fn maybe_isomorphic<N: OrdSymbol, E: OrdSymbol, Ty: EdgeType>(&self, x: &Graph<N, E, Ty>, y: &Graph<N, E, Ty>) -> bool {
        self.hash(x) == self.hash(y)
    }

    pub fn init<N: OrdSymbol, E: OrdSymbol, Ty: EdgeType, I: Hash>(&self, x: &Graph<N, E, Ty>, hasher: &(impl Fn(usize) -> I + Sync + Send)) -> Vec<usize> {
        (0..x.len()).into_par_iter().map(|index| self.init_node(x, index, hasher)).collect()
    }

    pub fn init_node<N: OrdSymbol, E: OrdSymbol, Ty: EdgeType, I: Hash>(&self, x: &Graph<N, E, Ty>, i: usize, hasher: &impl Fn(usize) -> I) -> usize {
        if self.edge_label_init && size_of::<E>() != 0 {
            Self::get_hash((hasher(i), x.nodes[i].1.iter().map(|(_, e)| e).sorted_unstable().collect_vec()))
        } else {
            Self::get_hash((hasher(i), x.nodes[i].1.len()))
        }
    }

    pub fn convolve<N: OrdSymbol, E: OrdSymbol, Ty: EdgeType>(x: &Graph<N, E, Ty>, hashes: &Vec<usize>) -> Vec<usize> {
        hashes.par_iter().enumerate().map(|(i, h)|
            Self::get_hash((h, x.nodes[i].1.iter().map(|(ne, e)| (hashes[*ne], e)).sorted_unstable().collect_vec()))
        ).collect()
    }

    pub fn get_hash(obj: impl Hash) -> usize {
        let mut hasher = FBuildHasher::default().build_hasher();
        obj.hash(&mut hasher);
        hasher.finish() as usize
    }
}

/// Coding edge indices independently of the i.i.d. labels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EdgesIID<EdgeC: Codec, IndicesC: EdgeIndicesCodec> {
    pub indices: IndicesC,
    pub label: EdgeC,
}

impl<EdgeC: Codec<Symbol: Ord> + Clone, IndicesC: EdgeIndicesCodec> Codec for EdgesIID<EdgeC, IndicesC> {
    type Symbol = Multiset<Edge<EdgeC::Symbol>>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let (indices, labels) = Self::split(x);
        self.labels(indices.len()).push(m, &labels);
        self.indices.push(m, &indices);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let mut indices = self.indices.pop(m).into_ordered();
        indices.sort_unstable();
        let labels = self.labels(indices.len()).pop(m);
        Unordered(indices.into_iter().zip_eq(labels.into_iter()).collect_vec())
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let (indices, edges) = Self::split(x);
        let labels_codec = self.labels(indices.len());
        Some(self.indices.bits(&indices)? + labels_codec.bits(&edges)?)
    }
}

impl<EdgeC: Codec<Symbol: Ord> + Clone, IndicesC: EdgeIndicesCodec<Ty=Ty>, Ty: EdgeType> EdgesIID<EdgeC, IndicesC> {
    fn split(x: &Multiset<Edge<<EdgeC as Codec>::Symbol>>) -> (Multiset<EdgeIndex>, Vec<<EdgeC as Codec>::Symbol>) {
        let (indices, labels) = x.to_ordered().iter().cloned().sorted_unstable_by_key(|(i, _)| *i).unzip();
        (Unordered(indices), labels)
    }

    fn labels(&self, len: usize) -> IID<EdgeC> {
        IID::new(self.label.clone(), len)
    }

    pub fn new(indices: IndicesC, label: EdgeC) -> Self {
        Self { indices, label }
    }
}

pub trait EdgeIndicesCodec: Codec<Symbol=Multiset<EdgeIndex>> {
    type Ty: EdgeType;

    fn loops(&self) -> bool;
}
