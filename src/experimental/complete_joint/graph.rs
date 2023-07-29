//! Joint shuffle coding for graphs with node/edge attributes ("node/edge-labelled").

use crate::autoregressive::chunked::ChunkedPrefix;
use crate::autoregressive::graph::{GraphPrefix, GraphPrefixingChain};
use crate::autoregressive::multiset::MapOrbits;
use crate::autoregressive::PrefixingChain;
use crate::codec::{EqSymbol, OrdSymbol};
use crate::experimental::complete_joint::aut::{AutCanonizable, Automorphisms, PermutationGroup};
use crate::experimental::complete_joint::sparse_perm::SparsePerm;
use crate::graph::{Directed, EdgeType, Graph, NeighborMap, PlainGraph};
use crate::permutable::{FBuildHasher, Perm, Permutable, Permutation};
use itertools::Itertools;
use nauty_Traces_sys::{nauty_check, optionblk, ran_init, sparsegraph, sparsenauty, statsblk, SparseGraph, Traces, TracesOptions, TracesStats, FALSE, NAUTYVERSIONID, SETWORDSNEEDED, TRUE, WORDSIZE};
use std::cell::RefCell;
use std::hash::Hash;
use std::os::raw::c_int;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum AutomorphismsBackend { SparseNauty, Traces }

thread_local! {
    /// The backend used to compute undirected graph automorphisms and isomorphisms.
    /// Directed graphs are always handled by nauty.
    pub static AUTOMORPHISMS_BACKEND: RefCell<AutomorphismsBackend> = RefCell::new(AutomorphismsBackend::SparseNauty);
}

#[allow(unused)]
pub fn with_automorphisms_backend<F: FnOnce() -> R, R>(backend: AutomorphismsBackend, f: F) -> R {
    AUTOMORPHISMS_BACKEND.with(|b| {
        let old = b.replace(backend);
        let out = f();
        assert_eq!(backend, b.replace(old));
        out
    })
}

impl<N: EqSymbol, E: EqSymbol, Ty: EdgeType> Graph<N, E, Ty> {
    /// Automorphisms of the corresponding unlabelled graph (ignoring node and edge labels),
    /// respecting the partitions induced by the given labels.
    pub fn unlabelled_automorphisms<L: OrdSymbol>(&self, labels: Option<&[L]>) -> Automorphisms {
        if self.len() == 0 { // Traces breaks on graph size 0.
            return Automorphisms {
                group: PermutationGroup::new(0, vec![]),
                canon: Perm::identity(0),
                decanon: Perm::identity(0),
                orbit_ids: vec![],
                bits: 0.,
            };
        }

        let defaultptn = if labels.is_none() { TRUE } else { FALSE };

        let (mut lab, mut ptn) = if let Some(labels) = labels {
            MapOrbits::<L, FBuildHasher>::new(labels, labels.len()).orbits.into_iter().
                sorted_unstable_by_key(|p| p.0.clone()).
                flat_map(|p| {
                    p.1.into_iter().sorted_unstable().enumerate().rev().
                        map(|(i, x)| (x as i32, if i == 0 { 0 } else { 1 }))
                }).
                unzip()
        } else { (vec![0; self.len()], vec![0; self.len()]) };
        let mut orbit_ids = vec![0; self.len()];

        unsafe {
            nauty_check(WORDSIZE as c_int, SETWORDSNEEDED(self.len()) as c_int,
                        self.len() as c_int, NAUTYVERSIONID as c_int);
        }

        let sg = &mut self.to_nauty();
        let lab_ptr = lab.as_mut_ptr();
        let ptn_ptr = ptn.as_mut_ptr();
        let orbs_ptr = orbit_ids.as_mut_ptr();

        thread_local! {
            /// Collect generators via static C callback function:
            static GENERATORS: RefCell<Vec<SparsePerm>> = RefCell::new(vec![]);
        }
        extern "C" fn push_generator(ordinal: c_int, perm: *mut c_int, n: c_int) {
            let generator = SparsePerm::from((0..n).map(
                |i| unsafe { *perm.offset(i as isize) } as usize).collect_vec());
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

        let (grpsize1, grpsize2) = if
        AUTOMORPHISMS_BACKEND.with_borrow(|b| b == &AutomorphismsBackend::Traces) && !Ty::is_directed() {
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

        let decanon = Perm::from(lab.into_iter().map(|x| x as usize).collect_vec());
        let canon = decanon.inverse();
        let grpsize2: f64 = grpsize2.try_into().unwrap();
        let bits: f64 = grpsize1.log2() + grpsize2 * 10f64.log2();
        let group = PermutationGroup::new(self.len(), generators);
        Automorphisms { group, canon, decanon, orbit_ids, bits }
    }

    fn to_nauty(&self) -> SparseGraph {
        let d = self.degrees().map(|x| x as i32).collect_vec();
        let v = d.iter().map(|d| *d as usize).scan(0, |acc, d| {
            let out = Some(*acc);
            *acc += d;
            out
        }).collect();
        // We need deterministic results for getting the automorphism group generators from nauty,
        // so we sort the neighbors of each node:
        // TODO no tests fail without it
        let e = self.nodes.iter().map(|(_, ne)| ne.iter().map(
            |(i, _)| *i as i32).sorted_unstable().collect()).concat();
        SparseGraph { v, d, e }
    }
}

/// A graph with natural numbers as node labels.
pub type NodeLabelledGraph<Ty = Directed> = Graph<usize, (), Ty>;
/// A graph with natural numbers as edge and node labels.
pub type EdgeLabelledGraph<Ty = Directed> = Graph<usize, usize, Ty>;

impl<Ty: EdgeType> AutCanonizable for PlainGraph<Ty> {
    fn automorphisms(&self) -> Automorphisms { self.unlabelled_automorphisms::<()>(None) }
}

impl<Ty: EdgeType> AutCanonizable for NodeLabelledGraph<Ty> {
    fn automorphisms(&self) -> Automorphisms {
        self.unlabelled_automorphisms(Some(&self.node_labels().collect_vec()))
    }
}

impl<Ty: EdgeType> EdgeLabelledGraph<Ty> {
    /// Equivalent graph where labelled edges are converted into labelled nodes.
    fn with_edges_as_nodes(&self) -> NodeLabelledGraph<Ty> {
        let edge_label_shift = self.node_labels().max().unwrap_or(0) + 1;
        let mut extended = NodeLabelledGraph::new(self.node_labels(), vec![]);
        for ((i, j), e) in self.edges() {
            let is_undirected = self.edge((i, j)) == self.edge((j, i));
            if is_undirected && i > j {
                continue;
            }
            let new_node = extended.len();
            extended.nodes.push((edge_label_shift + e, NeighborMap::default()));

            assert!(extended.insert_plain_edge((i, new_node)));
            if Ty::is_directed() {
                assert!(extended.insert_plain_edge((new_node, j)));
            }
            if is_undirected && i != j {
                assert!(extended.insert_plain_edge((j, new_node)));
                if Ty::is_directed() {
                    assert!(extended.insert_plain_edge((new_node, i)));
                }
            }
        }
        extended
    }

    fn truncate_permutation(&self, p: &Perm) -> Perm {
        let v = p.iter().take(self.len()).collect_vec();
        for &i in v.iter() {
            assert!(i < self.len());
        }
        Perm::from(v)
    }

    fn truncate_sparse_permutation(&self, p: &SparsePerm) -> SparsePerm {
        SparsePerm {
            len: self.len(),
            indices: p.indices.clone().into_iter().filter(|(k, i)| {
                let keep = k < &self.len();
                assert_eq!(keep, i < &self.len());
                keep
            }).collect(),
        }
    }
}

impl<Ty: EdgeType> AutCanonizable for EdgeLabelledGraph<Ty> {
    fn automorphisms(&self) -> Automorphisms {
        let Automorphisms { group, canon, mut orbit_ids, bits, .. } = self.with_edges_as_nodes().automorphisms();
        let group = PermutationGroup::new(self.len(), group.
            generators.into_iter().map(|g| self.truncate_sparse_permutation(&g)).collect_vec());
        let canon = self.truncate_permutation(&canon);
        let decanon = canon.inverse();
        orbit_ids.truncate(self.len());
        Automorphisms { group, canon, decanon, orbit_ids, bits }
    }
}

pub type EdgeLabelledGraphPrefix<Ty = Directed> = GraphPrefix<usize, usize, Ty>;

impl<Ty: EdgeType> AutCanonizable for EdgeLabelledGraphPrefix<Ty> {
    fn automorphisms(&self) -> Automorphisms {
        let graph = self.with_unknown_nodes_uniquely_labelled();
        let mut a = graph.automorphisms();
        a.group.adjust_len(self.len);
        a
    }
}

impl<Ty: EdgeType> EdgeLabelledGraphPrefix<Ty> {
    fn with_unknown_nodes_uniquely_labelled(&self) -> EdgeLabelledGraph<Ty> {
        let node_label_shift = self.graph.node_labels().take(self.len).
            into_iter().max().unwrap_or(0) + 1;
        self.graph.nodes.iter().enumerate().map(
            |(i, (n, ne))| (
                if self.unknown_nodes().contains(&i) {
                    node_label_shift + i - self.unknown_nodes().start
                } else { n.clone() }, ne.clone())).collect_vec().into()
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::bench::TestConfig;
    use crate::codec::OrdSymbol;
    use crate::codec::{assert_bits_eq, Codec, EqSymbol, Uniform, UniformCodec};
    use crate::experimental::graph::tests::{graph_codecs, plain_erdos_renyi, test_graph_shuffle_codecs, with_sampled_symbols, PlainGraphIID, UnGraphIID};
    use crate::experimental::graph::{EmptyCodec, ErdosRenyi, GraphIID};
    use crate::graph::{EdgeType, UnGraph, Undirected};
    use crate::joint::Canonizable;
    use crate::permutable::PermutationUniform;
    use crate::permutable::Unordered;
    use timeit::timeit_loops;

    pub fn node_labelled<Ty: EdgeType, EdgeC: Codec<Symbol: OrdSymbol> + Clone>(
        graph_codecs: impl IntoIterator<Item=GraphIID<EmptyCodec, EdgeC, ErdosRenyi<Ty>>>, num_labels: usize,
    ) -> impl Iterator<Item=GraphIID<Uniform, EdgeC, ErdosRenyi<Ty>>> {
        graph_codecs.into_iter().map(move |c| GraphIID::new(
            c.nodes.len, c.edges.indices, Uniform::new(num_labels), c.edges.label))
    }

    pub fn node_labelled_codec_for(x: &UnGraph<usize, impl EqSymbol>, edge_prob: f64, loops: bool) -> UnGraphIID<Uniform> {
        let num_labels = x.node_labels().max().unwrap() + 1;
        let c = plain_erdos_renyi(x.len(), edge_prob, loops);
        UnGraphIID::new(c.nodes.len, c.edges.indices, Uniform::new(num_labels), c.edges.label)
    }

    pub fn with_edge_labelled_uniform(graphs: impl IntoIterator<Item=EdgeLabelledGraph<Undirected>>, edge_prob: f64, loops: bool) -> impl Iterator<Item=(UnGraphIID<Uniform, Uniform>, EdgeLabelledGraph<Undirected>)> {
        graphs.into_iter().map(move |x| {
            let c = node_labelled_codec_for(&x, edge_prob, loops);
            let num_labels = x.edge_labels().into_iter().max().unwrap_or(0) + 1;
            (GraphIID::new(c.nodes.len, c.edges.indices, c.nodes.item, Uniform::new(num_labels)), x)
        })
    }

    pub fn with_node_labelled_uniform(graphs: impl IntoIterator<Item=NodeLabelledGraph<Undirected>>, edge_prob: f64, loops: bool) -> impl Iterator<Item=(UnGraphIID<Uniform>, NodeLabelledGraph<Undirected>)> {
        graphs.into_iter().map(move |x| {
            let num_labels = x.node_labels().max().unwrap() + 1;
            let c = plain_erdos_renyi(x.len(), edge_prob, loops);
            (GraphIID::new(c.nodes.len, c.edges.indices, Uniform::new(num_labels), c.edges.label), x)
        })
    }

    #[test]
    fn edge_labelled_permutable_axioms_fixed_slow_cases() {
        let codec = edge_labelled([plain_erdos_renyi::<Directed>(10, 0.4, false)]).into_iter().next().unwrap();
        for seed in 0..5 {
            codec.sample(seed).with_edges_as_nodes().automorphisms();
        }
    }

    #[test]
    fn edge_labelled_permutable_axioms_fixed_small_slow_case() {
        let codec = edge_labelled([plain_erdos_renyi::<Directed>(9, 0.5, false)]).into_iter().next().unwrap();
        let extended = codec.sample(49).with_edges_as_nodes();
        assert!(timeit_loops!(1, { extended.automorphisms(); }) < 0.001);
    }

    #[test]
    fn water_automorphisms() {
        let h2o = EdgeLabelledGraph::<Undirected>::new(
            [0, 1, 0],
            [((1, 0), 0), ((1, 2), 0)]);
        assert_bits_eq(1., h2o.automorphisms().bits);
    }

    #[test]
    fn hydrogen_peroxide_automorphisms() {
        let h2o2 = EdgeLabelledGraph::<Undirected>::new(
            [0, 1, 1, 0],
            [((1, 0), 0), ((2, 3), 0), ((1, 2), 0)]);
        assert_bits_eq(1., h2o2.automorphisms().bits);
    }

    #[test]
    fn boric_acid_automorphisms() {
        let bh3o3 = EdgeLabelledGraph::<Undirected>::new(
            [0, 1, 2, 1, 2, 1, 2],
            [
                ((0, 1), 0), ((0, 3), 0), ((0, 5), 0),
                ((1, 2), 0), ((3, 4), 0), ((5, 6), 0)]);
        assert_bits_eq(PermutationUniform::<Perm>::new(3).uni_bits(), bh3o3.automorphisms().bits);
    }

    #[test]
    fn ethylene_automorphisms() {
        let c2h4 = EdgeLabelledGraph::<Undirected>::new(
            [0, 0, 1, 1, 0, 0],
            [((2, 0), 0), ((2, 1), 0), ((2, 3), 1), ((3, 4), 0), ((3, 5), 0)]);
        assert_bits_eq(3., c2h4.automorphisms().bits);
    }

    #[test]
    fn chiral_automorphisms() {
        let chiral = EdgeLabelledGraph::<Directed>::new(
            [0, 0, 0, 1, 2],
            [((0, 1), 0), ((0, 2), 1), ((0, 3), 0), ((0, 4), 0)]);
        assert_bits_eq(0., chiral.automorphisms().bits);
    }

    pub fn edge_labelled<Ty: EdgeType>(graph_codecs: impl IntoIterator<Item=PlainGraphIID<ErdosRenyi<Ty>>>) -> impl Iterator<Item=GraphIID<Uniform, Uniform, ErdosRenyi<Ty>>> {
        graph_codecs.into_iter().map(|c| {
            GraphIID::new(c.nodes.len, c.edges.indices, Uniform::new(7), Uniform::new(3))
        })
    }

    #[test]
    fn sampled_node_labelled_shuffle_digraph() {
        test_graph_shuffle_codecs(with_sampled_symbols(node_labelled(graph_codecs::<Directed>(false), 7)), &TestConfig { pu: false, ..TestConfig::test(0) });
    }

    #[test]
    fn sampled_node_labelled_shuffle_ungraph() {
        test_graph_shuffle_codecs(with_sampled_symbols(node_labelled(graph_codecs::<Undirected>(false), 7)), &TestConfig::test(0));
    }

    #[test]
    fn sampled_edge_labelled_shuffle_digraph() {
        for loops in [false, true] {
            test_graph_shuffle_codecs(with_sampled_symbols(edge_labelled(graph_codecs::<Directed>(loops))), &TestConfig { pu: false, ..TestConfig::test(0) });
        }
    }

    #[test]
    fn test_dense_edge_indices_codec() {
        for (c, x) in with_sampled_symbols(edge_labelled(graph_codecs::<Directed>(false))) {
            c.edges.indices.test(&Unordered(x.edge_indices()), 0);
        }
    }

    #[test]
    fn test_edges_iid() {
        for (c, x) in with_sampled_symbols(edge_labelled(graph_codecs::<Directed>(false))) {
            c.edges.test(&Unordered(x.edges()), 0);
        }
    }

    #[test]
    fn sampled_edge_labelled_shuffle_ungraph() {
        let codec_and_graphs = with_sampled_symbols(edge_labelled(graph_codecs::<Undirected>(false))).into_iter().collect_vec();
        for (_, g) in &codec_and_graphs {
            assert!(!g.has_selfloops());
        }
        test_graph_shuffle_codecs(codec_and_graphs, &TestConfig::test(0));
    }

    #[test]
    fn no_traces_issue() {
        with_automorphisms_backend(AutomorphismsBackend::Traces, || {
            let g = NodeLabelledGraph::<Undirected>::new(
                [0, 1, 1, 0, 1, 1],
                [(0, 1), (1, 2), (3, 4), (4, 5)].map(|e| (e, ())));
            let canon_graph = g.canonized();
            for seed in 0..50 {
                assert_eq!(canon_graph, g.shuffled(seed).canonized());
            }
        })
    }
}

impl<N: EqSymbol + Default, E: EqSymbol, Ty: EdgeType> AutCanonizable for ChunkedPrefix<GraphPrefix<N, E, Ty>>
where
    GraphPrefix<N, E, Ty>: AutCanonizable,
{
    fn automorphisms(&self) -> Automorphisms {
        let mut prefix = GraphPrefixingChain::new().prefix(self.inner.graph.clone());
        prefix.len = self.len;
        prefix.automorphisms()
    }
}
