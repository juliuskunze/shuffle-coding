use itertools::Itertools;
use petgraph::Directed;

use crate::graph::{AsEdges, EdgeType, Graph, NeighborMap, PartialGraph};
use crate::permutable::{Automorphisms, GroupPermutable, GroupPermutableFromFused, Orbit, Orbits, Permutable, Permutation, PermutationGroup};

/// A graph with natural numbers as node labels.
pub type NodeLabelledGraph<Ty = Directed> = Graph<usize, (), Ty>;
/// A graph with natural numbers as edge and node labels.
pub type EdgeLabelledGraph<Ty = Directed> = Graph<usize, usize, Ty>;

impl<Ty: EdgeType> GroupPermutableFromFused for NodeLabelledGraph<Ty> where NodeLabelledGraph<Ty>: AsEdges<usize> {
    fn auts(&self) -> Automorphisms {
        self.unlabelled_automorphisms(Some(self.node_labels().collect_vec().orbits()))
    }
}

impl<Ty: EdgeType> EdgeLabelledGraph<Ty> where EdgeLabelledGraph<Ty>: AsEdges<usize, usize>, NodeLabelledGraph<Ty>: AsEdges<usize> {
    /// Equivalent graph where labelled edges are converted into labelled nodes.
    pub fn with_edges_as_nodes(&self) -> NodeLabelledGraph<Ty> {
        let edge_label_shift = self.node_labels().max().unwrap_or(0) + 1;
        let mut extended = NodeLabelledGraph::new(self.node_labels(), vec![]);
        for ((i, j), e) in self.edges() {
            let is_undirected = self.edge((i, j)) == self.edge((j, i));
            if is_undirected && i > j {
                continue;
            }
            let new_node = extended.len();
            extended.nodes.push((edge_label_shift + e, NeighborMap::new()));

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

    fn truncate_permutation(&self, p: &Permutation) -> Permutation {
        Permutation { len: self.len(), indices: p.indices.clone().into_iter().filter(|(k, i)| {
            let keep = k < &self.len();
            assert_eq!(keep, i < &self.len());
            keep
        }).collect() }
    }

    fn truncated_orbits(&self, orbits: Orbits) -> Vec<Orbit> {
        orbits.into_iter().filter(|o| {
            let keep = o.iter().all(|&i| i < self.len());
            assert!(keep || !o.iter().any(|&i| i < self.len()));
            keep
        }).collect()
    }
}

impl<Ty: EdgeType> GroupPermutableFromFused for EdgeLabelledGraph<Ty> where EdgeLabelledGraph<Ty>: AsEdges<usize, usize>, NodeLabelledGraph<Ty>: AsEdges<usize> {
    fn auts(&self) -> Automorphisms {
        let Automorphisms { group, canon, orbits, bits, .. } = self.with_edges_as_nodes().automorphisms();
        let group = PermutationGroup::new(self.len(), group.
            generators.into_iter().map(|g| self.truncate_permutation(&g)).collect_vec());
        let canon = self.truncate_permutation(&canon);
        let decanon = canon.inverse();
        let orbits = self.truncated_orbits(orbits);
        Automorphisms { group, canon, decanon, orbits, bits }
    }
}

pub type PartialEdgeLabelledGraph<Ty = Directed> = PartialGraph<usize, usize, Ty>;

impl<Ty: EdgeType> GroupPermutableFromFused for PartialEdgeLabelledGraph<Ty> where Graph<Option<usize>, usize, Ty>: AsEdges<Option<usize>, usize>, EdgeLabelledGraph<Ty>: AsEdges<usize, usize>, NodeLabelledGraph<Ty>: AsEdges<usize> {
    fn auts(&self) -> Automorphisms {
        let graph = self.with_unknown_nodes_uniquely_labelled();
        let mut a = graph.automorphisms();
        a.group.adjust_len(self.len());
        a
    }
}

impl<Ty: EdgeType> PartialEdgeLabelledGraph<Ty> where Graph<Option<usize>, usize, Ty>: AsEdges<Option<usize>, usize>, EdgeLabelledGraph<Ty>: AsEdges<usize, usize> {
    fn with_unknown_nodes_uniquely_labelled(&self) -> EdgeLabelledGraph<Ty> {
        let node_label_shift = self.graph.node_labels().filter_map(|n| n).
            into_iter().max().unwrap_or(0) + 1;
        EdgeLabelledGraph::from_neighbors(self.graph.nodes.iter().enumerate().map(
            |(i, (n, ne))| (
                if self.unknown_nodes().contains(&i) {
                    node_label_shift + i - self.unknown_nodes().start
                } else { n.clone().unwrap() }, ne.clone())))
    }
}

#[cfg(test)]
pub mod tests {
    use petgraph::Undirected;

    use crate::ans::{assert_bits_eq, Codec, Message, Symbol, Uniform, UniformCodec};
    use crate::graph::{AsEdges, EdgeType, PlainGraph, UnGraph};
    use crate::graph_ans::{EmptyCodec, ErdosRenyi, GraphIID};
    use crate::graph_ans::tests::{graph_codecs, plain_erdos_renyi, PlainGraphIID, test_graph_shuffle_codecs, UnGraphIID, with_sampled_symbols};
    use crate::multiset::OrdSymbol;
    use crate::permutable::{GroupPermutable, Unordered};
    use crate::shuffle_ans::{PermutationUniform, TestConfig};

    use super::*;

    pub fn node_labelled<Ty: EdgeType, EdgeC: Codec>(
        graph_codecs: impl IntoIterator<Item=GraphIID<EmptyCodec, EdgeC, ErdosRenyi<Ty>>>, num_labels: usize,
    ) -> impl Iterator<Item=GraphIID<Uniform, EdgeC, ErdosRenyi<Ty>>> where
        EdgeC::Symbol: OrdSymbol,
        Graph<(), EdgeC::Symbol, Ty>: AsEdges<(), EdgeC::Symbol>,
        Graph<usize, EdgeC::Symbol, Ty>: AsEdges<usize, EdgeC::Symbol> {
        graph_codecs.into_iter().map(move |c| GraphIID::new(
            c.nodes.len, c.edges.indices, Uniform::new(num_labels), c.edges.label))
    }

    pub fn node_labelled_codec_for(x: &UnGraph<usize, impl Symbol>, edge_prob: f64, loops: bool) -> UnGraphIID<Uniform> {
        let num_labels = x.node_labels().max().unwrap() + 1;
        let c = plain_erdos_renyi(x.len(), edge_prob, loops);
        UnGraphIID::new(c.nodes.len, c.edges.indices, Uniform::new(num_labels), c.edges.label)
    }

    pub fn with_edge_labelled_uniform(graphs: impl IntoIterator<Item=EdgeLabelledGraph<Undirected>>, edge_prob: f64, loops: bool) -> impl Iterator<Item=(UnGraphIID<Uniform, Uniform>, EdgeLabelledGraph<Undirected>)>
        where UnGraph<usize>: AsEdges<usize> {
        graphs.into_iter().map(move |x| {
            let c = node_labelled_codec_for(&x, edge_prob, loops);
            let num_labels = x.edge_labels().into_iter().max().unwrap_or(0) + 1;
            (GraphIID::new(c.nodes.len, c.edges.indices, c.nodes.item, Uniform::new(num_labels)), x)
        })
    }

    pub fn with_node_labelled_uniform(graphs: impl IntoIterator<Item=NodeLabelledGraph<Undirected>>, edge_prob: f64, loops: bool) -> impl Iterator<Item=(UnGraphIID<Uniform>, NodeLabelledGraph<Undirected>)> where {
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
        assert_bits_eq(PermutationUniform { len: 3 }.uni_bits(), bh3o3.automorphisms().bits);
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

    pub fn edge_labelled<Ty: EdgeType>(graph_codecs: impl IntoIterator<Item=PlainGraphIID<ErdosRenyi<Ty>>>) -> impl Iterator<Item=GraphIID<Uniform, Uniform, ErdosRenyi<Ty>>>
        where PlainGraph<Ty>: AsEdges, EdgeLabelledGraph<Ty>: AsEdges<usize, usize> {
        graph_codecs.into_iter().map(|c| {
            GraphIID::new(c.nodes.len, c.edges.indices, Uniform::new(7), Uniform::new(3))
        })
    }

    #[test]
    fn sampled_node_labelled_shuffle_digraph() {
        test_graph_shuffle_codecs(with_sampled_symbols(node_labelled(graph_codecs::<Directed>(false), 7)), &TestConfig::test(0));
    }

    #[test]
    fn sampled_node_labelled_shuffle_ungraph() {
        test_graph_shuffle_codecs(with_sampled_symbols(node_labelled(graph_codecs::<Undirected>(false), 7)), &TestConfig::test(0));
    }

    #[test]
    fn sampled_edge_labelled_shuffle_digraph() {
        for loops in [false, true] {
            test_graph_shuffle_codecs(with_sampled_symbols(edge_labelled(graph_codecs::<Directed>(loops))), &TestConfig::test(0));
        }
    }

    #[test]
    fn test_dense_edge_indices_codec() {
        for (c, x) in with_sampled_symbols(edge_labelled(graph_codecs::<Directed>(false))) {
            c.edges.indices.test(&Unordered(x.edge_indices()), &Message::random(0));
        }
    }

    #[test]
    fn test_edges_iid() {
        for (c, x) in with_sampled_symbols(edge_labelled(graph_codecs::<Directed>(false))) {
            c.edges.test(&Unordered(x.edges()), &Message::random(0));
        }
    }

    #[test]
    fn sampled_edge_labelled_ungraph_shuffle() {
        let codec_and_graphs = with_sampled_symbols(edge_labelled(graph_codecs::<Undirected>(false))).into_iter().collect_vec();
        for (_, g) in &codec_and_graphs {
            assert!(!g.has_selfloops());
        }
        test_graph_shuffle_codecs(codec_and_graphs, &TestConfig::test(0));
    }

    #[test]
    fn no_traces_issue() {
        let g = NodeLabelledGraph::<Undirected>::new(
            [0, 1, 1, 0, 1, 1],
            [(0, 1), (1, 2), (3, 4), (4, 5)].map(|e| (e, ())));
        let canon_graph = g.canonized().0;
        for seed in 0..50 {
            assert_eq!(canon_graph, g.shuffled(seed).canonized().0);
        }
    }
}
