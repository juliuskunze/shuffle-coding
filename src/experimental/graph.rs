//! Experimental codecs for graphs. These implementations do not scale well with graph size.

use crate::autoregressive::graph::{GraphPrefix, GraphPrefixingChain};
use crate::autoregressive::UncachedPrefixFn;
use crate::codec::{Bernoulli, Codec, ConstantCodec, Message, MutDistribution, OrdSymbol, Symbol, IID};
use crate::graph::{Directed, EdgeIndex, EdgeType, Graph, PlainGraph};
use crate::joint::multiset::{joint_multiset_shuffle_codec, MultisetShuffleCodec};
use crate::multiset::Multiset;
use crate::permutable::graph::{EdgeIndicesCodec, EdgesIID};
use crate::permutable::{FHashSet, Len, Permutable, Unordered};
use ftree::FenwickTree;
use itertools::Itertools;
use std::cell::{RefCell, RefMut};
use std::fmt::Debug;
use std::iter::repeat_n;
use std::marker::PhantomData;
use std::vec::IntoIter;

pub type EmptyCodec = ConstantCodec<()>;

/// Coding edge indices independently of the i.i.d. node and edge labels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphIID<
    NodeC: Codec = EmptyCodec,
    EdgeC: Codec<Symbol: OrdSymbol> = EmptyCodec,
    IndicesC: EdgeIndicesCodec = ErdosRenyi<Directed>> {
    pub nodes: IID<NodeC>,
    pub edges: EdgesIID<EdgeC, IndicesC>,
}

impl<NodeC: Codec<Symbol: Eq>, EdgeC: Codec<Symbol: OrdSymbol> + Clone, IndicesC: EdgeIndicesCodec<Ty=Ty>, Ty: EdgeType> Codec for GraphIID<NodeC, EdgeC, IndicesC> {
    type Symbol = Graph<NodeC::Symbol, EdgeC::Symbol, Ty>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.edges.push(m, &Unordered(x.edges()));
        self.nodes.push(m, &x.node_labels().collect());
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        Self::Symbol::new(self.nodes.pop(m), self.edges.pop(m).into_ordered())
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        Some(self.nodes.bits(&x.node_labels().collect())? + self.edges.bits(&Unordered(x.edges()))?)
    }
}

impl<NodeC: Codec, EdgeC: Codec<Symbol: OrdSymbol> + Clone, IndicesC: EdgeIndicesCodec<Ty=Ty>, Ty: EdgeType> GraphIID<NodeC, EdgeC, IndicesC> {
    pub fn new(num_nodes: usize, edge_indices: IndicesC, node: NodeC, edge: EdgeC) -> Self {
        Self { nodes: IID::new(node, num_nodes), edges: EdgesIID::new(edge_indices, edge) }
    }
}

pub trait Alphabet<S>: IntoIterator<Item=S> + Clone {
    fn len(&self) -> usize;
}

impl<S: OrdSymbol> Alphabet<S> for Vec<S> {
    fn len(&self) -> usize { self.len() }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseSetIID<S: OrdSymbol, A: Alphabet<S>> {
    pub alphabet: A,
    pub contains: IID<Bernoulli>,
    pub phantom: PhantomData<S>,
}

impl<S: OrdSymbol, A: Alphabet<S>> Codec for DenseSetIID<S, A> {
    type Symbol = Multiset<S>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.contains.push(m, &self.dense(x));
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let dense = self.contains.pop(m);
        Unordered(self.alphabet.clone().into_iter().zip_eq(dense).filter_map(|(i, b)| if b { Some(i.clone()) } else { None }).collect_vec())
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.contains.bits(&self.dense(x))
    }
}

impl<S: OrdSymbol, A: Alphabet<S>> DenseSetIID<S, A> {
    pub fn new(contains: Bernoulli, alphabet: A) -> Self {
        Self { contains: IID::new(contains, alphabet.len()), alphabet, phantom: PhantomData }
    }

    fn dense(&self, x: &Multiset<S>) -> Vec<bool> {
        let mut x = x.to_ordered().iter().cloned().collect::<FHashSet<_>>();
        let as_vec = self.alphabet.clone().into_iter().map(|i| x.remove(&i)).collect_vec();
        assert!(x.is_empty());
        as_vec
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ErdosRenyi<Ty: EdgeType, I: Alphabet<EdgeIndex> = AllEdgeIndices<Ty>> {
    pub dense: DenseSetIID<EdgeIndex, I>,
    pub loops: bool,
    pub phantom: PhantomData<Ty>,
}

impl<Ty: EdgeType, I: Alphabet<EdgeIndex>> Codec for ErdosRenyi<Ty, I> {
    type Symbol = Multiset<EdgeIndex>;
    fn push(&self, m: &mut Message, x: &Self::Symbol) { self.dense.push(m, x) }
    fn pop(&self, m: &mut Message) -> Self::Symbol { self.dense.pop(m) }
    fn bits(&self, x: &Self::Symbol) -> Option<f64> { self.dense.bits(x) }
}

impl<Ty: EdgeType, I: Alphabet<EdgeIndex>> EdgeIndicesCodec for ErdosRenyi<Ty, I> {
    type Ty = Ty;

    fn loops(&self) -> bool { self.loops }
}

impl<Ty: EdgeType, I: Alphabet<EdgeIndex>> ErdosRenyi<Ty, I> {
    pub fn new(edge: Bernoulli, indices: I, loops: bool) -> ErdosRenyi<Ty, I> {
        ErdosRenyi { dense: DenseSetIID::new(edge, indices), loops, phantom: PhantomData }
    }
}

pub fn erdos_renyi_indices<Ty: EdgeType>(num_nodes: usize, edge: Bernoulli, loops: bool) -> ErdosRenyi<Ty> {
    ErdosRenyi::new(edge, AllEdgeIndices::<Ty> { num_nodes, loops, phantom: PhantomData }, loops)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AllEdgeIndices<Ty: EdgeType> {
    pub num_nodes: usize,
    pub loops: bool,
    pub phantom: PhantomData<Ty>,
}

impl<Ty: EdgeType> IntoIterator for AllEdgeIndices<Ty> {
    type Item = EdgeIndex;
    type IntoIter = IntoIter<EdgeIndex>;

    fn into_iter(self) -> Self::IntoIter {
        let num_nodes = self.num_nodes;
        let loops = self.loops;
        let mut out = if loops { (0..num_nodes).map(|i| (i, i)).collect_vec() } else { vec![] };
        let fwd = (0..num_nodes).flat_map(|j| (0..j).map(move |i| (i, j)));
        if Ty::is_directed() {
            out.extend(fwd.flat_map(|(i, j)| [(i, j), (j, i)]));
        } else {
            out.extend(fwd);
        }
        out.into_iter()
    }
}

impl<Ty: EdgeType> Alphabet<EdgeIndex> for AllEdgeIndices<Ty> {
    fn len(&self) -> usize { num_all_edge_indices::<Ty>(self.num_nodes, self.loops) }
}

pub fn num_all_edge_indices<Ty: EdgeType>(num_nodes: usize, loops: bool) -> usize {
    (if loops { num_nodes } else { 0 }) + (num_nodes * num_nodes - num_nodes) / if Ty::is_directed() { 1 } else { 2 }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolyaUrnEdgeIndexCodec<Ty: EdgeType> {
    graph: PlainGraph<Ty>,
    codec: RefCell<FenwickTree<usize>>,
    loops: bool,
    redraws: bool,
}

impl<Ty: EdgeType> PolyaUrnEdgeIndexCodec<Ty> {
    fn new(graph: PlainGraph<Ty>, loops: bool, redraws: bool) -> Self {
        assert!(!Ty::is_directed());
        if !loops {
            assert!(!graph.has_selfloops());
        }
        let masses = graph.nodes.iter().map(|(_, es)| es.len() + 1);
        let codec = RefCell::new(FenwickTree::from_iter(masses));
        Self { graph, codec, loops, redraws }
    }

    fn use_codec_without<R>(&self, node: usize, mut body: impl FnMut(RefMut<FenwickTree<usize>>) -> R) -> R {
        let mut codec = self.codec.borrow_mut();
        let mut excluded = vec![];
        if !self.loops {
            excluded.push((node, codec.remove_all(&node)));
        }
        if !self.redraws {
            for (&neighbor, _) in &self.graph.nodes[node].1 {
                excluded.push((neighbor, codec.remove_all(&neighbor)));
            }
        }
        let result = body(codec);
        for (node, mass) in excluded {
            self.codec.borrow_mut().insert(node, mass)
        }
        result
    }

    fn remove_edge(&mut self, (i, j): (usize, usize)) {
        assert!(self.graph.remove_plain_edge((i, j)));
        let codec = self.codec.get_mut();
        codec.remove(&i, 1);
        if i != j {
            codec.remove(&j, 1);
        }
    }

    fn insert_edge(&mut self, (i, j): (usize, usize)) {
        assert!(self.graph.insert_plain_edge((i, j)));
        let codec = self.codec.get_mut();
        codec.insert(i, 1);
        if i != j {
            codec.insert(j, 1);
        }
    }
}

impl<Ty: EdgeType> Codec for PolyaUrnEdgeIndexCodec<Ty> {
    type Symbol = VecEdgeIndex;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        if let [node, node_] = x[..] {
            self.use_codec_without(node, |codec| codec.push(m, &node_));
            self.codec.push(m, &node);
        } else {
            panic!("Expected two nodes, got {:?}", x)
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let node = self.codec.pop(m);
        let node_ = self.use_codec_without(node, |codec| codec.pop(m));
        vec![node, node_]
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        if let [node, node_] = x[..] {
            Some(self.codec.bits(&node)? + self.use_codec_without(node, |codec| codec.bits(&node_))?)
        } else {
            panic!("Expected two nodes, got {:?}", x)
        }
    }
}

impl<Ty: EdgeType> Len for PolyaUrnEdgeIndexCodec<Ty> {
    fn len(&self) -> usize { 2 }
}

/// Reuse Vec's implementation of Permutable for undirected edges of a Polya Urn.
type VecEdgeIndex = Vec<usize>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolyaUrnEdgeIndicesCodec<Ty: EdgeType> {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub loops: bool,
    pub redraws: bool,
    pub phantom: PhantomData<Ty>,
}

impl<Ty: EdgeType> PolyaUrnEdgeIndicesCodec<Ty> {
    pub fn new(num_nodes: usize, num_edges: usize, loops: bool, redraws: bool) -> Self {
        Self { num_nodes, num_edges, loops, redraws, phantom: PhantomData }
    }

    fn edge_codec(&self, graph: PlainGraph<Ty>) -> MultisetShuffleCodec<usize, PolyaUrnEdgeIndexCodec<Ty>> {
        joint_multiset_shuffle_codec(PolyaUrnEdgeIndexCodec::new(graph, self.loops, self.redraws))
    }
}

impl<Ty: EdgeType> Codec for PolyaUrnEdgeIndicesCodec<Ty> {
    type Symbol = Vec<EdgeIndex>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        assert_eq!(x.len(), self.num_edges);
        let graph = PlainGraph::<Ty>::new(repeat_n((), self.num_nodes), x.iter().map(|i| (i.clone(), ())));
        let mut edge_codec = self.edge_codec(graph);
        for (i, j) in x.iter().rev() {
            edge_codec.ordered.remove_edge((*i, *j));
            edge_codec.push(m, &Unordered(vec![*i, *j]));
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let graph = Graph::plain_empty(self.num_nodes);
        let mut edge_codec = self.edge_codec(graph);
        let mut edges = vec![];
        for _i in 0..self.num_edges {
            let edge = edge_codec.pop(m).into_ordered();
            if let [i, j] = edge[..] {
                edge_codec.ordered.insert_edge((i, j));
                edges.push((i, j));
            } else {
                panic!("Expected two nodes, got {:?}", edge)
            };
        }
        edges
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        if self.loops && self.redraws {
            let m = &mut Message::zeros();
            let bits = m.virtual_bits();
            self.push(m, x);
            Some(m.virtual_bits() - bits)
        } else { None }
    }
}

impl<Ty: EdgeType> Len for PolyaUrnEdgeIndicesCodec<Ty> {
    fn len(&self) -> usize { self.num_edges }
}

pub type PolyaUrn<Ty> = MultisetShuffleCodec<EdgeIndex, PolyaUrnEdgeIndicesCodec<Ty>>;

impl<Ty: EdgeType> EdgeIndicesCodec for PolyaUrn<Ty> {
    type Ty = Ty;

    fn loops(&self) -> bool {
        self.ordered().loops
    }
}

impl<Ty: EdgeType> PolyaUrn<Ty> {
    pub fn ordered(&self) -> &PolyaUrnEdgeIndicesCodec<Ty> {
        &self.ordered
    }
}

pub fn polya_urn<Ty: EdgeType>(num_nodes: usize, num_edges: usize, loops: bool, redraws: bool) -> PolyaUrn<Ty> {
    joint_multiset_shuffle_codec(PolyaUrnEdgeIndicesCodec::new(num_nodes, num_edges, loops, redraws))
}

/// Autoregressive version of the Erdos-Renyi model for graphs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ErdosRenyiSliceCodecs<NodeC: Codec + Symbol, EdgeC: Codec<Symbol: Ord> + Symbol, Ty: EdgeType> {
    pub len: usize,
    pub has_edge: Bernoulli,
    pub node: NodeC,
    pub edge: EdgeC,
    pub loops: bool,
    pub phantom: PhantomData<Ty>,
}

impl<NodeC: Codec + Symbol, EdgeC: Codec<Symbol: Ord> + Symbol, Ty: EdgeType> Len for ErdosRenyiSliceCodecs<NodeC, EdgeC, Ty> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<NodeC: Codec<Symbol: Eq + Default> + Symbol, EdgeC: Codec<Symbol: Ord> + Symbol, Ty: EdgeType> UncachedPrefixFn<GraphPrefixingChain<NodeC::Symbol, EdgeC::Symbol, Ty>> for ErdosRenyiSliceCodecs<NodeC, EdgeC, Ty> {
    type Output = (NodeC, EdgesIID<EdgeC, ErdosRenyi<Ty, Vec<EdgeIndex>>>);

    fn apply(&self, x: &GraphPrefix<NodeC::Symbol, EdgeC::Symbol, Ty>) -> Self::Output {
        let node = x.len;
        let unknown = node + 1..x.graph.len();
        let mut slice_indices = unknown.clone().map(|j| (node, j)).collect_vec();
        if Ty::is_directed() {
            slice_indices.extend(unknown.map(|j| (j, node)));
        }
        if self.loops {
            slice_indices.push((node, node));
        }

        let indices = ErdosRenyi::<Ty, Vec<EdgeIndex>>::new(self.has_edge.clone(), slice_indices, self.loops);
        (self.node.clone(), EdgesIID::new(indices, self.edge.clone()))
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::autoregressive::chunked::{chunked_hashing_shuffle_codec, geom_chunk_sizes};
    use crate::autoregressive::graph::{GraphPrefix, GraphPrefixingChain, PolyaUrnSliceCodecs};
    use crate::autoregressive::{Autoregressive, AutoregressiveShuffleCodec};
    use crate::bench::TestConfig;
    use crate::codec::tests::test_and_print_vec;
    use crate::codec::{Symbol, MAX_SIZE};
    use crate::experimental::autoregressive::joint::JointSliceCodecs;
    use crate::experimental::complete_joint::aut::{with_isomorphism_test_max_len, AutCanonizable};
    use crate::experimental::graph::ErdosRenyiSliceCodecs;
    use crate::graph::{DiGraph, UnGraph, Undirected};
    use crate::joint::incomplete::cr_joint_shuffle_codec;
    use crate::joint::test::test_complete_joint_shuffle_codecs;
    use crate::permutable::graph::EdgeIndicesCodec;
    use crate::permutable::{Complete, Permutable, PermutableCodec};

    pub type UnGraphIID<NodeC = EmptyCodec, EdgeC = EmptyCodec, IndicesC = ErdosRenyi<Undirected>> = GraphIID<NodeC, EdgeC, IndicesC>;
    pub type PlainGraphIID<IndicesC = ErdosRenyi<Directed>> = GraphIID<EmptyCodec, EmptyCodec, IndicesC>;

    pub fn plain_erdos_renyi<Ty: EdgeType>(n: usize, p: f64, loops: bool) -> PlainGraphIID<ErdosRenyi<Ty>> {
        let edge = Bernoulli::new((p * MAX_SIZE as f64) as usize, MAX_SIZE);
        let edge_indices = erdos_renyi_indices(n, edge, loops);
        GraphIID::new(n, edge_indices, EmptyCodec::default(), EmptyCodec::default())
    }

    pub fn test_joint_shuffle_codecs<N: OrdSymbol + Default, E: OrdSymbol, Ty: EdgeType, C: PermutableCodec<Symbol=Graph<N, E, Ty>> + Symbol>(
        codecs: &Vec<C>,
        unordered: &Vec<Unordered<C::Symbol>>,
        config: &TestConfig)
    where
        Graph<N, E, Ty>: AutCanonizable,
        GraphPrefix<N, E, Ty>: AutCanonizable,
    {
        test_complete_joint_shuffle_codecs(codecs, unordered, config);

        if config.joint {
            if !config.no_incomplete && !Ty::is_directed() {
                test_and_print_vec(codecs.iter().cloned().map(|c| cr_joint_shuffle_codec(c, config.color_refinement())), unordered, config.seeds());
            }
        }

        if config.joint_ar {
            if config.complete {
                let complete = |c| AutoregressiveShuffleCodec::new(JointSliceCodecs::new(c), Complete);
                test_and_print_vec(codecs.iter().cloned().map(complete), unordered, config.seeds());
            }

            if !config.no_incomplete {
                let incomplete = |c| AutoregressiveShuffleCodec::new(JointSliceCodecs::new(c), config.color_refinement_max_local_info());
                test_and_print_vec(codecs.iter().cloned().map(incomplete), unordered, config.seeds());
            }
        }
    }

    pub fn test_graph_shuffle_codecs<NodeC: Codec<Symbol: OrdSymbol + Default> + Symbol, EdgeC: Codec<Symbol: OrdSymbol> + Symbol, Ty: EdgeType>(
        er_codecs_and_graphs: impl IntoIterator<Item=(GraphIID<NodeC, EdgeC, ErdosRenyi<Ty>>, Graph<NodeC::Symbol, EdgeC::Symbol, Ty>)>,
        config: &TestConfig)
    where
        Graph<NodeC::Symbol, EdgeC::Symbol, Ty>: AutCanonizable,
        GraphPrefix<NodeC::Symbol, EdgeC::Symbol, Ty>: AutCanonizable,
    {
        with_isomorphism_test_max_len(config.isomorphism_test_max_len, || {
            let er_and_graphs = er_codecs_and_graphs.into_iter().collect_vec();
            let (er, graphs): (Vec<_>, Vec<_>) = er_and_graphs.iter().cloned().unzip();
            let unordered = &graphs.iter().map(|x| Unordered(x.clone())).collect();

            if config.er {
                test_joint_shuffle_codecs(&er, unordered, config);
            }
            if config.pu {
                let pu = er_and_graphs.into_iter().map(|(c, x)| {
                    let edge_indices = polya_urn(x.len(), x.num_edges(), false, false);
                    GraphIID::new(x.len(), edge_indices, c.nodes.item.clone(), c.edges.label.clone())
                }).collect_vec();
                test_joint_shuffle_codecs(&pu, unordered, config);
            }

            if config.full_ar {
                let cr = config.color_refinement_max_local_info();
                if config.ae {
                    let aer = er.clone().into_iter().map(|c| Autoregressive::new(ErdosRenyiSliceCodecs {
                        len: c.nodes.len,
                        has_edge: c.edges.indices.dense.contains.item.clone(),
                        node: c.nodes.item.clone(),
                        edge: c.edges.label.clone(),
                        loops: c.edges.indices.loops(),
                        phantom: PhantomData,
                    }));
                    if config.complete {
                        let codecs = aer.clone().map(|c| AutoregressiveShuffleCodec::new(c.0.slices, Complete));
                        test_and_print_vec(codecs, unordered, config.seeds());
                    }
                    if !config.no_incomplete {
                        let codecs = aer.map(|c| AutoregressiveShuffleCodec::new(c.0.slices, cr.clone()));
                        test_and_print_vec(codecs, unordered, config.seeds());
                    }
                }
                if config.ap && !Ty::is_directed() {
                    let apu = er.clone().into_iter().map(|c| Autoregressive::new(PolyaUrnSliceCodecs {
                        len: c.nodes.len,
                        node: c.nodes.item.clone(),
                        edge: c.edges.label.clone(),
                        loops: c.edges.indices.loops(),
                        phantom: PhantomData,
                    }));
                    if config.complete {
                        let codecs = apu.clone().map(|c| AutoregressiveShuffleCodec::new(c.0.slices, Complete));
                        test_and_print_vec(codecs, unordered, config.seeds());
                    }
                    if !config.no_incomplete {
                        let codecs = apu.map(|c| AutoregressiveShuffleCodec::new(c.0.slices, cr.clone()));
                        test_and_print_vec(codecs, unordered, config.seeds());
                    }
                }
            }
            if config.ar {
                let cr = config.color_refinement();
                if config.ap && !Ty::is_directed() {
                    let apu = er.clone().into_iter().map(|c| Autoregressive::new(PolyaUrnSliceCodecs {
                        len: c.nodes.len,
                        node: c.nodes.item.clone(),
                        edge: c.edges.label.clone(),
                        loops: c.edges.indices.loops(),
                        phantom: PhantomData,
                    }));
                    if !config.no_incomplete {
                        let codecs = apu.map(|c| {
                            let chunk_sizes = geom_chunk_sizes(config.chunks, c.0.len, config.chunks_base);
                            chunked_hashing_shuffle_codec(GraphPrefixingChain::new(), c.0.slices, cr.clone(), chunk_sizes, c.0.len)
                        });
                        test_and_print_vec(codecs, unordered, config.seeds());
                    }
                }
            }
        })
    }


    pub fn graph_codecs<Ty: EdgeType>(loops: bool) -> impl Iterator<Item=PlainGraphIID<ErdosRenyi<Ty>>> {
        (0..20).map(move |len| plain_erdos_renyi::<Ty>(len, 0.4, loops))
    }

    fn small_codecs_and_digraphs() -> impl Iterator<Item=(GraphIID, Graph)> {
        small_digraphs().into_iter().map(|x| (plain_erdos_renyi(x.len(), 0.1, false), x))
    }

    pub fn small_digraphs() -> impl Iterator<Item=Graph> {
        [
            DiGraph::plain(0, vec![]),
            DiGraph::plain(1, vec![]),
            DiGraph::plain(3, vec![(0, 1)]),
            DiGraph::plain(4, vec![(0, 1), (1, 2), (2, 3), (3, 0)]),
            DiGraph::plain(11, vec![
                (0, 1), (0, 2), (0, 3),
                (1, 4), (2, 5), (3, 6),
                (4, 7), (5, 8), (6, 9),
                (7, 10), (8, 10), (9, 10)])
        ].into_iter()
    }

    fn small_codecs_and_ungraphs() -> impl Iterator<Item=(UnGraphIID, UnGraph)> {
        small_ungraphs().into_iter().map(|x| (plain_erdos_renyi(x.len(), 0.1, false), x))
    }

    fn small_ungraphs() -> impl Iterator<Item=UnGraph> {
        [
            UnGraph::plain(0, vec![]),
            UnGraph::plain(1, vec![]),
            UnGraph::plain(3, vec![(0, 1)]),
            UnGraph::plain(4, vec![(0, 1), (1, 2), (2, 3), (3, 0)]),
            UnGraph::plain(11, vec![
                (0, 1), (0, 2), (0, 3),
                (1, 4), (2, 5), (3, 6),
                (4, 7), (5, 8), (6, 9),
                (7, 10), (8, 10), (9, 10)])
        ].into_iter()
    }

    pub fn with_sampled_symbols<C: Codec>(codecs: impl IntoIterator<Item=C>) -> impl Iterator<Item=(C, C::Symbol)> {
        codecs.into_iter().enumerate().map(|(seed, c)| {
            let symbol = c.sample(seed);
            (c, symbol)
        })
    }

    #[test]
    fn small_shuffle_digraphs() {
        for seed in 0..20 {
            test_graph_shuffle_codecs(small_codecs_and_digraphs(), &TestConfig { pu: false, ..TestConfig::test(seed) });
        }
    }

    #[test]
    fn small_shuffle_ungraphs() {
        for seed in 0..20 {
            test_graph_shuffle_codecs(small_codecs_and_ungraphs(), &TestConfig::test(seed));
        }
    }

    #[test]
    fn sampled_shuffle_digraphs() {
        test_graph_shuffle_codecs(with_sampled_symbols(graph_codecs::<Directed>(false)), &TestConfig { pu: false, ..TestConfig::test(0) });
    }

    #[test]
    fn sampled_shuffle_ungraphs() {
        test_graph_shuffle_codecs(with_sampled_symbols(graph_codecs::<Undirected>(false)), &TestConfig::test(0));
    }

    #[test]
    fn incomplete_shuffle_coding_stochastic() {
        let graph = UnGraph::plain(8, [(0, 1), (0, 4), (0, 7), (1, 5), (1, 6), (2, 6), (3, 4), (3, 5), (4, 7)]);
        let codec = plain_erdos_renyi::<Undirected>(8, 0.4, false);
        test_graph_shuffle_codecs(vec![(codec, graph)], &TestConfig::test(0));
    }

    #[test]
    fn incomplete_shuffle_coding_no_iter() {
        for convs in [0, 1, 2] {
            let graph = UnGraph::plain(8, [(0, 1), (0, 4), (0, 7), (1, 5), (1, 6), (2, 6), (3, 4), (3, 5), (4, 7)]);
            let codec = plain_erdos_renyi::<Undirected>(graph.len(), 0.4, false);
            let mut config = TestConfig::test(0);
            config.joint = false;
            config.interleaved_coset_joint = false;
            config.convs = convs;
            test_graph_shuffle_codecs(vec![(codec, graph)], &config);
        }
    }

    #[test]
    fn sample_polya_urn() {
        for loops in [false, true] {
            for redraws in [false] { // Sampling with redraws not supported.
                // TODO num_nodes = 6 fails because if node is sampled that already has edges to all other nodes, we get an error:
                let c = PolyaUrnEdgeIndicesCodec::<Undirected>::new(24, 9, loops, redraws);
                c.test_on_samples(50);
            }
        }
    }
}
