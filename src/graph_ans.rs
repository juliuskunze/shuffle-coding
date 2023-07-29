use std::collections::HashSet;
use std::fmt::Debug;
use std::iter::repeat;
use std::marker::PhantomData;

use itertools::Itertools;
use petgraph::Directed;

use crate::ans::{Bernoulli, Codec, ConstantCodec, IID, Message, MutDistribution, MutCategorical};
use crate::graph::{AsEdges, Edge, EdgeIndex, EdgeType, Graph, PlainGraph};
use crate::multiset::{Multiset, OrdSymbol};
use crate::permutable::Unordered;
use crate::shuffle_ans::{RCosetUniform, shuffle_codec, ShuffleCodec};

pub type EmptyCodec = ConstantCodec<()>;

/// Coding edge indices independently of the i.i.d. node and edge labels.
#[derive(Clone, Debug)]
pub struct GraphIID<
    NodeC: Codec = EmptyCodec,
    EdgeC: Codec = EmptyCodec,
    IndicesC: EdgeIndicesCodec = ErdosRenyi<Directed>>
    where EdgeC::Symbol: OrdSymbol, Graph<NodeC::Symbol, EdgeC::Symbol, IndicesC::Ty>: AsEdges<NodeC::Symbol, EdgeC::Symbol>
{
    pub nodes: IID<NodeC>,
    pub edges: EdgesIID<EdgeC, IndicesC>,
}

impl<NodeC: Codec, EdgeC: Codec, IndicesC: EdgeIndicesCodec<Ty=Ty>, Ty: EdgeType> Codec for GraphIID<NodeC, EdgeC, IndicesC>
    where EdgeC::Symbol: OrdSymbol, Graph<NodeC::Symbol, EdgeC::Symbol, Ty>: AsEdges<NodeC::Symbol, EdgeC::Symbol>
{
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

impl<NodeC: Codec, EdgeC: Codec, IndicesC: EdgeIndicesCodec<Ty=Ty>, Ty: EdgeType> GraphIID<NodeC, EdgeC, IndicesC>
    where EdgeC::Symbol: OrdSymbol, Graph<NodeC::Symbol, EdgeC::Symbol, Ty>: AsEdges<NodeC::Symbol, EdgeC::Symbol> {
    pub fn new(num_nodes: usize, edge_indices: IndicesC, node: NodeC, edge: EdgeC) -> Self {
        Self { nodes: IID::new(node, num_nodes), edges: EdgesIID::new(edge_indices, edge) }
    }
}

/// petgraph::Directed/Undirected doesn't impl Eq, workaround:
impl<NodeC: Codec + PartialEq, EdgeC: Codec + PartialEq, IndicesC: EdgeIndicesCodec<Ty=Ty> + PartialEq, Ty: EdgeType> PartialEq for GraphIID<NodeC, EdgeC, IndicesC>
    where EdgeC::Symbol: OrdSymbol, Graph<NodeC::Symbol, EdgeC::Symbol, Ty>: AsEdges<NodeC::Symbol, EdgeC::Symbol> {
    fn eq(&self, other: &Self) -> bool {
        self.nodes == other.nodes && self.edges == other.edges
    }
}

impl<NodeC: Codec + Eq, EdgeC: Codec + Eq, IndicesC: EdgeIndicesCodec<Ty=Ty> + Eq, Ty: EdgeType> Eq for GraphIID<NodeC, EdgeC, IndicesC> where EdgeC::Symbol: OrdSymbol, Graph<NodeC::Symbol, EdgeC::Symbol, Ty>: AsEdges<NodeC::Symbol, EdgeC::Symbol> {}

/// Coding edge indices independently of the i.i.d. labels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EdgesIID<EdgeC: Codec = EmptyCodec, IndicesC: EdgeIndicesCodec = ErdosRenyi<Directed>> {
    pub indices: IndicesC,
    pub label: EdgeC,
}

impl<EdgeC: Codec, IndicesC: EdgeIndicesCodec<Ty=Ty>, Ty: EdgeType> Codec for EdgesIID<EdgeC, IndicesC> where EdgeC::Symbol: OrdSymbol {
    type Symbol = Multiset<Edge<EdgeC::Symbol>>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let (indices, labels) = Self::split(x);
        self.labels(indices.len()).push(m, &labels);
        self.indices.push(m, &indices);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let indices = self.indices.pop(m).canonized();
        let labels = self.labels(indices.len()).pop(m);
        Unordered(indices.into_iter().zip_eq(labels.into_iter()).collect_vec())
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let (indices, edges) = Self::split(x);
        let labels_codec = self.labels(indices.len());
        Some(self.indices.bits(&indices)? + labels_codec.bits(&edges)?)
    }
}

impl<EdgeC: Codec, IndicesC: EdgeIndicesCodec<Ty=Ty>, Ty: EdgeType> EdgesIID<EdgeC, IndicesC> {
    fn split(x: &Multiset<Edge<<EdgeC as Codec>::Symbol>>) -> (Multiset<EdgeIndex>, Vec<<EdgeC as Codec>::Symbol>) {
        let (indices, labels) = x.to_ordered().iter().sorted_by_key(|(i, _)| *i).cloned().unzip();
        (Unordered(indices), labels)
    }

    fn labels(&self, len: usize) -> IID<EdgeC> {
        IID::new(self.label.clone(), len)
    }

    pub fn new(indices: IndicesC, label: EdgeC) -> Self {
        Self { indices, label }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseSetIID<S: OrdSymbol> {
    pub alphabet: Vec<S>,
    pub contains: IID<Bernoulli>,
}

impl<S: OrdSymbol> Codec for DenseSetIID<S> {
    type Symbol = Multiset<S>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.contains.push(m, &self.dense(x));
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let dense = self.contains.pop(m);
        Unordered(self.alphabet.iter().zip_eq(dense).filter_map(|(i, b)| if b { Some(i.clone()) } else { None }).collect_vec())
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.contains.bits(&self.dense(x))
    }
}

impl<S: OrdSymbol> DenseSetIID<S> {
    pub fn new(contains: Bernoulli, alphabet: Vec<S>) -> Self {
        Self { contains: IID::new(contains, alphabet.len()), alphabet }
    }

    fn dense(&self, x: &Multiset<S>) -> Vec<bool> {
        let mut x = x.to_ordered().iter().cloned().collect::<HashSet<_>>();
        let as_vec = self.alphabet.iter().map(|i| x.remove(i)).collect_vec();
        assert!(x.is_empty());
        as_vec
    }
}

pub trait EdgeIndicesCodec: Codec<Symbol=Multiset<EdgeIndex>> {
    type Ty: EdgeType;

    fn loops(&self) -> bool;
}

#[derive(Clone, Debug)]
pub struct ErdosRenyi<Ty> {
    pub dense: DenseSetIID<EdgeIndex>,
    pub loops: bool,
    pub phantom: PhantomData<Ty>,
}

impl<Ty: EdgeType> Codec for ErdosRenyi<Ty> {
    type Symbol = Multiset<EdgeIndex>;
    fn push(&self, m: &mut Message, x: &Self::Symbol) { self.dense.push(m, x); }
    fn pop(&self, m: &mut Message) -> Self::Symbol { self.dense.pop(m) }
    fn bits(&self, x: &Self::Symbol) -> Option<f64> { self.dense.bits(x) }
}

impl<Ty: EdgeType> EdgeIndicesCodec for ErdosRenyi<Ty> {
    type Ty = Ty;

    fn loops(&self) -> bool { self.loops }
}

/// petgraph::Directed/Undirected doesn't impl Eq, workaround:
impl<Ty: EdgeType> PartialEq for ErdosRenyi<Ty> {
    fn eq(&self, other: &Self) -> bool {
        self.dense == other.dense && self.loops == other.loops
    }
}

impl<Ty: EdgeType> Eq for ErdosRenyi<Ty> {}

pub fn erdos_renyi_indices<Ty: EdgeType>(num_nodes: usize, edge: Bernoulli, loops: bool) -> ErdosRenyi<Ty> {
    ErdosRenyi { dense: DenseSetIID::new(edge, all_edge_indices::<Ty>(num_nodes, loops)), loops, phantom: Default::default() }
}

fn all_edge_indices<Ty: EdgeType>(num_nodes: usize, loops: bool) -> Vec<EdgeIndex> {
    let mut out = if loops { (0..num_nodes).map(|i| (i, i)).collect_vec() } else { vec![] };
    let fwd = (0..num_nodes).flat_map(|j| (0..j).map(move |i| (i, j)));
    if Ty::is_directed() {
        out.extend(fwd.flat_map(|(i, j)| [(i, j), (j, i)]));
    } else {
        out.extend(fwd);
    }
    out
}

pub fn num_all_edge_indices<Ty: EdgeType>(num_nodes: usize, loops: bool) -> usize {
    (if loops { num_nodes } else { 0 }) + (num_nodes * num_nodes - num_nodes) / if Ty::is_directed() { 1 } else { 2 }
}

#[derive(Clone, Debug)]
struct PolyaUrnEdgeIndexCodec<Ty: EdgeType> where PlainGraph<Ty>: AsEdges {
    graph: PlainGraph<Ty>,
    codec: MutCategorical,
    loops: bool,
    redraws: bool,
}

impl<Ty: EdgeType> PolyaUrnEdgeIndexCodec<Ty> where PlainGraph<Ty>: AsEdges {
    fn new(graph: PlainGraph<Ty>, loops: bool, redraws: bool) -> Self {
        assert!(!Ty::is_directed());
        if !loops {
            assert!(!graph.has_selfloops());
        }
        let masses = graph.nodes.iter().map(|(_, es)| es.len() + 1).enumerate();
        let codec = MutCategorical::new(masses, true);
        Self { graph, codec, loops, redraws }
    }

    fn codec_without(&self, node: usize) -> MutCategorical {
        let mut codec = self.codec.clone();
        if !self.loops {
            codec.remove_all(&node);
        }
        if !self.redraws {
            for (&neighbor, _) in &self.graph.nodes[node].1 {
                codec.remove_all(&neighbor);
            }
        }
        codec
    }

    fn remove_edge(&mut self, (i, j): (usize, usize)) {
        assert!(self.graph.remove_plain_edge((i, j)));
        self.codec.remove(&i, 1);
        if i != j {
            self.codec.remove(&j, 1);
        }
    }

    fn insert_edge(&mut self, (i, j): (usize, usize)) {
        assert!(self.graph.insert_plain_edge((i, j)));
        self.codec.insert(i, 1);
        if i != j {
            self.codec.insert(j, 1);
        }
    }
}

impl<Ty: EdgeType> Codec for PolyaUrnEdgeIndexCodec<Ty> where PlainGraph<Ty>: AsEdges {
    type Symbol = VecEdgeIndex;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        if let [node, node_] = x[..] {
            self.codec_without(node).push(m, &node_);
            self.codec.push(m, &node);
        } else {
            panic!("Expected two nodes, got {:?}", x)
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let node = self.codec.pop(m);
        let node_ = self.codec_without(node).pop(m);
        vec![node, node_]
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        if let [node, node_] = x[..] {
            Some(self.codec.bits(&node)? + self.codec_without(node).bits(&node_)?)
        } else {
            panic!("Expected two nodes, got {:?}", x)
        }
    }
}

/// Reuse Vec's implementation of Permutable for undirected edges of a Polya Urn.
type VecEdgeIndex = Vec<usize>;

#[derive(Clone, Debug)]
pub struct PolyaUrnEdgeIndicesCodec<Ty: EdgeType> where PlainGraph<Ty>: AsEdges {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub loops: bool,
    pub redraws: bool,
    pub phantom: PhantomData<Ty>,
}

impl<Ty: EdgeType> PolyaUrnEdgeIndicesCodec<Ty> where PlainGraph<Ty>: AsEdges {
    pub fn new(num_nodes: usize, num_edges: usize, loops: bool, redraws: bool) -> Self {
        Self { num_nodes, num_edges, loops, redraws, phantom: Default::default() }
    }

    fn edge_codec(&self, graph: PlainGraph<Ty>) -> ShuffleCodec<PolyaUrnEdgeIndexCodec<Ty>, impl Fn(&VecEdgeIndex) -> RCosetUniform + Clone> {
        shuffle_codec(PolyaUrnEdgeIndexCodec::new(graph, self.loops, self.redraws))
    }
}

impl<Ty: EdgeType> Codec for PolyaUrnEdgeIndicesCodec<Ty> where PlainGraph<Ty>: AsEdges {
    type Symbol = Vec<EdgeIndex>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        assert_eq!(x.len(), self.num_edges);
        let graph = PlainGraph::<Ty>::new(repeat(()).take(self.num_nodes), x.iter().map(|i| (i.clone(), ())));
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
            let edge = edge_codec.pop(m);
            if let [i, j] = edge.to_ordered()[..] {
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

impl<Ty: EdgeType> EdgeIndicesCodec for PolyaUrn<Ty> where PlainGraph<Ty>: AsEdges {
    type Ty = Ty;

    fn loops(&self) -> bool { self.ordered.loops }
}

pub fn polya_urn<Ty: EdgeType>(num_nodes: usize, num_edges: usize, loops: bool, redraws: bool) -> PolyaUrn<Ty> where PlainGraph<Ty>: AsEdges {
    shuffle_codec(PolyaUrnEdgeIndicesCodec::new(num_nodes, num_edges, loops, redraws))
}

pub type PolyaUrn<Ty> = ShuffleCodec<PolyaUrnEdgeIndicesCodec<Ty>, fn(&Vec<EdgeIndex>) -> RCosetUniform>;

#[cfg(test)]
pub mod tests {
    use petgraph::Undirected;

    use crate::ans::{BoxCodec, EnumCodec, OptionCodec};
    use crate::graph::{DiGraph, PartialGraph, UnGraph};
    use crate::permutable::{GroupPermutable, Permutable};
    use crate::shuffle_ans::interleaved::{InterleavedShuffleCodec, OrbitPartitioner};
    use crate::shuffle_ans::test::{test_and_print_vec, test_shuffle_codecs};
    use crate::shuffle_ans::TestConfig;

    use super::*;

    pub type UnGraphIID<NodeC = EmptyCodec, EdgeC = EmptyCodec, IndicesC = ErdosRenyi<Undirected>> = GraphIID<NodeC, EdgeC, IndicesC>;
    pub type PlainGraphIID<IndicesC = ErdosRenyi<Directed>> = GraphIID<EmptyCodec, EmptyCodec, IndicesC>;

    pub fn plain_erdos_renyi<Ty: EdgeType>(n: usize, p: f64, loops: bool) -> PlainGraphIID<ErdosRenyi<Ty>> where PlainGraph<Ty>: AsEdges<(), ()> {
        let norm = 1 << 28;
        let mass = (p * norm as f64) as usize;
        let edge_indices = erdos_renyi_indices(n, Bernoulli::new(mass, norm), loops);
        GraphIID::new(n, edge_indices, EmptyCodec::default(), EmptyCodec::default())
    }

    pub fn test_graph_shuffle_codecs<NodeC: Codec, EdgeC: Codec, Ty: EdgeType>(
        codecs_and_graphs: impl IntoIterator<Item=(GraphIID<NodeC, EdgeC, ErdosRenyi<Ty>>, Graph<NodeC::Symbol, EdgeC::Symbol, Ty>)>,
        config: &TestConfig) where
        NodeC::Symbol: OrdSymbol, EdgeC::Symbol: OrdSymbol,
        <GraphIID<NodeC, EdgeC, ErdosRenyi<Ty>> as Codec>::Symbol: GroupPermutable,
        PartialGraph<NodeC::Symbol, EdgeC::Symbol, Ty>: GroupPermutable,
        Graph<NodeC::Symbol, EdgeC::Symbol, Ty>: AsEdges<NodeC::Symbol, EdgeC::Symbol>,
        Graph<Option<NodeC::Symbol>, EdgeC::Symbol, Ty>: AsEdges<Option<NodeC::Symbol>, EdgeC::Symbol>
    {
        let (codecs, graphs): (Vec<_>, Vec<_>) = codecs_and_graphs.into_iter().unzip();
        let unordered = &graphs.iter().map(|x| Unordered(x.clone())).collect();
        test_shuffle_codecs(&codecs, unordered, config);
        let slice_codec_fn = |c: GraphIID<NodeC, EdgeC, ErdosRenyi<Ty>>| move |len| {
            let edge = OptionCodec { is_some: c.edges.indices.dense.contains.item.clone(), some: c.edges.label.clone() };
            let dense_edges = IID::new(edge.clone(), len * if Ty::is_directed() { 2 } else { 1 });
            let selfloop = if c.edges.indices.loops() { EnumCodec::A(BoxCodec::new(edge)) } else { EnumCodec::B(ConstantCodec(None)) };
            (c.nodes.item.clone(), (dense_edges, selfloop))
        };

        if config.interleaved {
            let interleaved_codecs = codecs.clone().into_iter().map(|c| InterleavedShuffleCodec::<PartialGraph<_, _, _>, _, _, _>::new(c.nodes.len, slice_codec_fn(c), OrbitPartitioner));
            test_and_print_vec(interleaved_codecs, unordered, &config.initial_message());
        }

        println!();
    }

    pub fn graph_codecs<Ty: EdgeType>(loops: bool) -> impl Iterator<Item=PlainGraphIID<ErdosRenyi<Ty>>> where PlainGraph<Ty>: AsEdges<(), ()> {
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

    fn small_ungraphs() -> impl IntoIterator<Item=UnGraph> {
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
        ]
    }

    pub fn with_sampled_symbols<C: Codec>(codecs: impl IntoIterator<Item=C>) -> impl IntoIterator<Item=(C, C::Symbol)> {
        codecs.into_iter().enumerate().map(|(seed, c)| {
            let symbol = c.sample(seed);
            (c, symbol)
        })
    }

    #[test]
    fn small_shuffle_digraphs() {
        for seed in 0..20 {
            test_graph_shuffle_codecs(small_codecs_and_digraphs(), &TestConfig::test(seed));
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
        test_graph_shuffle_codecs(with_sampled_symbols(graph_codecs::<Directed>(false)), &TestConfig::test(0));
    }

    #[test]
    fn sampled_shuffle_ungraphs() {
        test_graph_shuffle_codecs(with_sampled_symbols(graph_codecs::<Undirected>(false)), &TestConfig::test(0));
    }

    #[test]
    fn fast_shuffle_coding_stochastic() {
        let graph = UnGraph::plain(8, [(0, 1), (0, 4), (0, 7), (1, 5), (1, 6), (2, 6), (3, 4), (3, 5), (4, 7)]);
        let codec = plain_erdos_renyi::<Undirected>(8, 0.4, false);
        test_graph_shuffle_codecs(vec![(codec, graph)], &TestConfig::test(0));
    }

    #[test]
    fn fast_shuffle_coding_no_iter() {
        for half_iter in [false, true] {
            for wl_iter in [0, 1, 2] {
                let graph = UnGraph::plain(8, [(0, 1), (0, 4), (0, 7), (1, 5), (1, 6), (2, 6), (3, 4), (3, 5), (4, 7)]);
                let codec = plain_erdos_renyi::<Undirected>(graph.len(), 0.4, false);
                let mut config = TestConfig::test(0);
                config.blockwise = false;
                config.coset_interleaved = false;
                config.interleaved = false;
                config.wl_extra_half_iter = half_iter;
                config.wl_iter = wl_iter;
                test_graph_shuffle_codecs(vec![(codec, graph)], &config);
            }
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