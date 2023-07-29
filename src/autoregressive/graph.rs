//! Autoregressive shuffle coding for graphs. This file implements prefixes on graphs,
//! and provides some autoregressive models for graphs.
use std::marker::PhantomData;
use std::mem;
use std::ops::Range;

use crate::autoregressive::{InnerSliceCodecs, PrefixFn, PrefixingChain};
use crate::codec::{Categorical, Codec, ConstantCodec, EnumCodec, EqSymbol, Message, MutDistribution, OrdSymbol, Symbol, Uniform};
use crate::graph::{Directed, Edge, EdgeIndex, EdgeType, Graph};
use crate::joint::multiset::{joint_multiset_shuffle_codec, MultisetShuffleCodec};
use crate::multiset::Multiset;
use crate::permutable::graph::{EdgeIndicesCodec, EdgesIID};
use crate::permutable::{Len, Permutable, Unordered};
use ftree::FenwickTree;
use itertools::{Either, Itertools};
use std::cell::RefCell;
use std::iter::repeat_n;

/// Graph where the edge structure between the last num_hidden_nodes nodes is unknown.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphPrefix<N: EqSymbol = (), E: EqSymbol = (), Ty: EdgeType = Directed> {
    pub graph: Graph<N, E, Ty>,
    pub len: usize,
}

impl<N: EqSymbol, E: EqSymbol, Ty: EdgeType> Permutable for GraphPrefix<N, E, Ty> {
    fn len(&self) -> usize { self.len }

    fn swap(&mut self, i: usize, j: usize) {
        assert!(i < self.len);
        assert!(j < self.len);
        self.graph.swap(i, j);
    }
}

impl<N: EqSymbol, E: EqSymbol, Ty: EdgeType> GraphPrefix<N, E, Ty> {
    pub fn unknown_nodes(&self) -> Range<usize> {
        self.len..self.graph.len()
    }

    pub fn node_label_or_index(&self, index: usize) -> Either<&N, usize> {
        if index < self.len { Either::Left(&self.graph.nodes[index].0) } else { Either::Right(index) }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphPrefixingChain<N: EqSymbol, E: EqSymbol, Ty: EdgeType>(PhantomData<(N, E, Ty)>);

impl<N: EqSymbol + Default, E: EqSymbol, Ty: EdgeType> PrefixingChain for GraphPrefixingChain<N, E, Ty> {
    type Prefix = GraphPrefix<N, E, Ty>;
    type Full = Graph<N, E, Ty>;
    type Slice = (N, Multiset<Edge<E>>);

    fn pop_slice(&self, prefix: &mut Self::Prefix) -> Self::Slice {
        prefix.len -= 1;
        let new_unknown_node = prefix.len;
        let mut edges = prefix.graph.nodes[new_unknown_node].1.iter().
            filter(|(j, _)| prefix.unknown_nodes().contains(j)).
            map(|(j, l)| ((new_unknown_node, *j), l.clone())).
            collect_vec();
        for (e, l) in &edges {
            assert_eq!(l, &prefix.graph.remove_edge(*e).unwrap());
        }

        if Ty::is_directed() {
            // TODO OPTIMIZE Store reverse edges to avoid quadratic runtime for sparse graphs:
            edges.extend(prefix.unknown_nodes().
                filter_map(|i| {
                    let edge = (i, new_unknown_node);
                    prefix.graph.remove_edge(edge).map(|label| (edge, label))
                }));
        }

        (mem::take(&mut prefix.graph.nodes[new_unknown_node].0), Unordered(edges))
    }

    fn push_slice(&self, prefix: &mut Self::Prefix, (node, Unordered(edges)): &Self::Slice) {
        let new_known_node = prefix.len;
        let old = mem::replace(&mut prefix.graph.nodes[new_known_node].0, node.clone());
        assert_eq!(old, N::default());

        for ((i, j), e) in edges {
            assert!(prefix.unknown_nodes().contains(i));
            assert!(prefix.unknown_nodes().contains(j));
            assert!(*i == new_known_node || *j == new_known_node);
            assert!(prefix.graph.insert_edge((*i, *j), e.clone()).is_none());
        }

        prefix.len += 1;
    }

    fn prefix(&self, graph: Self::Full) -> Self::Prefix {
        let len = graph.len();
        Self::Prefix { graph, len }
    }
    fn full(&self, prefix: Self::Prefix) -> Self::Full {
        assert_eq!(prefix.graph.len(), prefix.len);
        prefix.graph
    }
}

impl<N: EqSymbol, E: EqSymbol, Ty: EdgeType> GraphPrefixingChain<N, E, Ty> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<N: EqSymbol + Default, E: EqSymbol, Ty: EdgeType, GraphS: PrefixFn<GraphPrefixingChain<N, E, Ty>, Output: Codec<Symbol=<GraphPrefixingChain<N, E, Ty> as PrefixingChain>::Slice>> + Len> InnerSliceCodecs<GraphPrefixingChain<N, E, Ty>> for GraphS {
    fn prefixing_chain(&self) -> GraphPrefixingChain<N, E, Ty> {
        GraphPrefixingChain::new()
    }

    fn empty_prefix(&self) -> impl Codec<Symbol=GraphPrefix<N, E, Ty>> {
        let graph = Graph::empty(repeat_n(N::default(), self.len()));
        ConstantCodec(GraphPrefix { graph, len: 0 })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolyaUrnSliceCodec<Ty: EdgeType = Directed> {
    /// RefCell to avoid cloning on every push/pop by allowing for mutable borrows:
    pub first_edge_codec: RefCell<FenwickTree<usize>>,
    pub node: usize,
    pub loops: bool,
    pub graph_len: usize,
    pub phantom: PhantomData<Ty>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Tailed<T: AsRef<[usize]>, C: Codec<Symbol=usize>> {
    main: Categorical<T>,
    tail: C,
}

impl<T: AsRef<[usize]>, C: Codec<Symbol=usize>> Codec for Tailed<T, C> {
    type Symbol = usize;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let num = self.num();
        if *x >= num {
            self.tail.push(m, &(x - num));
        }

        self.main.push(m, x.min(&num));
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let x = self.main.pop(m);
        if x == self.num() {
            x + self.tail.pop(m)
        } else {
            x
        }
    }

    fn bits(&self, _x: &Self::Symbol) -> Option<f64> { None }
}

impl<T: AsRef<[usize]>, C: Codec<Symbol=usize>> Tailed<T, C> {
    fn num(&self) -> usize {
        self.main.len() - 1
    }
}

const ZIPF: Categorical<[usize; 257]> = Categorical::new_const::<256>([
    39885, 9971, 4432, 2493, 1595, 1108, 814, 623, 492, 399, 330, 277, 236, 203, 177, 156, 138, 123,
    110, 100, 90, 82, 75, 69, 64, 59, 55, 51, 47, 44, 42, 39, 37, 35, 33, 31, 29, 28, 26, 25, 24,
    23, 22, 21, 20, 19, 18, 17, 17, 16, 15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 11, 10, 10, 10, 9,
    9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 74]);
const SHIFTED_ZIPF_SIZE: usize = ZIPF.cummasses.len() - 1;
const SHIFTED_ZIPF: Categorical<[usize; SHIFTED_ZIPF_SIZE]> = {
    let shift = ZIPF.cummasses[1];
    let mut cummasses = [0; SHIFTED_ZIPF_SIZE];
    let mut i = 0;
    while i < SHIFTED_ZIPF_SIZE {
        cummasses[i] = ZIPF.cummasses[i + 1] - shift;
        i += 1;
    }
    Categorical { cummasses }
};

impl<Ty: EdgeType> PolyaUrnSliceCodec<Ty> {
    const LEN_MAIN: Categorical<[usize; 4]> = Categorical::new_const([17, 17, 16]);
    const SMALL_GRAPH_LEN: usize = 100_000;

    /// Slice len approximation codec according to the Zipf distribution Q(k) ~ 1/(k+1)^2.
    /// The far tail is approximated by a uniform distribution.
    /// For smaller graphs we use Q(0)=Q(1)=34%, Q(k)1/k^2 for k>1 instead.
    fn len_codec(&self) -> impl Codec<Symbol=usize> {
        let num_lengths = 1 + self.loops as usize +
            (self.graph_len - self.node - 1) * if Ty::is_directed() { 2 } else { 1 };

        // #feature(adt_const_params) will allow deduplicating this code a lot:
        if self.graph_len > Self::SMALL_GRAPH_LEN {
            return EnumCodec::A(if num_lengths <= ZIPF.len() {
                EnumCodec::A(ZIPF.truncated(num_lengths))
            } else {
                EnumCodec::B(Tailed { main: ZIPF, tail: Uniform::new(num_lengths - (ZIPF.len() - 1)) })
            });
        }

        EnumCodec::B(
            if num_lengths <= Self::LEN_MAIN.len() {
                EnumCodec::A(Self::LEN_MAIN.truncated(num_lengths))
            } else {
                EnumCodec::B(Tailed {
                    main: Self::LEN_MAIN,
                    tail: {
                        let num_tail = num_lengths - (Self::LEN_MAIN.len() - 1);
                        if num_tail <= SHIFTED_ZIPF.len() {
                            EnumCodec::A(SHIFTED_ZIPF.truncated(num_tail))
                        } else {
                            EnumCodec::B(Tailed {
                                main: SHIFTED_ZIPF,
                                tail: Uniform::new(num_tail - (SHIFTED_ZIPF.len() - 1)),
                            })
                        }
                    },
                })
            })
    }

    fn from_graph<N: EqSymbol, E: EqSymbol>(prefix: &GraphPrefix<N, E, Ty>, loops: bool) -> Self {
        assert!(!Ty::is_directed(), "Autoregressive Polya model does not support directed graphs.");
        let node = prefix.len;
        let graph_len = prefix.graph.len();
        let min = node + !loops as usize;
        let masses = (0..graph_len).
            map(|j| if j < min { 0 } else { prefix.graph.nodes[j].1.len() + 1 });
        let first_edge_codec = RefCell::new(FenwickTree::from_iter(masses));
        Self { first_edge_codec, node, graph_len, loops, phantom: PhantomData }
    }

    fn codec(&self, len: usize) -> MultisetShuffleCodec<EdgeIndex, PolyaUrnSliceEdgeIndicesCodec<Ty>> {
        assert!(len > 0);
        joint_multiset_shuffle_codec(PolyaUrnSliceEdgeIndicesCodec {
            first_edge_codec: &self.first_edge_codec,
            len,
            node: self.node,
            phantom: self.phantom,
        })
    }
}

impl<Ty: EdgeType> Codec for PolyaUrnSliceCodec<Ty> {
    type Symbol = Multiset<EdgeIndex>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let len = x.len();
        if len > 0 {
            self.codec(len).push(m, x);
        }
        self.len_codec().push(m, &len);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let len = self.len_codec().pop(m);
        if len > 0 {
            self.codec(len).pop(m)
        } else {
            Unordered(vec![])
        }
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { None }
}

impl<Ty: EdgeType> EdgeIndicesCodec for PolyaUrnSliceCodec<Ty> {
    type Ty = Ty;

    fn loops(&self) -> bool { self.loops }
}

#[derive(Clone)]
pub struct PolyaUrnSliceEdgeIndicesCodec<'a, Ty: EdgeType = Directed> {
    /// RefCell to avoid cloning on every push/pop by allowing for mutable borrows:
    pub first_edge_codec: &'a RefCell<FenwickTree<usize>>,
    pub len: usize,
    pub node: usize,
    pub phantom: PhantomData<Ty>,
}

impl<'a, Ty: EdgeType> Codec for PolyaUrnSliceEdgeIndicesCodec<'a, Ty> {
    type Symbol = Vec<EdgeIndex>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let mut edge_codec = self.first_edge_codec.borrow_mut();
        assert_eq!(self.len, x.len());
        let masses = x.iter().map(|(node_, j)| {
            assert_eq!(self.node, *node_);
            edge_codec.remove_all(j)
        }).collect_vec();

        for ((_, j), mass) in x.iter().rev().zip_eq(masses.iter().rev()) {
            edge_codec.insert(*j, *mass);
            edge_codec.push(m, j);
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let mut edge_codec = self.first_edge_codec.borrow_mut();
        let mut edges = vec![];
        let mut masses = vec![];
        for _ in 0..self.len {
            let j = edge_codec.pop(m);
            masses.push(edge_codec.remove_all(&j));
            edges.push((self.node, j))
        }

        // Revert changes:
        for ((_, j), mass) in edges.iter().zip_eq(masses.into_iter()) {
            edge_codec.insert(*j, mass)
        }

        edges
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { None }
}

impl<'a, Ty: EdgeType> Len for PolyaUrnSliceEdgeIndicesCodec<'a, Ty> {
    fn len(&self) -> usize { self.len }
}

/// Autoregressive approximation of the Polya urn model for graphs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolyaUrnSliceCodecs<NodeC: Codec + Symbol, EdgeC: Codec + Symbol, Ty: EdgeType> {
    pub len: usize,
    pub node: NodeC,
    pub edge: EdgeC,
    pub loops: bool,
    pub phantom: PhantomData<Ty>,
}

impl<NodeC: Codec<Symbol: Eq + Default> + Symbol, EdgeC: Codec<Symbol: OrdSymbol> + Symbol, Ty: EdgeType> PrefixFn<GraphPrefixingChain<NodeC::Symbol, EdgeC::Symbol, Ty>> for PolyaUrnSliceCodecs<NodeC, EdgeC, Ty> {
    type Output = (NodeC, EdgesIID<EdgeC, PolyaUrnSliceCodec<Ty>>);

    fn apply(&self, x: &GraphPrefix<NodeC::Symbol, EdgeC::Symbol, Ty>) -> Self::Output {
        let indices = PolyaUrnSliceCodec::from_graph(x, self.loops);
        (self.node.clone(), EdgesIID::new(indices, self.edge.clone()))
    }

    fn update_after_pop_slice(&self, image: &mut Self::Output, x: &GraphPrefix<NodeC::Symbol, EdgeC::Symbol, Ty>, slice: &<GraphPrefixingChain<NodeC::Symbol, EdgeC::Symbol, Ty> as PrefixingChain>::Slice) {
        let indices = &mut image.1.indices;
        indices.node -= 1;

        let new_node = indices.node + !self.loops as usize;

        let edge_codec = indices.first_edge_codec.get_mut();
        if new_node < edge_codec.len() {
            edge_codec.insert(new_node, x.graph.nodes[new_node].1.len() + 1);
        }
        for ((node_, j), _) in slice.1.0.iter() {
            assert_eq!(indices.node, *node_);
            if *j != new_node {
                edge_codec.remove(j, 1);
            }
        }
    }

    fn update_after_push_slice(&self, image: &mut Self::Output, _: &GraphPrefix<NodeC::Symbol, EdgeC::Symbol, Ty>, slice: &<GraphPrefixingChain<NodeC::Symbol, EdgeC::Symbol, Ty> as PrefixingChain>::Slice) {
        let indices = &mut image.1.indices;
        let edge_codec = indices.first_edge_codec.get_mut();
        for ((node_, j), _) in slice.1.0.iter() {
            assert_eq!(indices.node, *node_);
            edge_codec.insert(*j, 1);
        }
        let i = indices.node + !self.loops as usize;
        if i < edge_codec.len() {
            edge_codec.remove_all(&i);
        }
        indices.node += 1;
    }

    fn swap(&self, _: &mut Self::Output, _: usize, _: usize) {} // Invariant to swap.
}

impl<NodeC: Codec + Symbol, EdgeC: Codec<Symbol: OrdSymbol> + Symbol, Ty: EdgeType> Len for PolyaUrnSliceCodecs<NodeC, EdgeC, Ty> {
    fn len(&self) -> usize {
        self.len
    }
}
