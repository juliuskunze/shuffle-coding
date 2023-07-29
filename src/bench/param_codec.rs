use crate::autoregressive::graph::{GraphPrefixingChain, PolyaUrnSliceCodecs};
use crate::autoregressive::Autoregressive;
use crate::codec::{Bernoulli, Categorical, Codec, ConstantCodec, Independent, LogUniform, Message, OrdSymbol, Symbol, Uniform, IID};
use crate::experimental::graph::{erdos_renyi_indices, num_all_edge_indices, polya_urn, AllEdgeIndices, EmptyCodec, ErdosRenyi, ErdosRenyiSliceCodecs, GraphIID, PolyaUrn};
use crate::graph::{Directed, EdgeType};
use crate::permutable::graph::EdgeIndicesCodec;
use crate::permutable::Len;
use itertools::Itertools;
use std::marker::PhantomData;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UniformParamCodec {
    pub size: Uniform,
}

impl Codec for UniformParamCodec {
    type Symbol = Uniform;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.size.push(m, &x.size);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let size = self.size.pop(m);
        Uniform::new(size)
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.size.bits(&x.size)
    }
}

impl UniformParamCodec {
    pub fn new(size: Uniform) -> Self { Self { size } }
}

impl Default for UniformParamCodec {
    fn default() -> Self { Self::new(Uniform::max()) }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SortedDiffRunLengthCodec;

impl Codec for SortedDiffRunLengthCodec {
    /// Array of ascendingly sorted integers.
    type Symbol = Vec<usize>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        debug_assert_eq!(&x.iter().sorted_unstable().cloned().collect_vec(), x);
        MaxBenfordIID.push(m, &Self::diffs_lens(x));
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let diffs_lens = MaxBenfordIID.pop(m);
        assert_eq!(diffs_lens.len() % 2, 0);
        let diffs = diffs_lens.iter().enumerate().filter(|&(i, _)| i % 2 == 0).map(|(_, &x)| x).collect_vec();
        let lens = diffs_lens.iter().enumerate().filter(|&(i, _)| i % 2 != 0).map(|(_, &x)| x).collect_vec();
        assert_eq!(diffs.len(), lens.len());
        let values = diffs.into_iter().scan(0, |state, x| {
            *state += x;
            Some(*state)
        }).collect_vec();
        assert_eq!(values.len(), lens.len());
        values.into_iter().zip(lens).flat_map(|(v, len)| vec![v; len]).collect()
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        MaxBenfordIID.bits(&Self::diffs_lens(x))
    }
}

impl SortedDiffRunLengthCodec {
    fn diffs_lens(x: &Vec<usize>) -> Vec<usize> {
        let run_lengths = x.iter().chunk_by(|&&x| x).
            into_iter().map(|(v, g)| (v, g.count())).collect_vec();
        [(0, 0)].iter().chain(run_lengths.iter()).collect_vec().windows(2).
            flat_map(|window| {
                if let [(prev, _), (next, len)] = window {
                    assert!(next > prev);
                    [next - prev, *len]
                } else { unreachable!() }
            }).collect_vec()
    }
}

impl Default for SortedDiffRunLengthCodec {
    fn default() -> Self { Self }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxBenfordIID;

impl Codec for MaxBenfordIID {
    type Symbol = Vec<usize>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let len = x.len();
        let bits = LogUniform::get_bits(x.iter().max().unwrap());
        self.iid(len, bits).push(m, &x);
        Self::bits_codec().push(m, &bits);
        Self::len_codec().push(m, &len);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        self.iid(Self::len_codec().pop(m), Self::bits_codec().pop(m)).pop(m)
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let len = x.len();
        let bits = LogUniform::get_bits(x.iter().max().unwrap());
        Some(self.iid(len, bits).bits(x)? + Self::len_codec().bits(&len)? + Self::bits_codec().bits(&bits)?)
    }
}

impl MaxBenfordIID {
    fn iid(&self, len: usize, bits: usize) -> IID<LogUniform> {
        IID::new(LogUniform::new(bits + 1), len)
    }

    fn bits_codec() -> &'static LogUniform {
        static C: &LogUniform = &LogUniform::new(6);
        C
    }

    fn len_codec() -> &'static LogUniform {
        LogUniform::max()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CategoricalParamCodec;

impl Codec for CategoricalParamCodec {
    type Symbol = Categorical;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        MaxBenfordIID.push(m, &x.masses().collect());
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let masses = MaxBenfordIID.pop(m);
        Categorical::from_iter(masses)
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        MaxBenfordIID.bits(&x.masses().collect())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BernoulliParamCodec;

impl Codec for BernoulliParamCodec {
    type Symbol = Bernoulli;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        CategoricalParamCodec.push(m, &x.categorical.clone())
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        Bernoulli { categorical: CategoricalParamCodec.pop(m) }
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        CategoricalParamCodec.bits(&x.categorical.clone())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ErdosRenyiParamCodec<Ty: EdgeType> {
    pub nums_nodes: Vec<usize>,
    pub loops: bool,
    pub edge_codec: BernoulliParamCodec,
    pub phantom: PhantomData<Ty>,
}

impl<Ty: EdgeType> Codec for ErdosRenyiParamCodec<Ty> {
    type Symbol = Vec<ErdosRenyi<Ty>>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        assert!(!x.is_empty());
        self.edge_codec.push(m, &x.first().unwrap().dense.contains.item)
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let edge_index = self.edge_codec.pop(m);
        self.nums_nodes.iter().map(|&num_nodes| erdos_renyi_indices::<Ty>(num_nodes, edge_index.clone(), self.loops)).collect()
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.edge_codec.bits(&x.first().unwrap().dense.contains.item)
    }
}

impl<Ty: EdgeType> ErdosRenyiParamCodec<Ty> {
    pub fn new(nums_nodes: Vec<usize>, loops: bool) -> Self {
        Self { nums_nodes, loops, edge_codec: BernoulliParamCodec, phantom: PhantomData }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolyaUrnParamCodec<Ty: EdgeType> {
    pub nums_nodes: Vec<usize>,
    pub nums_edges: Independent<Uniform>,
    pub loops: bool,
    pub redraws: bool,
    pub phantom: PhantomData<Ty>,
}

impl<Ty: EdgeType> Codec for PolyaUrnParamCodec<Ty> {
    type Symbol = Vec<PolyaUrn<Ty>>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        assert!(x.iter().all(|x| x.loops() == self.loops));
        assert!(x.iter().all(|x| x.ordered().redraws == self.redraws));
        assert_eq!(x.iter().map(|x| x.ordered().num_nodes).collect_vec(), self.nums_nodes);
        self.nums_edges.push(m, &x.iter().map(|x| x.ordered().num_edges).collect())
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let nums_edges = self.nums_edges.pop(m);

        self.nums_nodes.iter().zip_eq(nums_edges.into_iter()).map(|(num_nodes, nums_edges)|
            polya_urn(*num_nodes, nums_edges, self.loops, self.redraws)).collect()
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.nums_edges.bits(&x.iter().map(|x| x.ordered().num_edges).collect())
    }
}

impl<Ty: EdgeType> PolyaUrnParamCodec<Ty> {
    pub fn new(nums_nodes: Vec<usize>, loops: bool, redraws: bool) -> Self {
        let nums_edges = Independent::new(nums_nodes.iter().map(
            |n| { Uniform::new(num_all_edge_indices::<Ty>(*n, loops) + 1) }));
        Self { nums_nodes, nums_edges, loops, redraws, phantom: PhantomData }
    }
}

pub type EmptyParamCodec = ConstantCodec<EmptyCodec>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphDatasetParamCodec<
    IndicesParamCFromNumsNodesAndLoops: Fn(Vec<usize>, bool) -> IndicesParamC + Clone,
    NodeParamC: Codec<Symbol: Codec> = EmptyParamCodec,
    EdgeParamC: Codec<Symbol: Codec<Symbol: OrdSymbol>> = EmptyParamCodec,
    IndicesC: Symbol + EdgeIndicesCodec = ErdosRenyi<Directed>,
    IndicesParamC: Codec<Symbol=Vec<IndicesC>> = ErdosRenyiParamCodec<<IndicesC as EdgeIndicesCodec>::Ty>> {
    /// Code the sequence of numbers of nodes for each graph as a categorical distribution:
    pub node: NodeParamC,
    pub edge: EdgeParamC,
    pub indices: IndicesParamCFromNumsNodesAndLoops,
}

impl<
    NodeParamC: Codec<Symbol: Codec<Symbol: Eq>>,
    EdgeParamC: Codec<Symbol: Codec<Symbol: OrdSymbol>>,
    IndicesC: EdgeIndicesCodec + Symbol,
    IndicesParamC: Codec<Symbol=Vec<IndicesC>>,
    IndicesParamCFromNumsNodesAndLoops: Fn(Vec<usize>, bool) -> IndicesParamC + Clone>
Codec for GraphDatasetParamCodec<IndicesParamCFromNumsNodesAndLoops, NodeParamC, EdgeParamC, IndicesC, IndicesParamC> {
    type Symbol = Independent<GraphIID<NodeParamC::Symbol, EdgeParamC::Symbol, IndicesC>>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let first = x.codecs.first().clone().unwrap();
        self.edge.push(m, &first.edges.label);
        self.node.push(m, &first.nodes.item);
        let mut nums_nodes = x.codecs.iter().map(|x| x.nodes.len).collect_vec();
        let indices = x.codecs.iter().map(|x| x.edges.indices.clone()).collect();
        let loops = first.edges.indices.loops();
        (self.indices)(nums_nodes.clone(), loops).push(m, &indices);
        Bernoulli::new(1, 2).push(m, &loops);
        nums_nodes.reverse();
        SortedDiffRunLengthCodec::default().push(m, &nums_nodes);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let mut nums_nodes = SortedDiffRunLengthCodec::default().pop(m);
        nums_nodes.reverse();
        let loops = Bernoulli::new(1, 2).pop(m);
        let indices = (self.indices)(nums_nodes.clone(), loops).pop(m);
        let node = self.node.pop(m);
        let edge = self.edge.pop(m);
        Independent::new(nums_nodes.into_iter().zip_eq(indices.into_iter()).map(|(n, indices)|
            GraphIID::new(n, indices, node.clone(), edge.clone())))
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let first = x.codecs.first().clone().unwrap();
        let mut nums_nodes = x.codecs.iter().map(|x| x.nodes.len).collect_vec();
        nums_nodes.reverse();
        let indices = x.codecs.iter().map(|x| x.edges.indices.clone()).collect();
        let loops_bit = 1.;
        Some(SortedDiffRunLengthCodec::default().bits(&nums_nodes)? + (self.indices)(nums_nodes, first.edges.indices.loops()).bits(&indices)? +
            self.node.bits(&first.nodes.item)? + self.edge.bits(&first.edges.label)? + loops_bit)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AutoregressiveErdosReyniGraphDatasetParamCodec<
    IndicesParamCFromNumsNodesAndLoops: Fn(Vec<usize>, bool) -> IndicesParamC + Clone,
    NodeParamC: Codec<Symbol: Codec> = EmptyParamCodec,
    EdgeParamC: Codec<Symbol: Codec<Symbol: OrdSymbol>> = EmptyParamCodec,
    IndicesC: EdgeIndicesCodec + Symbol = ErdosRenyi<Directed>,
    IndicesParamC: Codec<Symbol=Vec<IndicesC>> = ErdosRenyiParamCodec<<IndicesC as EdgeIndicesCodec>::Ty>> {
    pub joint: GraphDatasetParamCodec<IndicesParamCFromNumsNodesAndLoops, NodeParamC, EdgeParamC, IndicesC, IndicesParamC>,
}

impl<
    Ty: EdgeType,
    NodeParamC: Codec<Symbol: Codec<Symbol: Eq + Default>>,
    EdgeParamC: Codec<Symbol: Codec<Symbol: OrdSymbol>>,
    IndicesParamC: Codec<Symbol=Vec<ErdosRenyi<Ty>>>,
    IndicesParamCFromNumsNodesAndLoops: Fn(Vec<usize>, bool) -> IndicesParamC + Clone>
Codec for AutoregressiveErdosReyniGraphDatasetParamCodec<IndicesParamCFromNumsNodesAndLoops, NodeParamC, EdgeParamC, ErdosRenyi<Ty>, IndicesParamC> {
    type Symbol = Independent<Autoregressive<GraphPrefixingChain<<NodeParamC::Symbol as Codec>::Symbol, <EdgeParamC::Symbol as Codec>::Symbol, Ty>, ErdosRenyiSliceCodecs<NodeParamC::Symbol, EdgeParamC::Symbol, Ty>>>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let er = Independent::new(x.codecs.iter().map(|c| &c.0.slices).map(
            |c| GraphIID::new(c.len, ErdosRenyi::new(c.has_edge.clone(),
                                                     AllEdgeIndices { num_nodes: c.len, loops: c.loops, phantom: PhantomData },
                                                     c.loops), c.node.clone(), c.edge.clone())
        ));

        self.joint.push(m, &er)
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        Independent::new(self.joint.pop(m).codecs.into_iter().map(
            |c| Autoregressive::new(ErdosRenyiSliceCodecs {
                len: c.nodes.len,
                has_edge: c.edges.indices.dense.contains.item.clone(),
                node: c.nodes.item.clone(),
                edge: c.edges.label.clone(),
                loops: c.edges.indices.loops(),
                phantom: PhantomData,
            })))
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { None }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AutoregressivePolyaUrnGraphDatasetParamCodec<
    NodeParamC: Codec<Symbol: Codec>,
    EdgeParamC: Codec<Symbol: Codec<Symbol: OrdSymbol>>,
    Ty: EdgeType> {
    pub node: NodeParamC,
    pub edge: EdgeParamC,
    pub phantom: PhantomData<Ty>,
}

impl<
    NodeParamC: Codec<Symbol: Codec<Symbol: Eq + Default>>,
    EdgeParamC: Codec<Symbol: Codec<Symbol: OrdSymbol>>,
    Ty: EdgeType>
Codec for AutoregressivePolyaUrnGraphDatasetParamCodec<NodeParamC, EdgeParamC, Ty> {
    type Symbol = Independent<Autoregressive<GraphPrefixingChain<<NodeParamC::Symbol as Codec>::Symbol, <EdgeParamC::Symbol as Codec>::Symbol, Ty>, PolyaUrnSliceCodecs<NodeParamC::Symbol, EdgeParamC::Symbol, Ty>>>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let first = &x.codecs.first().unwrap().0.slices;
        self.edge.push(m, &first.edge);
        self.node.push(m, &first.node);
        Bernoulli::new(1, 2).push(m, &first.loops);
        let nums_nodes = x.codecs.iter().map(|x| x.len()).rev().collect_vec();
        SortedDiffRunLengthCodec::default().push(m, &nums_nodes);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let nums_nodes = SortedDiffRunLengthCodec::default().pop(m);
        let loops = Bernoulli::new(1, 2).pop(m);
        let node = self.node.pop(m);
        let edge = self.edge.pop(m);
        Independent::new(nums_nodes.into_iter().rev().map(|n|
            Autoregressive::new(PolyaUrnSliceCodecs { len: n, node: node.clone(), edge: edge.clone(), loops, phantom: PhantomData })))
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { None }
}


#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParametrizedIndependent<
    ParamC: Codec<Symbol: Codec>,
    Infer: Fn(&<ParamC::Symbol as Codec>::Symbol) -> ParamC::Symbol + Clone> {
    pub param_codec: ParamC,
    pub infer: Infer,
}

impl<
    ParamC: Codec<Symbol: Codec>,
    Infer: Fn(&<ParamC::Symbol as Codec>::Symbol) -> ParamC::Symbol + Clone>
Codec for ParametrizedIndependent<ParamC, Infer> {
    type Symbol = <ParamC::Symbol as Codec>::Symbol;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let data_codec = (self.infer)(&x);
        data_codec.push(m, &x);
        self.param_codec.push(m, &data_codec);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        self.param_codec.pop(m).pop(m)
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let data_codec = (self.infer)(&x);
        Some(data_codec.bits(x)? + self.param_codec.bits(&data_codec)?)
    }
}

pub struct WrappedParametrizedIndependent<
    ParamC: Codec<Symbol: Codec>,
    Infer: Fn(&<ParamC::Symbol as Codec>::Symbol) -> ParamC::Symbol + Clone,
    C: Symbol + Codec> {
    pub parametrized_codec: ParametrizedIndependent<ParamC, Infer>,
    pub codec_from_inner: Box<dyn Fn(ParamC::Symbol) -> C>,
    pub data_to_inner: Box<dyn Fn(C::Symbol) -> <ParamC::Symbol as Codec>::Symbol>,
}

impl<ParamC: Codec<Symbol: Codec>,
    Infer: Fn(&<ParamC::Symbol as Codec>::Symbol) -> ParamC::Symbol + Clone, C: Symbol + Codec>
Codec for WrappedParametrizedIndependent<ParamC, Infer, C> {
    type Symbol = C::Symbol;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let param = self.infer(&x);
        (self.codec_from_inner)(param.clone()).push(m, &x);
        self.parametrized_codec.param_codec.push(m, &param);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let param = self.parametrized_codec.param_codec.pop(m);
        (self.codec_from_inner)(param).pop(m)
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let param = self.infer(x);
        Some((self.codec_from_inner)(param.clone()).bits(x)? + self.parametrized_codec.param_codec.bits(&param)?)
    }
}

impl<
    ParamC: Codec<Symbol: Codec>,
    Infer: Fn(&<ParamC::Symbol as Codec>::Symbol) -> ParamC::Symbol + Clone,
    C: Symbol + Codec> WrappedParametrizedIndependent<ParamC, Infer, C> {
    pub fn infer(&self, x: &<C as Codec>::Symbol) -> ParamC::Symbol {
        (self.parametrized_codec.infer)(&(self.data_to_inner)(x.clone()))
    }
}

#[cfg(test)]
pub mod tests {
    use crate::codec::{Bernoulli, Categorical, Codec, ConstantCodec, Independent, Uniform};
    use crate::experimental::graph::{erdos_renyi_indices, EmptyCodec, GraphIID};
    use crate::graph::Undirected;

    use super::*;

    #[test]
    fn sorted_diff_run_length() {
        let v = vec![1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 26];
        SortedDiffRunLengthCodec::default().test(&v, 0);
    }

    #[test]
    pub fn param_codecs() {
        UniformParamCodec::new(Uniform::new(10)).test(&Uniform::new(9), 0);

        let masses = vec!(1, 2, 3, 0, 0, 27, 54);
        let label = Categorical::from_iter(masses);
        CategoricalParamCodec.test(&label, 0);
        let edge_index = Bernoulli::new(20, 50);
        BernoulliParamCodec.test(&edge_index, 0);
        let indices = erdos_renyi_indices::<Undirected>(10, edge_index.clone(), false);
        let indices_param = |nums_nodes, loops|
            ErdosRenyiParamCodec { nums_nodes, loops, edge_codec: BernoulliParamCodec, phantom: PhantomData };
        indices_param(vec![10], false).test(&vec![indices.clone()], 0);
        let graphs = Independent::new(vec![GraphIID::new(10, indices, EmptyCodec::default(), label.clone())]);
        GraphDatasetParamCodec {
            node: ConstantCodec(EmptyCodec::default()),
            edge: CategoricalParamCodec,
            indices: indices_param,
        }.test(&graphs, 0);
    }
}
