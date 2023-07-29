use std::marker::PhantomData;

use itertools::Itertools;
use lazy_static::lazy_static;
use petgraph::Directed;

use crate::ans::{Benford, Bernoulli, Categorical, Codec, ConstantCodec, IID, Message, Symbol, Uniform, VecCodec};
use crate::graph::{AsEdges, EdgeType, Graph, PlainGraph};
use crate::graph_ans;
use crate::graph_ans::{EdgeIndicesCodec, EmptyCodec, ErdosRenyi, GraphIID, PolyaUrn};
use crate::multiset::OrdSymbol;

#[derive(Clone, Debug)]
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
    fn default() -> Self { Self::new(Uniform::max().clone()) }
}

#[derive(Clone, Debug)]
pub struct SortedDiffRunLengthCodec;

impl Codec for SortedDiffRunLengthCodec {
    /// Array of ascendingly sorted integers.
    type Symbol = Vec<usize>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        assert_eq!(&x.iter().sorted().cloned().collect_vec(), x);
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
        let run_lengths = x.iter().group_by(|&&x| x).
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

#[derive(Clone, Debug)]
pub struct MaxBenfordIID;

impl Codec for MaxBenfordIID {
    type Symbol = Vec<usize>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let len = x.len();
        let bits = Benford::get_bits(x.iter().max().unwrap());
        self.iid(len, bits).push(m, &x);
        Self::bits_codec().push(m, &bits);
        Self::len_codec().push(m, &len);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        self.iid(Self::len_codec().pop(m), Self::bits_codec().pop(m)).pop(m)
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let len = x.len();
        let bits = Benford::get_bits(x.iter().max().unwrap());
        Some(self.iid(len, bits).bits(x)? + Self::len_codec().bits(&len)? + Self::bits_codec().bits(&bits)?)
    }
}

impl MaxBenfordIID {
    fn iid(&self, len: usize, bits: usize) -> IID<Benford> {
        IID::new(Benford::new(bits + 1), len)
    }

    fn bits_codec() -> &'static Benford {
        lazy_static! {static ref C: Benford = Benford::new(6);}
        &C
    }

    fn len_codec() -> &'static Benford {
        Benford::max()
    }
}

#[derive(Clone, Debug)]
pub struct CategoricalParamCodec;

impl Codec for CategoricalParamCodec {
    type Symbol = Categorical;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        MaxBenfordIID.push(m, &x.masses);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let masses = MaxBenfordIID.pop(m);
        Categorical::new(masses)
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        MaxBenfordIID.bits(&x.masses)
    }
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
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
        self.nums_nodes.iter().map(|&num_nodes| graph_ans::erdos_renyi_indices::<Ty>(num_nodes, edge_index.clone(), self.loops)).collect()
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.edge_codec.bits(&x.first().unwrap().dense.contains.item)
    }
}

impl<Ty: EdgeType> ErdosRenyiParamCodec<Ty> {
    pub fn new(nums_nodes: Vec<usize>, loops: bool) -> Self {
        Self { nums_nodes, loops, edge_codec: BernoulliParamCodec, phantom: Default::default() }
    }
}

#[derive(Clone, Debug)]
pub struct PolyaUrnParamCodec<Ty: EdgeType> where PlainGraph<Ty>: AsEdges {
    pub nums_nodes: Vec<usize>,
    pub nums_edges: VecCodec<Uniform>,
    pub loops: bool,
    pub redraws: bool,
    pub phantom: PhantomData<Ty>,
}

impl<Ty: EdgeType> Codec for PolyaUrnParamCodec<Ty> where PlainGraph<Ty>: AsEdges {
    type Symbol = Vec<PolyaUrn<Ty>>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        assert!(x.iter().all(|x| x.ordered.loops == self.loops));
        assert!(x.iter().all(|x| x.ordered.redraws == self.redraws));
        assert_eq!(x.iter().map(|x| x.ordered.num_nodes).collect_vec(), self.nums_nodes);
        self.nums_edges.push(m, &x.iter().map(|x| x.ordered.num_edges).collect())
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let nums_edges = self.nums_edges.pop(m);

        self.nums_nodes.iter().zip_eq(nums_edges.into_iter()).map(|(num_nodes, nums_edges)|
            graph_ans::polya_urn(*num_nodes, nums_edges, self.loops, self.redraws)).collect()
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.nums_edges.bits(&x.iter().map(|x| x.ordered.num_edges).collect())
    }
}

impl<Ty: EdgeType> PolyaUrnParamCodec<Ty> where PlainGraph<Ty>: AsEdges {
    pub fn new(nums_nodes: Vec<usize>, loops: bool, redraws: bool) -> Self {
        let nums_edges = VecCodec::new(nums_nodes.iter().map(
            |n| {
                let size = graph_ans::num_all_edge_indices::<Ty>(*n, loops) + 1;
                Uniform::new(size)
            }));
        Self { nums_nodes, nums_edges, loops, redraws, phantom: Default::default() }
    }
}

pub type EmptyParamCodec = ConstantCodec<EmptyCodec>;

#[derive(Clone, Debug)]
pub struct GraphDatasetParamCodec<
    IndicesParamCFromNumsNodesAndLoops: Fn(Vec<usize>, bool) -> IndicesParamC + Clone,
    NodeC: Symbol + Codec = EmptyCodec,
    NodeParamC: Codec<Symbol=NodeC> = EmptyParamCodec,
    EdgeC: Symbol + Codec = EmptyCodec,
    EdgeParamC: Codec<Symbol=EdgeC> = EmptyParamCodec,
    IndicesC: Symbol + EdgeIndicesCodec = ErdosRenyi<Directed>,
    IndicesParamC: Codec<Symbol=Vec<IndicesC>> = ErdosRenyiParamCodec<<IndicesC as EdgeIndicesCodec>::Ty>>
    where EdgeC::Symbol: OrdSymbol {
    /// Code the sequence of numbers of nodes for each graph as a categorical distribution:
    pub node: NodeParamC,
    pub edge: EdgeParamC,
    pub indices: IndicesParamCFromNumsNodesAndLoops,
}

impl<
    NodeC: Symbol + Codec,
    NodeParamC: Codec<Symbol=NodeC>,
    EdgeC: Symbol + Codec,
    EdgeParamC: Codec<Symbol=EdgeC>,
    IndicesC: Symbol + EdgeIndicesCodec,
    IndicesParamC: Codec<Symbol=Vec<IndicesC>>,
    IndicesParamCFromNumsNodesAndLoops: Fn(Vec<usize>, bool) -> IndicesParamC + Clone>
Codec for GraphDatasetParamCodec<IndicesParamCFromNumsNodesAndLoops, NodeC, NodeParamC, EdgeC, EdgeParamC, IndicesC, IndicesParamC> where EdgeC::Symbol: OrdSymbol, Graph<NodeC::Symbol, EdgeC::Symbol, IndicesC::Ty>: AsEdges<NodeC::Symbol, EdgeC::Symbol> {
    type Symbol = VecCodec<GraphIID<NodeC, EdgeC, IndicesC>>;

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
        VecCodec::new(nums_nodes.into_iter().zip_eq(indices.into_iter()).map(|(n, indices)|
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
pub struct ParametrizedIndependent<
    DataC: Symbol + Codec,
    ParamC: Codec<Symbol=DataC>,
    Infer: Fn(&DataC::Symbol) -> DataC + Clone>
{
    pub param_codec: ParamC,
    pub infer: Infer,
}

impl<
    DataC: Symbol + Codec,
    ParamC: Codec<Symbol=DataC>,
    Infer: Fn(&DataC::Symbol) -> DataC + Clone>
Codec for ParametrizedIndependent<DataC, ParamC, Infer> {
    type Symbol = DataC::Symbol;

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
    InnerC: Symbol + Codec,
    ParamC: Codec<Symbol=InnerC>,
    Infer: Fn(&InnerC::Symbol) -> InnerC + Clone,
    C: Symbol + Codec>
{
    pub parametrized_codec: ParametrizedIndependent<InnerC, ParamC, Infer>,
    pub from_inner_codec: Box<dyn Fn(InnerC) -> C>,
    pub data_to_inner: Box<dyn Fn(C::Symbol) -> InnerC::Symbol>,
    pub data_from_inner: Box<dyn Fn(InnerC::Symbol) -> C::Symbol>,
}

impl<InnerC: Symbol + Codec, ParamC: Codec<Symbol=InnerC>,
    Infer: Fn(&InnerC::Symbol) -> InnerC + Clone,
    C: Symbol + Codec> Clone for WrappedParametrizedIndependent<InnerC, ParamC, Infer, C> {
    fn clone(&self) -> Self {
        unimplemented!()
    }
}

impl<InnerC: Symbol + Codec, ParamC: Codec<Symbol=InnerC>,
    Infer: Fn(&InnerC::Symbol) -> InnerC + Clone, C: Symbol + Codec>
Codec for WrappedParametrizedIndependent<InnerC, ParamC, Infer, C> {
    type Symbol = C::Symbol;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let param = self.infer(&x);
        (self.from_inner_codec)(param.clone()).push(m, &x);
        self.parametrized_codec.param_codec.push(m, &param);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let param = self.parametrized_codec.param_codec.pop(m);
        (self.from_inner_codec)(param).pop(m)
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let param = self.infer(x);
        Some((self.from_inner_codec)(param.clone()).bits(x)? + self.parametrized_codec.param_codec.bits(&param)?)
    }
}

impl<
    InnerC: Symbol + Codec,
    ParamC: Codec<Symbol=InnerC>,
    Infer: Fn(&InnerC::Symbol) -> InnerC + Clone,
    C: Symbol + Codec> WrappedParametrizedIndependent<InnerC, ParamC, Infer, C> {
    pub fn infer(&self, x: &<C as Codec>::Symbol) -> InnerC {
        (self.parametrized_codec.infer)(&(self.data_to_inner)(x.clone()))
    }
}

#[cfg(test)]
pub mod tests {
    use petgraph::Undirected;

    use crate::ans::{Bernoulli, Categorical, Codec, ConstantCodec, Message, Uniform, VecCodec};
    use crate::graph_ans::{EmptyCodec, erdos_renyi_indices, GraphIID};
    use crate::param_ans::{BernoulliParamCodec, CategoricalParamCodec, ErdosRenyiParamCodec, GraphDatasetParamCodec, SortedDiffRunLengthCodec, UniformParamCodec};

    #[test]
    fn sorted_diff_run_length() {
        let v = vec![1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 26];
        SortedDiffRunLengthCodec::default().test(&v, &Message::random(0));
    }

    #[test]
    pub fn param_codecs() {
        UniformParamCodec::new(Uniform::new(10)).test(&Uniform::new(9), &Message::random(0));

        let masses = vec!(1, 2, 3, 0, 0, 27, 54);
        let label = Categorical::new(masses);
        CategoricalParamCodec.test(&label, &Message::random(0));
        let edge_index = Bernoulli::new(20, 50);
        BernoulliParamCodec.test(&edge_index, &Message::random(0));
        let indices = erdos_renyi_indices::<Undirected>(10, edge_index.clone(), false);
        let indices_param = |nums_nodes, loops|
            ErdosRenyiParamCodec { nums_nodes, loops, edge_codec: BernoulliParamCodec, phantom: Default::default() };
        indices_param(vec![10], false).test(&vec![indices.clone()], &Message::random(0));
        let graphs = VecCodec::new(vec![GraphIID::new(10, indices, EmptyCodec::default(), label.clone())]);
        GraphDatasetParamCodec {
            node: ConstantCodec(EmptyCodec::default()),
            edge: CategoricalParamCodec,
            indices: indices_param,
        }.test(&graphs, &Message::random(0));
    }
}
