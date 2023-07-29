//! "Joint autoregressive" shuffle coding, a variant of autoregressive shuffle coding that behaves
//! like joint shuffle coding: It allows using a joint model, and has the same initial bit overhead.
//! For graphs, it can have slightly better rate than joint shuffle coding,
//! since it runs color refinement on every prefix, which also makes it slow.
//! In fact, it is slower than actual joint shuffle coding for both graphs and multisets,
//! so mostly here for reference.

use crate::autoregressive::graph::{GraphPrefix, GraphPrefixingChain};
use crate::autoregressive::multiset::DynPrefixOrbitCodec;
use crate::autoregressive::prefix_orbit::FixPrefixOrbitCodec;
use crate::autoregressive::{AutoregressiveShuffleCodec, InnerSliceCodecs, PrefixFn, PrefixingChain,
                            UncachedPrefixFn, UnfusedAutoregressiveShuffleCodec};
use crate::codec::{Codec, ConstantCodec, EqSymbol, Message, OrdSymbol, Symbol};
use crate::experimental::complete_joint::aut::{AutCanonizable, Automorphisms};
use crate::graph::{ColorRefinement, EdgeType, Graph};
use crate::permutable::{Len, Permutable, PermutableCodec};
use std::marker::PhantomData;

/// Prefix type forming the basis for joint shuffle coding, where the empty (length 0) prefix
/// contains the full permutable object, and all slices are empty.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointPrefix<P: Permutable> {
    pub full: P,
    pub len: usize,
}

impl<P: Permutable> Permutable for JointPrefix<P> {
    fn len(&self) -> usize {
        self.len
    }

    fn swap(&mut self, i: usize, j: usize) {
        assert!(i < self.len());
        assert!(j < self.len());
        self.full.swap(i, j)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointPrefixingChain<P: Permutable>(PhantomData<P>);
impl<P: Permutable> JointPrefixingChain<P> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<P: Permutable> PrefixingChain for JointPrefixingChain<P> {
    type Prefix = JointPrefix<P>;
    type Full = P;
    type Slice = ();

    fn pop_slice(&self, prefix: &mut Self::Prefix) {
        prefix.len -= 1;
    }

    fn push_slice(&self, prefix: &mut Self::Prefix, _: &()) {
        prefix.len += 1;
    }

    fn prefix(&self, full: Self::Full) -> Self::Prefix {
        Self::Prefix { len: full.len(), full }
    }

    fn full(&self, prefix: Self::Prefix) -> Self::Full {
        prefix.full
    }
}

impl<C: PermutableCodec + Clone> UncachedPrefixFn<JointPrefixingChain<C::Symbol>> for JointSliceCodecs<C> {
    type Output = ConstantCodec<()>;

    fn apply(&self, _: &JointPrefix<C::Symbol>) -> Self::Output {
        ConstantCodec(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointSliceCodecs<C: PermutableCodec> {
    pub empty: EmptyJointPrefixCodec<C>,
}

impl<C: PermutableCodec> Len for JointSliceCodecs<C> {
    fn len(&self) -> usize {
        self.empty.full.len()
    }
}

impl<C: PermutableCodec + Clone> InnerSliceCodecs<JointPrefixingChain<C::Symbol>> for JointSliceCodecs<C> {
    fn prefixing_chain(&self) -> JointPrefixingChain<C::Symbol> {
        JointPrefixingChain::new()
    }

    fn empty_prefix(&self) -> impl Codec<Symbol=JointPrefix<C::Symbol>> {
        self.empty.clone()
    }
}

impl<C: PermutableCodec> JointSliceCodecs<C> {
    pub fn new(full: C) -> Self {
        Self { empty: EmptyJointPrefixCodec { full } }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EmptyJointPrefixCodec<C: PermutableCodec> {
    pub full: C,
}

impl<C: PermutableCodec> Codec for EmptyJointPrefixCodec<C> {
    type Symbol = JointPrefix<C::Symbol>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.full.push(m, &x.full)
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        JointPrefix { len: 0, full: self.full.pop(m) }
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.full.bits(&x.full)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointVecOrbitCodecs {
    len: usize,
}

impl JointVecOrbitCodecs {
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl<S: OrdSymbol + Default> PrefixFn<JointPrefixingChain<Vec<S>>> for JointVecOrbitCodecs {
    type Output = FixPrefixOrbitCodec;

    fn apply(&self, x: &JointPrefix<Vec<S>>) -> Self::Output {
        FixPrefixOrbitCodec::new(x.full.clone(), x.len)
    }

    fn update_after_pop_slice(&self, orbits: &mut Self::Output, _x: &JointPrefix<Vec<S>>, _slice: &()) {
        orbits.pop_id();
    }

    fn update_after_push_slice(&self, orbits: &mut Self::Output, _x: &JointPrefix<Vec<S>>, _slice: &()) {
        orbits.push_id()
    }

    fn swap(&self, orbits: &mut Self::Output, i: usize, j: usize) {
        orbits.swap(i, j)
    }
}

#[allow(unused)]
pub fn autoregressive_joint_multiset_shuffle_codec<S: OrdSymbol + Default, C: PermutableCodec<Symbol=Vec<S>> + Symbol>(ordered: C) -> UnfusedAutoregressiveShuffleCodec<JointPrefixingChain<Vec<S>>, JointSliceCodecs<C>, JointVecOrbitCodecs> {
    let len = ordered.len();
    AutoregressiveShuffleCodec::new(JointSliceCodecs::new(ordered), JointVecOrbitCodecs::new(len))
}

impl<N: EqSymbol + Default, E: EqSymbol, Ty: EdgeType> AutCanonizable for JointPrefix<Graph<N, E, Ty>>
where
    GraphPrefix<N, E, Ty>: AutCanonizable,
{
    fn automorphisms(&self) -> Automorphisms {
        let mut prefix = GraphPrefixingChain::new().prefix(self.full.clone());
        prefix.len = self.len;
        prefix.automorphisms()
    }
}

impl<N: OrdSymbol + Default, E: OrdSymbol, Ty: EdgeType> PrefixFn<JointPrefixingChain<Graph<N, E, Ty>>> for ColorRefinement {
    type Output = DynPrefixOrbitCodec<Vec<usize>>;

    fn apply(&self, x: &JointPrefix<Graph<N, E, Ty>>) -> Self::Output {
        let chain = GraphPrefixingChain::new();
        let mut prefix = chain.prefix(x.full.clone());
        prefix.len = x.len;
        PrefixFn::<GraphPrefixingChain<N, E, Ty>>::apply(self, &prefix)
    }

    fn update_after_pop_slice(&self, orbits: &mut Self::Output, x: &JointPrefix<Graph<N, E, Ty>>, _slice: &()) {
        if Ty::is_directed() {
            *orbits = PrefixFn::<JointPrefixingChain<_>>::apply(self, x);
            return;
        }

        orbits.pop_id();
        self.update_around::<N, E, Ty, _>(&x.full, orbits, vec![x.len()]);
    }

    fn update_after_push_slice(&self, orbits: &mut Self::Output, x: &JointPrefix<Graph<N, E, Ty>>, _slice: &()) {
        if Ty::is_directed() {
            *orbits = PrefixFn::<JointPrefixingChain<_>>::apply(self, x);
            return;
        }

        orbits.push_id();
        self.update_around::<N, E, Ty, _>(&x.full, orbits, vec![x.len() - 1]);
    }

    fn swap(&self, orbits: &mut Self::Output, i: usize, j: usize) {
        orbits.swap(i, j)
    }
}
