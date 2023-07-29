//! Autoregressive shuffle coding on the Cartesian product of two permutable classes.
//! Mostly here as a reference, more specific impls can typically be more efficient.
use crate::autoregressive::{OrbitCodec, PrefixFn, PrefixingChain, UncachedPrefixFn};
use crate::codec::avl::AvlCategorical;
use crate::codec::{Codec, Message, OrdSymbol};
use crate::permutable::FHashMap;
use crate::permutable::Len;

impl<P: PrefixingChain, Q: PrefixingChain> PrefixingChain for (P, Q) {
    type Prefix = (P::Prefix, Q::Prefix);
    type Full = (P::Full, Q::Full);
    type Slice = (P::Slice, Q::Slice);

    fn pop_slice(&self, prefix: &mut Self::Prefix) -> Self::Slice {
        (self.0.pop_slice(&mut prefix.0), self.1.pop_slice(&mut prefix.1))
    }

    fn push_slice(&self, prefix: &mut Self::Prefix, slice: &Self::Slice) {
        self.0.push_slice(&mut prefix.0, &slice.0);
        self.1.push_slice(&mut prefix.1, &slice.1);
    }

    fn prefix(&self, full: Self::Full) -> Self::Prefix {
        (self.0.prefix(full.0), self.1.prefix(full.1))
    }

    fn full(&self, prefix: Self::Prefix) -> Self::Full {
        (self.0.full(prefix.0), self.1.full(prefix.1))
    }
}

pub trait OrdOrbitCodec: OrbitCodec<Symbol: OrdSymbol + Default> + Len {}
impl<O: OrbitCodec<Symbol: OrdSymbol + Default> + Len> OrdOrbitCodec for O {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductOrbitCodec<A: OrdOrbitCodec, B: OrdOrbitCodec> {
    pub a: A,
    pub b: B,
    pub codec: AvlCategorical<(A::Symbol, B::Symbol)>,
}

impl<A: OrdOrbitCodec, B: OrdOrbitCodec> Codec for ProductOrbitCodec<A, B> {
    type Symbol = (A::Symbol, B::Symbol);
    fn push(&self, m: &mut Message, x: &Self::Symbol) { self.codec.push(m, x) }
    fn pop(&self, m: &mut Message) -> Self::Symbol { self.codec.pop(m) }
    fn bits(&self, x: &Self::Symbol) -> Option<f64> { self.codec.bits(x) }
}

impl<A: OrdOrbitCodec, B: OrdOrbitCodec> OrbitCodec for ProductOrbitCodec<A, B> {
    fn id(&self, index: usize) -> Self::Symbol {
        (self.a.id(index), self.b.id(index))
    }

    fn index(&self, id: &Self::Symbol) -> usize {
        (0..self.a.len()).find(|&i| self.a.id(i) == id.0 && self.b.id(i) == id.1).
            expect(&format!("Orbit id not found: {id:?}"))
    }
}

impl<A: OrdOrbitCodec, B: OrdOrbitCodec> ProductOrbitCodec<A, B> {
    fn new(a: A, b: B) -> Self {
        assert_eq!(a.len(), b.len());
        let mut counts = FHashMap::default();
        for index in 0..a.len() {
            *counts.entry((a.id(index), b.id(index))).or_insert(0) += 1;
        }
        Self { a, b, codec: AvlCategorical::from_iter(counts) }
    }
}

pub trait OrdOrbitCodecs<P: PrefixingChain>: PrefixFn<P, Output: OrdOrbitCodec> {}
impl<P: PrefixingChain, O: PrefixFn<P, Output: OrdOrbitCodec>> OrdOrbitCodecs<P> for O {}

impl<PA: PrefixingChain, PB: PrefixingChain, A: OrdOrbitCodecs<PA>, B: OrdOrbitCodecs<PB>> UncachedPrefixFn<(PA, PB)> for (A, B) {
    type Output = ProductOrbitCodec<A::Output, B::Output>;

    fn apply(&self, x: &(PA::Prefix, PB::Prefix)) -> Self::Output {
        ProductOrbitCodec::new(self.0.apply(&x.0), self.1.apply(&x.1))
    }
}
