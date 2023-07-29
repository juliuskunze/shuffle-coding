//! Implements various elementary ANS codecs.

mod ans;
pub mod avl;

pub use crate::codec::ans::*;
use ftree::FenwickTree;
use itertools::Itertools;
use std::cell::RefCell;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Deref;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Uniform {
    pub size: usize,
}

impl Distribution for Uniform {
    type Symbol = usize;

    fn norm(&self) -> usize { self.size }
    fn pmf(&self, _x: &Self::Symbol) -> usize { 1 }
    fn cdf(&self, x: &Self::Symbol, i: usize) -> usize {
        assert_eq!(i, 0);
        *x
    }

    fn icdf(&self, cf: usize) -> (Self::Symbol, usize) {
        (cf as Self::Symbol, 0)
    }
}

impl Uniform {
    pub const fn new(size: usize) -> Self {
        assert!(size <= MAX_SIZE);
        Self { size }
    }

    pub fn max() -> Self {
        Self::new(MAX_SIZE)
    }
}

impl UniformCodec for Uniform {
    fn uni_bits(&self) -> f64 {
        (self.size as f64).log2()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Categorical<M: AsRef<[usize]> = Vec<usize>> {
    pub cummasses: M,
}

impl<T: AsRef<[usize]>> Distribution for Categorical<T> {
    type Symbol = usize;

    fn norm(&self) -> usize { *self.cummasses.as_ref().last().unwrap() }
    fn pmf(&self, x: &Self::Symbol) -> usize { self.cummasses.as_ref()[*x + 1] - self.cummasses.as_ref()[*x] }
    fn cdf(&self, x: &Self::Symbol, i: usize) -> usize { self.cummasses.as_ref()[*x] + i }
    fn icdf(&self, cf: usize) -> (Self::Symbol, usize) {
        let x = self.cummasses.as_ref().partition_point(|&c| c <= cf) - 1;
        (x as Self::Symbol, cf - self.cummasses.as_ref()[x])
    }
}

impl<M: AsRef<[usize]>> Categorical<M> {
    pub fn from_iter(masses: M) -> Self
    where
        M: FromIterator<usize>,
    {
        let mut cummass = 0;
        let cummasses = masses.as_ref().iter().chain([&0]).map(|mass| {
            let out = cummass;
            cummass += mass;
            out
        }).collect();
        Self { cummasses }
    }

    pub fn prob(&self, x: usize) -> f64 {
        self.pmf(&x) as f64 / self.norm() as f64
    }

    pub fn entropy(&self) -> f64 {
        (0..self.len()).map(|x| {
            let p = self.prob(x);
            if p == 0. { 0. } else { -p.log2() * p }
        }).sum::<f64>()
    }

    /// A categorical distribution truncated to the first `len` masses.
    pub fn truncated(&self, len: usize) -> Categorical<&[usize]> {
        Categorical {
            cummasses: &self.cummasses.as_ref()[..len.min(self.len()) + 1],
        }
    }

    pub fn len(&self) -> usize {
        self.cummasses.as_ref().len() - 1
    }

    pub fn masses(&self) -> impl Iterator<Item=usize> + '_ {
        self.cummasses.as_ref().iter().tuple_windows().map(|(a, b)| b - a)
    }
}

impl<const NPP: usize> Categorical<[usize; NPP]> {
    pub const fn new_const<const N: usize>(masses: [usize; N]) -> Self {
        if N + 1 != NPP {
            panic!("NPP must be N + 1.");
        }

        let mut cummasses = [0; NPP];
        let mut i = 0;
        while i < N {
            cummasses[i + 1] = cummasses[i] + masses[i];
            i += 1;
        }
        Self { cummasses }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Bernoulli {
    pub categorical: Categorical,
}

impl Distribution for Bernoulli {
    type Symbol = bool;

    fn norm(&self) -> usize {
        self.categorical.norm()
    }

    fn pmf(&self, x: &Self::Symbol) -> usize {
        self.categorical.pmf(&(*x as usize))
    }

    fn cdf(&self, x: &Self::Symbol, i: usize) -> usize {
        self.categorical.cdf(&(*x as usize), i)
    }

    fn icdf(&self, cf: usize) -> (Self::Symbol, usize) {
        let (x, i) = self.categorical.icdf(cf);
        (x != 0, i)
    }
}

impl Bernoulli {
    pub fn prob(&self) -> f64 {
        self.categorical.prob(1)
    }

    pub fn new(mass: usize, norm: usize) -> Self {
        assert!(mass <= norm);
        Self { categorical: Categorical::from_iter(vec![norm - mass, mass]) }
    }
}

pub trait OrdSymbol: Symbol + Ord + Hash {}
impl<T: Symbol + Ord + Hash> OrdSymbol for T {}

/// Codec for a vector of independent symbols with distributions of the same type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Independent<C: Codec> {
    pub codecs: Vec<C>,
}

impl<C: Codec> Codec for Independent<C> {
    type Symbol = Vec<C::Symbol>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        assert_eq!(x.len(), self.codecs.len());
        for (x, codec) in x.iter().zip(&self.codecs).rev() {
            codec.push(m, &x)
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        self.codecs.iter().map(|codec| codec.pop(m)).collect()
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let mut total = 0.;
        for (c, x) in self.codecs.iter().zip_eq(x) {
            total += c.bits(x)?;
        }
        Some(total)
    }
}

impl<C: UniformCodec> UniformCodec for Independent<C> {
    fn uni_bits(&self) -> f64 {
        self.codecs.iter().map(|c| c.uni_bits()).sum()
    }
}

impl<C: Codec> Independent<C> {
    pub fn new(codecs: impl IntoIterator<Item=C>) -> Self { Self { codecs: codecs.into_iter().collect() } }
}

/// Codec with independent and identically distributed symbols.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IID<C: Codec> {
    pub item: C,
    pub len: usize,
}

impl<C: Codec> Codec for IID<C> {
    type Symbol = Vec<C::Symbol>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        assert_eq!(x.len(), self.len);
        for e in x.iter().rev() {
            self.item.push(m, e)
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        (0..self.len).map(|_| self.item.pop(m)).collect()
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let mut total = 0.;
        for x in x.iter() {
            total += self.item.bits(x)?;
        }
        Some(total)
    }
}

impl<C: UniformCodec> UniformCodec for IID<C> {
    fn uni_bits(&self) -> f64 {
        self.len as f64 * self.item.uni_bits()
    }
}

impl<C: Codec> IID<C> {
    pub fn new(item: C, len: usize) -> Self { Self { item, len } }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstantCodec<T: Symbol>(pub T);

impl<T: Symbol> Deref for ConstantCodec<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T: Default + Symbol> Default for ConstantCodec<T> {
    fn default() -> Self { Self(T::default()) }
}

impl<T: Symbol> Codec for ConstantCodec<T> {
    type Symbol = T;
    fn push(&self, _: &mut Message, x: &Self::Symbol) {
        // We do not have T: Eq here, so we use debug representation:
        debug_assert_eq!(format!("{:?}", x), format!("{:?}", &self.0));
    }
    fn pop(&self, _: &mut Message) -> Self::Symbol { self.0.clone() }
    fn bits(&self, _: &Self::Symbol) -> Option<f64> { Some(self.uni_bits()) }
}

impl<T: Symbol> UniformCodec for ConstantCodec<T> {
    fn uni_bits(&self) -> f64 { 0. }
}

impl<A: Codec, B: Codec> Codec for (A, B) {
    type Symbol = (A::Symbol, B::Symbol);

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.1.push(m, &x.1);
        self.0.push(m, &x.0);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let a = self.0.pop(m);
        (a, self.1.pop(m))
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        Some(self.0.bits(&x.0)? + self.1.bits(&x.1)?)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EnumCodec<A: Codec, B: Codec<Symbol=A::Symbol>> {
    A(A),
    B(B),
}

impl<A: Codec, B: Codec<Symbol=A::Symbol>> Codec for EnumCodec<A, B> {
    type Symbol = A::Symbol;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        match self {
            Self::A(a) => a.push(m, x),
            Self::B(b) => b.push(m, x),
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        match self {
            Self::A(a) => a.pop(m),
            Self::B(b) => b.pop(m),
        }
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        match self {
            Self::A(a) => a.bits(x),
            Self::B(b) => b.bits(x),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OptionCodec<C: Codec> {
    pub is_some: Bernoulli,
    pub some: C,
}

impl<C: Codec> Codec for OptionCodec<C> {
    type Symbol = Option<C::Symbol>;
    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        if let Some(x) = x {
            self.some.push(m, x);
        }
        self.is_some.push(m, &x.is_some());
    }
    fn pop(&self, m: &mut Message) -> Self::Symbol {
        if self.is_some.pop(m) {
            Some(self.some.pop(m))
        } else {
            None
        }
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let is_some_bits = self.is_some.bits(&x.is_some());
        if let Some(x) = x {
            Some(is_some_bits? + self.some.bits(x)?)
        } else {
            is_some_bits
        }
    }
}

pub trait MutDistribution: Distribution {
    fn insert(&mut self, x: Self::Symbol, mass: usize);
    fn remove(&mut self, x: &Self::Symbol, mass: usize);

    fn remove_all(&mut self, x: &Self::Symbol) -> usize {
        let count = self.pmf(x);
        self.remove(x, count);
        assert_eq!(self.pmf(x), 0);
        count
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogUniform {
    pub bits: Uniform,
}

impl Codec for LogUniform {
    type Symbol = usize;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let bits = Self::get_bits(x);
        assert!(bits < self.bits.size);
        if bits != 0 {
            let size = 1 << (bits - 1);
            Uniform::new(size).push(m, &(x & !size));
        }
        self.bits.push(m, &bits);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let bits = self.bits.pop(m);
        if bits == 0 { 0 } else {
            let size = 1 << (bits - 1);
            Uniform::new(size).pop(m) | size
        }
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let bits = Self::get_bits(x);
        Some(self.bits.uni_bits() + if bits == 0 { 0. } else {
            let size = 1 << (bits - 1);
            Uniform::new(size).uni_bits()
        })
    }
}

impl LogUniform {
    pub const fn new(excl_max_bits: usize) -> Self {
        assert!(excl_max_bits <= size_of::<<Self as Codec>::Symbol>() * 8);
        let size = excl_max_bits + 1;
        Self { bits: Uniform::new(size) }
    }

    pub fn max() -> &'static Self {
        static C: &'static LogUniform = &LogUniform::new(47);
        C
    }

    pub fn get_bits(x: &usize) -> usize {
        size_of::<<Self as Codec>::Symbol>() * 8 - x.leading_zeros() as usize
    }
}

impl<C: Codec> Codec for RefCell<C> {
    type Symbol = C::Symbol;

    fn push(&self, m: &mut Message, x: &Self::Symbol) { self.borrow().push(m, x) }
    fn pop(&self, m: &mut Message) -> Self::Symbol { self.borrow().pop(m) }
    fn bits(&self, x: &Self::Symbol) -> Option<f64> { self.borrow().bits(x) }
}

impl Distribution for FenwickTree<usize> {
    type Symbol = usize;

    fn norm(&self) -> usize {
        self.prefix_sum(self.len(), 0)
    }

    fn pmf(&self, x: &Self::Symbol) -> usize {
        self.prefix_sum(*x + 1, 0) - self.prefix_sum(*x, 0)
    }

    fn cdf(&self, x: &Self::Symbol, i: usize) -> usize {
        self.prefix_sum(*x, i)
    }

    fn icdf(&self, cf: usize) -> (Self::Symbol, usize) {
        let x = self.index_of(cf + 1);
        (x, cf - self.prefix_sum(x, 0))
    }
}

impl MutDistribution for FenwickTree<usize> {
    fn insert(&mut self, x: Self::Symbol, mass: usize) {
        self.add_at(x, mass);
    }

    fn remove(&mut self, x: &Self::Symbol, mass: usize) {
        self.sub_at(*x, mass);
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::bench::test_and_print;
    use crate::codec::ans::{assert_bits_close, Codec};
    use crate::codec::avl::AvlCategorical;
    use ftree::FenwickTree;
    use std::iter::repeat_n;
    use std::ops::Range;
    use timeit::timeit_loops;

    fn assert_entropy_eq(expected_entropy: f64, entropy: f64) {
        assert_bits_close(expected_entropy, entropy, 0.02);
    }

    pub fn test_and_print_vec<C: Codec<Symbol: Eq>>(codecs: impl IntoIterator<Item=C>, symbols: &Vec<C::Symbol>, seeds: Range<usize>) {
        test_and_print(&Independent::new(codecs), symbols, seeds);
    }

    #[test]
    fn create_random_messages_fast() {
        let sec = timeit_loops!(100000, { Message::random(0); });
        assert!(sec < 5e-6, "{}s is too slow for creating a random message.", sec);
    }

    const NUM_SAMPLES: usize = 1000;

    #[test]
    fn dists() {
        let masses = vec![0, 1, 2, 3, 0, 0, 1, 0];
        let c = Categorical::from_iter(masses.clone());
        assert_entropy_eq(c.entropy(), c.test_on_samples(NUM_SAMPLES).iter().sum::<f64>() / NUM_SAMPLES as f64);
        test_bernoulli(2, 10);
        test_bernoulli(0, 10);
        test_bernoulli(10, 10);
        FenwickTree::from_iter(masses).test_on_samples(10);

        let c = Uniform::max();
        c.test_on_samples(NUM_SAMPLES);
        IID::new(c.clone(), 2).test_on_samples(NUM_SAMPLES);
        Independent::new(repeat_n(c, 2)).test_on_samples(NUM_SAMPLES);
    }

    fn test_bernoulli(mass: usize, norm: usize) {
        let c = Bernoulli::new(mass, norm);
        assert_entropy_eq(c.categorical.entropy(), c.test_on_samples(NUM_SAMPLES).iter().sum::<f64>() / NUM_SAMPLES as f64);
    }

    #[test]
    fn truncated_benford() {
        let c = LogUniform::new(8);
        for i in 0..255 {
            c.test(&i, 0);
        }
    }

    #[test]
    fn empty_mut_categorical() {
        let mut c = AvlCategorical::default();
        assert_eq!(c.iter().collect_vec(), vec![]);
        c.insert(5, 1);
        assert_eq!(c.iter().collect_vec(), vec![(5, 1)]);
        c.remove(&5, 1);
        assert_eq!(c.iter().collect_vec(), vec![]);
        c.insert(3, 1);
        assert_eq!(c.iter().collect_vec(), vec![(3, 1)]);
        c.insert(2, 5);
        c.insert(7, 2);
        assert_eq!(c.iter().collect_vec(), vec![(2, 5), (3, 1), (7, 2)]);
        c.remove(&3, 1);
        assert_eq!(c.iter().collect_vec(), vec![(2, 5), (7, 2)]);
    }

    #[test]
    fn mut_categorical() {
        test_mut_dist_421x2xx0(AvlCategorical::from_iter([(7, 0), (0, 4), (1, 2), (2, 1), (4, 2)]));
    }

    #[test]
    fn fenwick() {
        test_mut_dist_421x2xx0(FenwickTree::from_iter([4, 2, 1, 0, 2, 0, 0, 0]));
    }

    fn test_mut_dist_421x2xx0<D: MutDistribution<Symbol=usize>>(mut dist: D) {
        assert_eq!(dist.norm(), 9);

        assert_eq!(dist.pmf(&0), 4);
        assert_eq!(dist.pmf(&1), 2);
        assert_eq!(dist.pmf(&2), 1);
        assert_eq!(dist.pmf(&4), 2);
        assert_eq!(dist.pmf(&7), 0);

        assert_eq!(dist.cdf(&0, 0), 0);
        assert_eq!(dist.cdf(&0, 1), 1);
        assert_eq!(dist.cdf(&0, 2), 2);
        assert_eq!(dist.cdf(&0, 3), 3);
        assert_eq!(dist.cdf(&1, 0), 4);
        assert_eq!(dist.cdf(&1, 1), 5);
        assert_eq!(dist.cdf(&2, 0), 6);
        assert_eq!(dist.cdf(&4, 0), 7);
        assert_eq!(dist.cdf(&4, 1), 8);

        assert_eq!(dist.icdf(0), (0, 0));
        assert_eq!(dist.icdf(1), (0, 1));
        assert_eq!(dist.icdf(2), (0, 2));
        assert_eq!(dist.icdf(3), (0, 3));
        assert_eq!(dist.icdf(4), (1, 0));
        assert_eq!(dist.icdf(5), (1, 1));
        assert_eq!(dist.icdf(6), (2, 0));
        assert_eq!(dist.icdf(7), (4, 0));
        assert_eq!(dist.icdf(8), (4, 1));

        dist.test_on_samples(50);

        dist.remove(&0, 2);

        assert_eq!(dist.norm(), 7);

        assert_eq!(dist.pmf(&0), 2);
        assert_eq!(dist.pmf(&1), 2);

        assert_eq!(dist.cdf(&0, 0), 0);
        assert_eq!(dist.cdf(&1, 0), 2);
        assert_eq!(dist.cdf(&2, 0), 4);
        assert_eq!(dist.cdf(&4, 0), 5);

        dist.test_on_samples(50);

        dist.remove(&0, 2);
        dist.test_on_samples(50);
        dist.remove(&4, 2);
        assert_eq!(dist.cdf(&1, 0), 0);
        assert_eq!(dist.cdf(&2, 0), 2);
        dist.test_on_samples(50);
    }

    #[test]
    fn mut_categorical_fail() {
        test_fail::<AvlCategorical<usize>>();
    }

    fn test_fail<D: MutDistribution<Symbol=usize> + FromIterator<(usize, usize)>>() {
        let dist = D::from_iter([(0, 1), (1, 2), (2, 2)]);
        assert_eq!(dist.norm(), 5);
        for seed in 0..500 {
            dist.test(&2, seed);
        }
        dist.test_on_samples(50);
    }
}
