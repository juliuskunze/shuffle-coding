use std::fmt::Debug;
use std::iter::empty;
use std::mem;
use std::mem::size_of;
use std::ops::{Deref, DerefMut};

use itertools::Itertools;
use lazy_static::lazy_static;
use rand::{Rng, SeedableRng, thread_rng};
use rand::prelude::SliceRandom;
use rand_pcg::Pcg64Mcg;

use crate::multiset::OrdSymbol;

type Head = u64;
type TailElement = u8;

const HEAD_PREC: usize = size_of::<Head>() * 8;
const TAIL_PREC: usize = size_of::<TailElement>() * 8;
const MAX_MIN_HEAD: Head = 1 << (HEAD_PREC - TAIL_PREC);
/// Uniform.max() would have inaccurate bits() values for larger sizes,
/// so we disallow them to simplify testing.
const MAX_SIZE: usize = (MAX_MIN_HEAD >> 10) as usize;


/// Portable PRNG with "good" performance on statistical tests.
/// 16 bytes of state, 7 GB/s, compared to StdRng's 136 bytes of state, 1.5 GB/s
/// according to https://rust-random.github.io/book/guide-rngs.html.
/// Initialization is ~20x faster than StdRng.
/// Using StdRng made tests that create many messages annoyingly slow.
/// Fast message creation is tested by `tests::create_random_message_fast`.
type TailRng = Pcg64Mcg;

#[derive(Clone, Debug)]
pub enum TailGenerator {
    Random { rng: TailRng, seed: usize },
    Zeros,
    Empty,
}


impl TailGenerator {
    fn pop(&mut self) -> TailElement {
        match self {
            Self::Random { rng, .. } => { rng.gen() }
            Self::Zeros => 0,
            Self::Empty => panic!("Message exhausted whilst attempting decode.")
        }
    }

    fn reset_clone(&self) -> Self {
        match self {
            Self::Random { seed, .. } => Self::random(*seed),
            Self::Zeros => Self::Zeros,
            Self::Empty => Self::Empty,
        }
    }

    fn random(seed: usize) -> Self {
        Self::Random { rng: TailRng::seed_from_u64(seed as u64), seed }
    }
}

impl Iterator for TailGenerator {
    type Item = TailElement;
    fn next(&mut self) -> Option<Self::Item> { Some(self.pop()) }
}

#[derive(Clone, Debug)]
pub struct Tail {
    elements: Vec<TailElement>,
    generator: TailGenerator,
    num_generated: usize,
}

impl PartialEq for Tail {
    fn eq(&self, other: &Self) -> bool {
        let mut a = self.clone();
        let mut b = other.clone();
        a.normalize();
        b.normalize();

        a.elements == b.elements && a.num_generated == b.num_generated && match (&a.generator, &b.generator) {
            (TailGenerator::Random { seed, .. }, TailGenerator::Random { seed: s, .. }) => seed == s,
            (TailGenerator::Zeros, TailGenerator::Zeros) => true,
            (TailGenerator::Empty, TailGenerator::Empty) => true,
            _ => false,
        }
    }
}

impl Eq for Tail {}

impl Tail {
    fn new(elements: impl IntoIterator<Item=TailElement>, generator: TailGenerator) -> Self {
        Self { elements: elements.into_iter().collect(), generator, num_generated: 0 }
    }

    fn push(&mut self, element: TailElement) {
        self.elements.push(element);
    }

    fn pop(&mut self) -> TailElement {
        self.elements.pop().unwrap_or_else(|| {
            self.num_generated += 1;
            self.generator.pop()
        })
    }

    fn len_minus_generated(&self) -> isize { self.elements.len() as isize - self.num_generated as isize }

    fn normalize(&mut self) {
        if self.num_generated == 0 { return; }

        let mut generated = self.generator.reset_clone().
            take(self.num_generated).collect_vec();
        generated.reverse();
        let num_ungenerated = generated.iter().zip(self.elements.iter()).
            take_while(|(g, e)| g == e).count();

        self.elements.drain(..num_ungenerated);
        self.num_generated -= num_ungenerated;
        self.generator = self.generator.reset_clone();
        for _ in 0..self.num_generated {
            self.generator.pop();
        }
    }
}

#[derive(Clone, Debug)]
pub struct Message {
    pub head: Head,
    pub tail: Tail,
}

impl Message {
    /// Shifts bits between head and tail until min_head <= head < min_head << TAIL_PREC.
    fn renorm(&mut self, min_head: Head) {
        self.renorm_up(min_head);
        self.renorm_down(min_head);
    }

    /// Shifts bits from tail to head until min_head <= head.
    fn renorm_up(&mut self, min_head: Head) {
        while self.head < min_head {
            self.head = self.head << TAIL_PREC | self.tail.pop() as Head
        }
    }

    /// Shifts bits from head to tail until head < min_head << TAIL_PREC.
    fn renorm_down(&mut self, min_head: Head) {
        loop {
            let new_head = self.head >> TAIL_PREC;
            if new_head < min_head { break; }
            self.tail.push(self.head as TailElement);
            self.head = new_head;
        }
    }

    pub fn flatten(mut self) -> Tail {
        self.renorm_down(1);
        let mut tail = self.tail;
        tail.push(self.head as TailElement);
        tail
    }

    pub fn unflatten(tail: Tail) -> Message {
        Message { head: 0, tail }
    }

    /// Actual number of bits to be sent/stored.
    pub fn bits(&self) -> usize {
        TAIL_PREC * self.clone().flatten().elements.len()
    }

    /// Precise virtual message length in bits where the head is counted as log(head) and
    /// generated tail is subtracted. Fractional in general and can be negative.
    /// The increase in virtual bits when pushing a symbol is its info content under the used codec.
    pub fn virtual_bits(&self) -> f64 {
        let mut clone: Message;
        let message = if self.head > 1 << 32 { self } else {
            clone = self.clone();
            // Avoid inaccuracy for small messages:
            clone.renorm_up(MAX_MIN_HEAD);
            &clone
        };
        (message.head as f64).log2() + (TAIL_PREC as isize * message.tail.len_minus_generated()) as f64
    }

    pub fn random(seed: usize) -> Message {
        let tail = Tail::new(empty(), TailGenerator::random(seed));
        let mut m = Message { head: 1, tail };
        m.renorm_up(MAX_MIN_HEAD);
        m
    }

    pub fn zeros() -> Self {
        Self { head: MAX_MIN_HEAD, tail: Tail::new(empty(), TailGenerator::Zeros) }
    }

    #[allow(unused)]
    pub fn empty() -> Self {
        Self { head: MAX_MIN_HEAD, tail: Tail::new(empty(), TailGenerator::Empty) }
    }
}

impl PartialEq for Message {
    fn eq(&self, other: &Self) -> bool {
        let mut m = self.clone();
        m.renorm(MAX_MIN_HEAD);
        let mut o = other.clone();
        o.renorm(MAX_MIN_HEAD);
        m.tail == o.tail && m.head == o.head
    }
}

pub trait Symbol: Clone + Debug + Eq {}

impl<T: Clone + Debug + Eq> Symbol for T {}

pub trait Codec: Clone {
    type Symbol: Symbol;
    fn push(&self, m: &mut Message, x: &Self::Symbol);
    fn pop(&self, m: &mut Message) -> Self::Symbol;
    /// Code length for the given symbol in bits if deterministic and known, None otherwise.
    fn bits(&self, x: &Self::Symbol) -> Option<f64>;

    fn sample(&self, seed: usize) -> Self::Symbol {
        self.pop(&mut Message::random(seed))
    }

    fn samples(&self, len: usize, seed: usize) -> Vec<Self::Symbol> {
        IID::new(self.clone(), len).sample(seed)
    }

    /// Any implementation should pass this test for any seed and valid symbol x.
    fn test_invertibility(&self, x: &Self::Symbol, initial: &Message) -> CodecTestResults {
        let m = &mut initial.clone();
        let enc_sec = timeit_loops!(1, { self.push(m, x) });
        let bits = m.bits();
        let amortized_bits = m.virtual_bits() - initial.virtual_bits();
        assert!(bits as f64 >= amortized_bits);
        let mut decoded: Option<Self::Symbol> = None;
        let dec_sec = timeit_loops!(1, { decoded = Some(self.pop(m)) });
        assert_eq!(x, &decoded.unwrap());
        assert_eq!(initial, m);
        assert_eq!(initial, &Message::unflatten(m.clone().flatten()));
        CodecTestResults { bits, amortized_bits, enc_sec, dec_sec }
    }

    /// Any implementation should pass this test for any seed and valid symbol x.
    fn test(&self, x: &Self::Symbol, initial: &Message) -> CodecTestResults {
        let out = self.test_invertibility(x, initial);
        if let Some(amortized_bits) = self.bits(x) {
            assert_bits_eq(amortized_bits, out.amortized_bits);
        }
        out
    }

    /// Any implementation should pass this test for any num_samples.
    fn test_on_samples(&self, num_samples: usize) -> Vec<f64> {
        (0..num_samples).map(|seed| self.test(&self.sample(seed), &Message::random(seed))).map(|r| r.amortized_bits).collect()
    }
}

/// Codec with a uniform distribution. All codes have the same length.
pub trait UniformCodec: Codec {
    /// Codec length for all symbols in bits.
    fn uni_bits(&self) -> f64;
}

pub struct CodecTestResults {
    pub bits: usize,
    pub amortized_bits: f64,
    pub enc_sec: f64,
    pub dec_sec: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Uniform {
    pub size: usize,
    max_min_head_div_size: Head,
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
    pub fn new(size: usize) -> Self {
        assert!(size <= MAX_SIZE);
        let max_min_head_div_size = MAX_MIN_HEAD / size as Head;
        Self { size, max_min_head_div_size }
    }

    pub fn max() -> &'static Self {
        lazy_static! {static ref C: Uniform = Uniform::new(MAX_SIZE);}
        &C
    }
}

impl UniformCodec for Uniform {
    fn uni_bits(&self) -> f64 {
        (self.size as f64).log2()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Categorical {
    /// None if the probability mass for a symbol is 0.
    pub masses: Vec<usize>,
    pub cummasses: Vec<usize>,
    pub norm: usize,
}

impl Distribution for Categorical {
    type Symbol = usize;

    fn norm(&self) -> usize { self.norm }
    fn pmf(&self, x: &Self::Symbol) -> usize { self.masses[*x] }
    fn cdf(&self, x: &Self::Symbol, i: usize) -> usize { self.cummasses[*x] + i }
    fn icdf(&self, cf: usize) -> (Self::Symbol, usize) {
        let x = self.cummasses.partition_point(|&c| c <= cf) - 1;
        (x as Self::Symbol, cf - self.cummasses[x])
    }
}

impl Categorical {
    pub fn new(masses: Vec<usize>) -> Self {
        let cummasses = masses.iter().scan(0, |acc, &x| {
            let out = Some(*acc);
            *acc += x;
            out
        }).collect();
        let norm = masses.iter().sum();
        Self { masses, cummasses, norm }
    }

    pub fn prob(&self, x: usize) -> f64 {
        self.masses[x] as f64 / self.norm() as f64
    }

    pub fn entropy(&self) -> f64 {
        (0..self.masses.len()).map(|x| {
            let p = self.prob(x);
            if p == 0. { 0. } else { -p.log2() * p }
        }).sum::<f64>()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Bernoulli {
    pub categorical: Categorical,
}

impl Distribution for Bernoulli {
    type Symbol = bool;

    fn norm(&self) -> usize {
        self.categorical.norm
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
        Self { categorical: Categorical::new(vec![norm - mass, mass]) }
    }
}

/// Categorical allowing insertion and removal of mass.
/// Insert, remove, pmf, cdf and icdf all have a runtime of O(log #symbols).
/// Implemented via an order statistic tree.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MutCategorical<S: OrdSymbol + Default = usize> {
    branches: Option<Box<(Self, Self)>>,
    count: usize,
    split: S,
}

impl<S: OrdSymbol + Default> Distribution for MutCategorical<S> {
    type Symbol = S;

    fn norm(&self) -> usize {
        self.count
    }

    fn pmf(&self, x: &Self::Symbol) -> usize {
        if let Some(branches) = &self.branches {
            let (left, right) = branches.deref();
            if x < &self.split { left } else { right }.pmf(x)
        } else {
            if x == &self.split { self.count } else { 0 }
        }
    }

    fn cdf(&self, x: &Self::Symbol, i: usize) -> usize {
        if let Some(branches) = &self.branches {
            let (left, right) = branches.deref();
            if x < &self.split { left.cdf(x, i) } else { left.count + right.cdf(x, i) }
        } else {
            assert_eq!(&self.split, x, "Symbol {x:?} not found in distribution.");
            assert!(i < self.count);
            i
        }
    }

    fn icdf(&self, cf: usize) -> (Self::Symbol, usize) {
        if let Some(branches) = &self.branches {
            let (left, right) = branches.deref();
            if cf < left.count { left.icdf(cf) } else { right.icdf(cf - left.count) }
        } else {
            assert!(cf < self.count);
            (self.split.clone(), cf)
        }
    }
}

impl<S: OrdSymbol + Default> MutDistribution for MutCategorical<S> {
    fn insert(&mut self, x: Self::Symbol, mass: usize) {
        let is_left = x < self.split;
        if let Some(branches) = &mut self.branches {
            let (left, right) = branches.deref_mut();
            if is_left { left } else { right }.insert(x, mass);
        } else {
            if x != self.split {
                let new = Self::leaf(x.clone(), mass);
                let prev = Self::leaf(self.split.clone(), self.count);
                self.branches = Some(Box::new(if is_left {
                    (new, prev)
                } else {
                    self.split = x;
                    (prev, new)
                }));
            }
        }
        self.count += mass;
    }

    fn remove(&mut self, x: &Self::Symbol, mass: usize) {
        assert!(mass <= self.count);
        self.count -= mass;

        if let Some(branches) = &mut self.branches {
            let (left, right) = branches.deref_mut();
            let is_left = x < &self.split;
            let tree = if is_left { left } else { right };
            tree.remove(x, mass);
            if tree.count == 0 {
                let (left, right) = branches.deref_mut();
                let remaining = if is_left { right } else { left };
                let dummy = Self::leaf(self.split.clone(), 0);
                *self = mem::replace(remaining, dummy);
            }
        }
    }
}

impl<S: OrdSymbol + Default> MutCategorical<S> {
    fn leaf(x: S, count: usize) -> Self {
        Self { branches: None, count, split: x }
    }

    pub fn new(iter: impl IntoIterator<Item=(S, usize)>, shuffle: bool) -> Self {
        let mut all = iter.into_iter().collect_vec();
        if shuffle {
            all.shuffle(&mut thread_rng());
        }
        let mut out: Option<Self> = None;
        for (x, count) in all {
            if let Some(out) = &mut out {
                out.insert(x, count);
            } else {
                out = Some(Self::leaf(x, count));
            }
        }
        out.unwrap_or_else(Self::default)
    }
}


#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VecCodec<C: Codec> {
    pub codecs: Vec<C>,
}

impl<C: Codec> Codec for VecCodec<C> {
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

impl<C: UniformCodec> UniformCodec for VecCodec<C> {
    fn uni_bits(&self) -> f64 {
        self.codecs.iter().map(|c| c.uni_bits()).sum()
    }
}

impl<C: Codec> VecCodec<C> {
    pub fn new(codecs: impl IntoIterator<Item=C>) -> Self { Self { codecs: codecs.into_iter().collect() } }
}

pub fn assert_bits_eq(expected_bits: f64, bits: f64) {
    assert_bits_close(expected_bits, bits, 1e-5);
}

pub fn assert_bits_close(expected_bits: f64, bits: f64, tol: f64) {
    let mismatch = (bits - expected_bits).abs() / expected_bits.abs().max(1.);
    assert!(mismatch < tol, "Expected {} bits, but got {} bits.", expected_bits, bits);
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
    fn push(&self, _: &mut Message, x: &Self::Symbol) { assert_eq!(x, &self.0); }
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
            EnumCodec::A(a) => a.push(m, x),
            EnumCodec::B(b) => b.push(m, x),
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        match self {
            EnumCodec::A(a) => a.pop(m),
            EnumCodec::B(b) => b.pop(m),
        }
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        match self {
            EnumCodec::A(a) => a.bits(x),
            EnumCodec::B(b) => b.bits(x),
        }
    }
}

#[derive(Clone, Debug)]
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

pub trait Distribution: Clone {
    type Symbol: Symbol;

    fn norm(&self) -> usize;
    fn pmf(&self, x: &Self::Symbol) -> usize;
    fn cdf(&self, x: &Self::Symbol, i: usize) -> usize;
    fn icdf(&self, cf: usize) -> (Self::Symbol, usize);
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

#[derive(Clone, Debug)]
pub struct BoxCodec<T: Codec>(Box<T>);

impl<T: Codec> BoxCodec<T> {
    pub fn new(x: T) -> Self { Self(Box::new(x)) }
}

impl<T: Codec + ?Sized> Codec for BoxCodec<T> {
    type Symbol = T::Symbol;
    fn push(&self, m: &mut Message, x: &Self::Symbol) { self.0.push(m, x) }
    fn pop(&self, m: &mut Message) -> Self::Symbol { self.0.pop(m) }
    fn bits(&self, x: &Self::Symbol) -> Option<f64> { self.0.bits(x) }
}

impl<D: Distribution> Codec for D {
    type Symbol = D::Symbol;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let p = self.pmf(x) as Head;
        let norm = self.norm() as Head;
        m.renorm(p * (MAX_MIN_HEAD / norm));
        let h_div_p = m.head / p;
        let h_mod_p = m.head % p;
        let i = self.cdf(x, h_mod_p as usize) as Head;
        m.head = norm * h_div_p + i
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let norm = self.norm() as Head;
        m.renorm(norm * (MAX_MIN_HEAD / norm));
        let h_div_p = m.head / norm;
        let i = m.head % norm;
        let (x, h_mod_p) = self.icdf(i as usize);
        let p = self.pmf(&x) as Head;
        m.head = p * h_div_p + h_mod_p as Head;
        return x;
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        Some(Uniform::new(self.norm()).uni_bits() - Uniform::new(self.pmf(x)).uni_bits())
    }
}

#[derive(Clone, Debug)]
pub struct Benford {
    pub bits: Uniform,
}

impl Codec for Benford {
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

impl Benford {
    pub fn new(excl_max_bits: usize) -> Self {
        assert!(excl_max_bits <= size_of::<<Self as Codec>::Symbol>() * 8);
        let size = excl_max_bits + 1;
        Self { bits: Uniform::new(size) }
    }

    pub fn max() -> &'static Self {
        lazy_static! {static ref C: Benford = Benford::new(47);}
        &C
    }

    pub fn get_bits(x: &usize) -> usize {
        size_of::<<Self as Codec>::Symbol>() * 8 - x.leading_zeros() as usize
    }
}

#[cfg(test)]
pub mod tests {
    use std::iter::repeat;

    use super::*;

    fn assert_entropy_eq(expected_entropy: f64, entropy: f64) {
        assert_bits_close(expected_entropy, entropy, 0.02);
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
        let c = Categorical::new(masses);
        assert_entropy_eq(c.entropy(), c.test_on_samples(NUM_SAMPLES).iter().sum::<f64>() / NUM_SAMPLES as f64);
        test_bernoulli(2, 10);
        test_bernoulli(0, 10);
        test_bernoulli(10, 10);
        Uniform::new(1 << 28).test_on_samples(NUM_SAMPLES);
        IID::new(Uniform::new(1 << 28), 2).test_on_samples(NUM_SAMPLES);
        VecCodec::new(repeat(Uniform::new(1 << 28)).take(2)).test_on_samples(NUM_SAMPLES);
    }

    fn test_bernoulli(mass: usize, norm: usize) {
        let c = Bernoulli::new(mass, norm);
        assert_entropy_eq(c.categorical.entropy(), c.test_on_samples(NUM_SAMPLES).iter().sum::<f64>() / NUM_SAMPLES as f64);
    }

    #[test]
    fn truncated_benford() {
        let c = Benford::new(8);
        for i in 0..255 {
            c.test(&i, &Message::random(0));
        }
    }

    #[test]
    fn tree_multiset() {
        let mut dist = MutCategorical::new(vec![0, 1, 2, 4, 0, 0, 1, 0, 4].into_iter().map(|x| (x, 1)), true);
        assert_eq!(dist.norm(), 9);

        assert_eq!(dist.pmf(&0), 4);
        assert_eq!(dist.pmf(&1), 2);
        assert_eq!(dist.pmf(&2), 1);
        assert_eq!(dist.pmf(&3), 0);
        assert_eq!(dist.pmf(&4), 2);

        assert_eq!(dist.cdf(&0, 0), 0);
        assert_eq!(dist.cdf(&1, 0), 4);
        assert_eq!(dist.cdf(&2, 0), 6);
        assert_eq!(dist.cdf(&4, 0), 7);

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
    fn test_fail() {
        let dist = MutCategorical::new(vec![(0, 1), (1, 2), (2, 2)], true);
        assert_eq!(dist.norm(), 5);
        for seed in 0..500 {
            dist.test(&2, &Message::random(seed));
        }
        dist.test_on_samples(50);
    }
}
