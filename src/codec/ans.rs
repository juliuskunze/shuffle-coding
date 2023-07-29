//! Implements the range variant of asymmetric numeral systems (ANS). 
//! For a tutorial, see https://arxiv.org/pdf/2001.09186.
//! The original paper introducing ANS is at https://arxiv.org/pdf/0902.0271.
use crate::codec::{Uniform, IID};
use itertools::Itertools;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use std::fmt::Debug;
use std::iter::empty;
#[cfg(any(test, feature = "bench"))]
use timeit::timeit_loops;

pub type Head = u64;
pub type TailElement = u8;

pub const HEAD_PREC: usize = size_of::<Head>() * 8;
pub const TAIL_PREC: usize = size_of::<TailElement>() * 8;
pub const MAX_MIN_HEAD: Head = 1 << (HEAD_PREC - TAIL_PREC);
/// Uniform.max() would have inaccurate bits() values for larger sizes,
/// so we disallow them to simplify testing.
pub const MAX_SIZE: usize = (MAX_MIN_HEAD >> 10) as usize;

pub trait Symbol: Clone + Debug + Sync + Send {}
impl<T: Clone + Debug + Sync + Send> Symbol for T {}

pub trait EqSymbol: Eq + Symbol {}
impl<T: Eq + Symbol> EqSymbol for T {}

/// A codec based on asymmetric numeral systems (ANS), encoding and decoding symbols of a given type.
pub trait Codec {
    /// The symbol type that the codec encodes and decodes.
    type Symbol: Symbol;
    /// Encode the given symbol onto a given message.
    fn push(&self, m: &mut Message, x: &Self::Symbol);
    /// Decode a symbol from the given message.
    fn pop(&self, m: &mut Message) -> Self::Symbol;
    /// Code length for the given symbol in bits if deterministic and known, None otherwise.
    fn bits(&self, x: &Self::Symbol) -> Option<f64>;

    fn sample(&self, seed: usize) -> Self::Symbol {
        self.pop(&mut Message::random(seed))
    }

    fn samples(&self, len: usize, seed: usize) -> Vec<Self::Symbol>
    where
        Self: Clone,
    {
        IID::new(self.clone(), len).sample(seed)
    }

    /// Any implementation should pass this test for any valid symbol x and message.
    #[cfg(any(test, feature = "bench"))]
    fn test_invertibility(&self, x: &Self::Symbol, seed: usize) -> CodecTestResults
    where
        Self::Symbol: Eq,
    {
        let initial = &Message::random(seed);
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
    #[cfg(any(test, feature = "bench"))]
    fn test(&self, x: &Self::Symbol, seed: usize) -> CodecTestResults
    where
        Self::Symbol: Eq,
    {
        let out = self.test_invertibility(x, seed);
        if let Some(amortized_bits) = self.bits(x) {
            assert_bits_eq(amortized_bits, out.amortized_bits);
        }
        out
    }

    #[cfg(test)]
    /// Any implementation should pass this test for any num_samples.
    fn test_on_samples(&self, num_samples: usize) -> Vec<f64>
    where
        Self::Symbol: Eq,
    {
        (0..num_samples).map(|seed| self.test(&self.sample(seed), seed)).map(|r| r.amortized_bits).collect()
    }
}

/// A distribution is a special ANS codec based on a probability mass function (pmf),
/// a cumulative distribution function (cdf), and its inverse (icdf).
/// Used for all elementary and some advanced custom codecs.
pub trait Distribution {
    type Symbol: Symbol;

    /// The normalization constant for the probability distribution.
    fn norm(&self) -> usize;
    /// The probability mass for the given symbol.
    fn pmf(&self, x: &Self::Symbol) -> usize;
    /// The cumulative distribution function for the distribution.
    fn cdf(&self, x: &Self::Symbol, i: usize) -> usize;
    /// The inverse cumulative distribution function for the distribution.
    fn icdf(&self, cf: usize) -> (Self::Symbol, usize);
}

impl<D: Distribution> Codec for D {
    type Symbol = D::Symbol;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let p = self.pmf(x) as Head;
        assert_ne!(p, 0);
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
        x
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        Some(Uniform::new(self.norm()).uni_bits() - Uniform::new(self.pmf(x)).uni_bits())
    }
}

/// Portable PRNG with "good" performance on statistical tests.
/// 16 bytes of state, 7 GB/s, compared to StdRng's 136 bytes of state, 1.5 GB/s
/// according to https://rust-random.github.io/book/guide-rngs.html.
/// Initialization is ~20x faster than StdRng.
/// Using StdRng made tests that create many messages annoyingly slow.
/// Fast message creation is tested by `tests::create_random_message_fast`.
type TailRng = Pcg64Mcg;

#[derive(Clone, Debug, Eq, PartialEq)]
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

#[derive(Clone, Debug, Eq)]
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

#[derive(Clone, Debug, Eq)]
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
        let message = if self.head > 1 << (HEAD_PREC / 2) { self } else {
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

pub fn assert_bits_eq(expected_bits: f64, bits: f64) {
    assert_bits_close(expected_bits, bits, 1e-3);
}

pub fn assert_bits_close(expected_bits: f64, bits: f64, tol: f64) {
    let mismatch = (bits - expected_bits).abs() / expected_bits.abs().max(1.);
    assert!(mismatch < tol, "Expected {} bits, but got {} bits.", expected_bits, bits);
}
