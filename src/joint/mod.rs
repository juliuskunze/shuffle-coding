//! Joint shuffle coding.

use std::fmt::Debug;
use std::marker::PhantomData;

#[cfg(test)]
use crate::codec::EqSymbol;
use crate::codec::{Codec, Message, UniformCodec};
#[cfg(test)]
use crate::permutable::PermutationUniform;
use crate::permutable::{Perm, Permutable, Permutation, Unordered};
pub mod multiset;
pub mod incomplete;

pub mod coset;

/// Ordered object with a canonical ordering function allowing to canonical labellings,
/// and therefore canonized objects. Required for joint shuffle coding.
pub trait Canonizable: Permutable {
    /// Returns the canonical ordering of the given object.
    fn canon(&self) -> Perm;

    fn canonized_and_canon(&self) -> (Self, Perm) {
        let c = self.canon();
        (&c * self, c)
    }

    fn canonized(&self) -> Self {
        self.canonized_and_canon().0
    }

    fn is_isomorphic(&self, other: &Self) -> bool
    where
        Self: Eq,
    {
        self.canonized() == other.canonized()
    }

    #[cfg(test)]
    /// Any implementation should pass this test for any seed.
    fn test_canon(&self, seed: usize) -> (Self, Perm)
    where
        Self: EqSymbol,
    {
        self.test_left_group_action_axioms(seed);

        let p = PermutationUniform::new(self.len()).sample(seed);

        // Canonical labelling axiom, expressed in 5 equivalent ways:
        let y = &p * self;
        let y_canon = y.canon();
        assert_eq!(y.permuted(&y_canon), self.permuted(&self.canon()));
        assert_eq!(&y_canon * &y, &self.canon() * self);
        assert_eq!(y.canonized(), self.canonized());
        assert!(y.is_isomorphic(self));
        assert_eq!(Unordered(y.clone()), Unordered(self.clone()));

        (y, y_canon)
    }
}

pub trait RCosetUniformCodec<P: Permutable>: UniformCodec<Symbol: Permutation> {
    fn from_canonized(canonized: &P) -> Self;
}

/// Joint shuffle coding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JointShuffleCodec<
    C: Codec<Symbol: Canonizable>,
    RCosetC: RCosetUniformCodec<C::Symbol>> {
    pub ordered: C,
    pub phantom: PhantomData<RCosetC>,
}

impl<C: Codec<Symbol: Canonizable>, RCosetC: RCosetUniformCodec<C::Symbol>>
Codec for JointShuffleCodec<C, RCosetC> {
    type Symbol = Unordered<C::Symbol>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let canonized = x.canonized();
        let coset_min = RCosetC::from_canonized(&canonized).pop(m);
        let x = canonized.permuted(&coset_min);
        self.ordered.push(m, &x)
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let (canonized, canon) = self.ordered.pop(m).canonized_and_canon();
        let coset_min = canon.inverse();
        RCosetC::from_canonized(&canonized).push(m, &RCosetC::Symbol::from(coset_min.0));
        Unordered(canonized)
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        let canonized = &x.canonized();
        self.ordered.bits(canonized).map(|bits| {
            bits - RCosetC::from_canonized(canonized).uni_bits()
        })
    }
}

impl<P: Eq + Canonizable> PartialEq<Self> for Unordered<P> {
    fn eq(&self, other: &Self) -> bool {
        self.to_ordered().is_isomorphic(other.to_ordered())
    }
}
impl<P: Eq + Canonizable> Eq for Unordered<P> {}

impl<P: Eq + Canonizable> PartialOrd<Self> for Unordered<P> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<P: Eq + Canonizable> Ord for Unordered<P> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_ordered().canon().cmp(&other.to_ordered().canon())
    }
}

impl<P: Canonizable> Unordered<P> {
    pub fn canonized(&self) -> P { self.to_ordered().canonized() }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::bench::TestConfig;
    use crate::codec::tests::test_and_print_vec;
    use crate::experimental::complete_joint::aut::{joint_shuffle_codec, AutCanonizable};
    use crate::experimental::complete_joint::interleaved::interleaved_coset_shuffle_codec;
    use crate::permutable::PermutableCodec;

    pub fn test_complete_joint_shuffle_codecs<C: PermutableCodec<Symbol: Eq + AutCanonizable> + Clone>(
        codecs: &Vec<C>,
        unordered: &Vec<Unordered<C::Symbol>>,
        config: &TestConfig) {
        if let Some(seed) = config.axioms_seed {
            for x in unordered {
                x.to_ordered().test(seed);
            }
        }
        if config.joint {
            if config.complete {
                test_and_print_vec(codecs.iter().cloned().map(joint_shuffle_codec), unordered, config.seeds());
            }
        }
        if config.interleaved_coset_joint {
            test_and_print_vec(codecs.iter().cloned().map(interleaved_coset_shuffle_codec), unordered, config.seeds());
        }
    }
}

