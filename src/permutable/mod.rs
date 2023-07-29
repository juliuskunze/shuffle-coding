//! Ordered objects / permutable classes.
use crate::codec::{Codec, EqSymbol, Message, OrdSymbol, Symbol, Uniform, UniformCodec};
use float_extras::f64::lgamma;
use fxhash::FxBuildHasher;
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::ops::{Deref, Mul};

pub mod graph;
pub mod multiset;

pub(crate) type FBuildHasher = FxBuildHasher;
pub(crate) type FHashMap<K, V> = HashMap<K, V, FBuildHasher>;
pub(crate) type FHashSet<K> = HashSet<K, FBuildHasher>;

/// An ordered object. Each type implementing this represents a permutable class.
pub trait Permutable: Clone {
    /// Length of the ordered object. The order of the corresponding permutable class.
    fn len(&self) -> usize;

    /// Swap two indices of the ordered object.
    fn swap(&mut self, i: usize, j: usize);

    /// Permutes the ordered object by a given permutation.
    /// By default, this is implemented based on swap.
    fn permuted(&self, x: &impl Permutation) -> Self {
        self._permuted_from_swap(x)
    }

    fn _permuted_from_swap<P: Permutation>(&self, x: &P) -> Self {
        assert_eq!(self.len(), x.len());
        let mut p = P::identity(self.len());
        let mut p_inv = P::identity(self.len());
        let mut out = self.clone();
        for i in (0..self.len()).rev() {
            let xi = x.apply(i);
            let j = p_inv.apply(xi);
            let pi = p.apply(i);
            p_inv.swap(pi, xi);
            out.swap(pi, xi);
            p.swap(i, j);
        }

        out
    }

    /// Shuffles the ordered object by a random permutation uniformly sampled with the given seed.
    fn shuffled(&self, seed: usize) -> Self {
        self.permuted(&PermutationUniform::<Perm>::new(self.len()).sample(seed))
    }

    /// Test left group action axioms for this ordered object.
    /// Any implementation should pass this test for any ordered object and seed.
    fn test_left_group_action_axioms(&self, seed: usize)
    where
        Self: EqSymbol,
    {
        let (p0, p1) = &PermutationUniform::<Perm>::new(self.len()).samples(2, seed).into_iter().collect_tuple().unwrap();
        assert_eq!((p0 * self).len(), self.len());
        assert_eq!(&(&Perm::identity(self.len()) * self), self);
        assert_eq!(p1 * &(p0 * self), &(p1 * p0) * self);
    }
}

/// The unordered object corresponding to a given ordered object.
/// The main difference is that equality is based on isomorphism.
#[derive(Clone, Debug, Hash)]
pub struct Unordered<P: Permutable>(pub P);

impl<P: Permutable> Unordered<P> {
    pub fn into_ordered(self) -> P { self.0 }
    pub fn to_ordered(&self) -> &P { &self.0 }
    pub fn len(&self) -> usize { self.to_ordered().len() }
}

/// Permutation a given length.
pub trait Permutation: Permutable + From<Vec<usize>> + Symbol {
    fn identity(len: usize) -> Self;

    fn inverse(&self) -> Self;

    fn iter(&self) -> impl Iterator<Item=usize> + '_;

    fn apply(&self, i: usize) -> usize;

    fn is_identity(&self) -> bool {
        self.iter().enumerate().all(|(i, x)| i == x)
    }

    fn create_swap(len: usize, i: usize, j: usize) -> Self {
        let mut out = Self::identity(len);
        out.swap(i, j);
        out
    }
}

/// Permutation a given length, densely represented.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Perm(pub Vec<usize>);

impl Permutable for Perm {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.0.swap(i, j);
    }

    fn permuted(&self, x: &impl Permutation) -> Self {
        Perm((0..self.len()).map(|i| x.apply(self.0[i])).collect())
    }
}

impl From<Vec<usize>> for Perm {
    fn from(items: Vec<usize>) -> Self {
        debug_assert!(items.iter().all(|&x| x < items.len()));
        Self(items)
    }
}

impl Permutation for Perm {
    fn identity(len: usize) -> Self {
        Self((0..len).collect())
    }

    fn inverse(&self) -> Self {
        let mut out = vec![0; self.0.len()];
        for (i, &x) in self.0.iter().enumerate() {
            out[x] = i;
        }
        Self(out)
    }

    fn iter(&self) -> impl Iterator<Item=usize> + '_ {
        self.0.iter().cloned()
    }

    fn apply(&self, i: usize) -> usize {
        self.0[i]
    }
}

impl Mul<usize> for &Perm {
    type Output = usize;

    fn mul(self, rhs: usize) -> usize { self.apply(rhs) }
}

impl<P: Permutable> Mul<&P> for &Perm {
    type Output = P;

    fn mul(self, rhs: &P) -> P { P::permuted(rhs, self) }
}

impl<P: Permutable, Q: Permutable> Permutable for (P, Q) {
    fn len(&self) -> usize {
        let s = self.0.len();
        assert_eq!(s, self.1.len());
        s
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.0.swap(i, j);
        self.1.swap(i, j);
    }
}

/// Pop is the modern version of the Fisher-Yates shuffle, and push its inverse. Both are O(len).
#[derive(Clone)]
pub struct PermutationUniform<P: Permutation = Perm> {
    pub len: usize,
    phantom: PhantomData<P>,
}

impl<P: Permutation + Symbol> Codec for PermutationUniform<P> {
    type Symbol = P;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let mut p = P::identity(self.len);
        let mut p_inv = P::identity(self.len);
        let mut js = vec![];
        for i in (0..self.len).rev() {
            let xi = x.apply(i);
            let j = p_inv.apply(xi);
            p_inv.swap(p.apply(i), xi);
            p.swap(i, j);
            js.push(j);
        }

        for (i, j) in js.iter().rev().enumerate() {
            let size = i + 1;
            Uniform::new(size).push(m, j);
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let mut p = P::identity(self.len);

        for i in (0..self.len).rev() {
            let size = i + 1;
            let j = Uniform::new(size).pop(m);
            p.swap(i, j);
        }

        p
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { Some(self.uni_bits()) }
}

impl<P: Permutation + Symbol> UniformCodec for PermutationUniform<P> {
    fn uni_bits(&self) -> f64 {
        lgamma((self.len + 1) as f64) / 2f64.ln()
    }
}

impl<P: Permutation> PermutationUniform<P> {
    pub fn new(len: usize) -> Self { Self { len, phantom: PhantomData } }
}

/// Len determines the length of a permutation in a permutable object.
/// Needs to be known upfront in some places, for example for decoding with autoregressive shuffle coding.
pub trait Len {
    fn len(&self) -> usize;
}

/// Codec for objects from a single permutable class of known length.
pub trait PermutableCodec: Codec<Symbol: Permutable> + Len {}

impl<C: Codec<Symbol: Permutable> + Len> PermutableCodec for C {}

#[cfg(test)]
pub mod tests {
    use crate::codec::Codec;
    use crate::permutable::{Perm, Permutation, PermutationUniform};

    #[test]
    fn perm_and_uniform() {
        test_perm_and_uniform::<Perm>()
    }

    pub fn test_perm_and_uniform<P: Eq + Permutation>() {
        let uniform = PermutationUniform::<P>::new(9);
        uniform.test(&P::from(vec![8, 5, 0, 2, 1, 3, 4, 6, 7]), 0);
        for perm in uniform.samples(100, 0) {
            uniform.test(&perm, 0);
            perm.test_left_group_action_axioms(0);
        }
    }
}

/// Ordered object with known orbits of its automorphism group.
pub trait Orbits {
    type Id: OrdSymbol;
    /// Return the orbits of the object's automorphism group.
    fn orbit_ids(&self) -> impl Deref<Target=Vec<Self::Id>>;
}

/// Map to a permutable object that is at least as symmetric as the input, in the sense that
/// the automorphism group of the input is a subgroup of the output's automorphism group.
pub trait Hashing<P>: Symbol {
    type Hash: Permutable;

    fn apply(&self, x: &P) -> Self::Hash;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Complete;

impl<P: Orbits> Hashing<P> for Complete {
    type Hash = Vec<P::Id>;

    fn apply(&self, x: &P) -> Self::Hash {
        x.orbit_ids().clone()
    }
}
