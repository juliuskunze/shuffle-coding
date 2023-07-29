use std::fmt::Debug;
use std::marker::PhantomData;

use crate::codec::{Codec, Message, OrdSymbol, Symbol, UniformCodec};
use crate::graph::{ColorRefinement, EdgeType, Graph};
use crate::joint::multiset::VecRCosetUniform;
use crate::joint::{Canonizable, JointShuffleCodec, RCosetUniformCodec};
use crate::permutable::{Hashing, Perm, Permutable, Permutation, Unordered};
#[derive(Clone, Debug)]
pub struct IncompletelyOrdered<P: Permutable, H: Hashing<P>> {
    inner: P,
    hasher: H,
}

impl<P: Eq + Canonizable, H: Eq + Hashing<P, Hash: Eq>> PartialEq for IncompletelyOrdered<P, H> {
    fn eq(&self, other: &Self) -> bool {
        assert_eq!(self.hasher, other.hasher);
        self.hash() == other.hash() && self.inner.is_isomorphic(&other.inner)
    }
}
impl<P: Eq + Canonizable, H: Eq + Hashing<P, Hash: Eq>> Eq for IncompletelyOrdered<P, H> {}

impl<P: Permutable, H: Hashing<P>> Permutable for IncompletelyOrdered<P, H> {
    fn len(&self) -> usize { self.inner.len() }
    fn swap(&mut self, i: usize, j: usize) {
        self.inner.swap(i, j);
    }

    fn permuted(&self, x: &impl Permutation) -> Self {
        Self::new(self.inner.permuted(x), self.hasher.clone())
    }
}

impl<P: Permutable, H: Hashing<P, Hash: Canonizable>> Canonizable for IncompletelyOrdered<P, H> {
    fn canon(&self) -> Perm {
        self.hasher.apply(&self.inner).canon()
    }
}

impl<P: Permutable, H: Hashing<P>> IncompletelyOrdered<P, H> {
    pub fn new(inner: P, hasher: H) -> Self {
        Self { inner, hasher }
    }

    pub fn hash(&self) -> H::Hash {
        self.hasher.apply(&self.inner)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IncompletelyOrderedCodec<P: Permutable, C: Codec<Symbol=P>, H: Hashing<P>> {
    inner: C,
    hasher: H,
}

impl<P: Permutable + Symbol, C: Codec<Symbol=P>, H: Hashing<P>> Codec for IncompletelyOrderedCodec<P, C, H> {
    type Symbol = IncompletelyOrdered<P, H>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.inner.push(m, &x.inner);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        IncompletelyOrdered::new(self.inner.pop(m), self.hasher.clone())
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.inner.bits(&x.inner)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IncompleteRCosetUniform<P: Permutable, H: Hashing<P>, HashRCosetUniform: RCosetUniformCodec<H::Hash>> {
    pub hash_rcoset_uniform: HashRCosetUniform,
    pub phantom: PhantomData<(P, H)>,
}

impl<P: Permutable, H: Hashing<P>, HashRCosetUniform: RCosetUniformCodec<H::Hash>> RCosetUniformCodec<IncompletelyOrdered<P, H>> for IncompleteRCosetUniform<P, H, HashRCosetUniform> {
    fn from_canonized(canonized: &IncompletelyOrdered<P, H>) -> Self {
        Self { hash_rcoset_uniform: HashRCosetUniform::from_canonized(&canonized.hash()), phantom: PhantomData }
    }
}

impl<P: Permutable, H: Hashing<P>, HashRCosetUniform: RCosetUniformCodec<H::Hash>> Codec for IncompleteRCosetUniform<P, H, HashRCosetUniform> {
    type Symbol = HashRCosetUniform::Symbol;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.hash_rcoset_uniform.push(m, x);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        self.hash_rcoset_uniform.pop(m)
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.hash_rcoset_uniform.bits(x)
    }
}

impl<P: Permutable, H: Hashing<P>, HashRCosetUniform: RCosetUniformCodec<H::Hash>> UniformCodec for IncompleteRCosetUniform<P, H, HashRCosetUniform> {
    fn uni_bits(&self) -> f64 {
        self.hash_rcoset_uniform.uni_bits()
    }
}

/// Implements autoregressive shuffle coding as a wrapper around PrefixShuffleCodec,
/// mapping prefixes of full length to unordered objects and back.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IncompleteJointShuffleCodec<P, C, H, HashRCosetUniform>
where
    P: Permutable + Symbol,
    C: Codec<Symbol=P>,
    H: Hashing<P, Hash: Canonizable>,
    HashRCosetUniform: RCosetUniformCodec<H::Hash>,
{
    pub inner: JointShuffleCodec<IncompletelyOrderedCodec<P, C, H>, IncompleteRCosetUniform<P, H, HashRCosetUniform>>,
}

impl<P, C, H, HashRCosetUniform> Codec for IncompleteJointShuffleCodec<P, C, H, HashRCosetUniform>
where
    P: Permutable + Symbol,
    C: Codec<Symbol=P>,
    H: Hashing<P, Hash: Canonizable>,
    HashRCosetUniform: RCosetUniformCodec<H::Hash>,
{
    type Symbol = Unordered<C::Symbol>;

    fn push(&self, m: &mut Message, Unordered(x): &Self::Symbol) {
        self.inner.push(m, &Unordered(IncompletelyOrdered::new(x.clone(), self.inner.ordered.hasher.clone())));
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        Unordered(self.inner.pop(m).0.inner)
    }

    fn bits(&self, Unordered(x): &Self::Symbol) -> Option<f64> {
        self.inner.bits(&Unordered(IncompletelyOrdered::new(x.clone(), self.inner.ordered.hasher.clone())))
    }
}

fn incomplete_joint_shuffle_codec<P, C, H, HashRCosetUniform>(inner: C, hasher: H) -> IncompleteJointShuffleCodec<P, C, H, HashRCosetUniform>
where
    P: Permutable + Symbol,
    C: Codec<Symbol=P>,
    H: Hashing<P, Hash: Canonizable>,
    HashRCosetUniform: RCosetUniformCodec<H::Hash>,
{
    IncompleteJointShuffleCodec { inner: JointShuffleCodec { ordered: IncompletelyOrderedCodec { inner, hasher }, phantom: PhantomData } }
}

pub fn cr_joint_shuffle_codec<N: OrdSymbol, E: OrdSymbol, Ty: EdgeType, C: Codec<Symbol=Graph<N, E, Ty>>>(inner: C, hasher: ColorRefinement) -> IncompleteJointShuffleCodec<Graph<N, E, Ty>, C, ColorRefinement, VecRCosetUniform<usize>> {
    incomplete_joint_shuffle_codec(inner, hasher)
}
