use crate::codec::{Codec, Message, Symbol, UniformCodec};
use crate::permutable::{Permutation, PermutationUniform};
pub trait StabilizerChainLike {
    type Perm: Permutation;
    type Automorphism: Symbol;

    /// Returns a pair of:
    /// - The lexicographical minimum of the coset containing the given labels.
    /// Can be used as a canonical representative of that coset.
    /// - The automorphism to be applied to this representative to restore the original labels.
    ///
    /// In short: (h, a) where h = min{labels * AUT}, a = h * labels^-1
    fn coset_min_and_restore_aut(&self, labels: Self::Perm) -> (Self::Perm, Self::Automorphism);

    fn coset_min(&self, labels: Self::Perm) -> Self::Perm {
        self.coset_min_and_restore_aut(labels).0
    }

    /// Restore permutation from lexicographical minimum of its coset.
    /// Inverse of coset_min_and_restore_aut.
    fn apply(coset_min: Self::Perm, restore_aut: Self::Automorphism) -> Self::Perm;

    fn len(&self) -> usize;
    fn automorphism_codec(&self) -> impl UniformCodec<Symbol=Self::Automorphism>;

    #[cfg(test)]
    #[allow(unused)]
    fn test(&self, labels: &Self::Perm)
    where
        Self::Perm: Eq,
    {
        let (coset_min, restore_aut) = self.coset_min_and_restore_aut(labels.clone());
        assert_eq!(labels, &Self::apply(coset_min, restore_aut));
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RCosetUniform<S: StabilizerChainLike> {
    pub chain: S,
}

impl<S: StabilizerChainLike> Codec for RCosetUniform<S> {
    type Symbol = S::Perm;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let coset_min = self.chain.coset_min(x.clone());
        let restore_aut = self.chain.automorphism_codec().pop(m);
        let perm = S::apply(coset_min, restore_aut);
        self.permutation_codec().push(m, &perm);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let perm = self.permutation_codec().pop(m);
        let (coset_min, restore_aut) = self.chain.coset_min_and_restore_aut(perm);
        self.chain.automorphism_codec().push(m, &restore_aut);
        coset_min
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { Some(self.uni_bits()) }
}

impl<S: StabilizerChainLike> UniformCodec for RCosetUniform<S> {
    fn uni_bits(&self) -> f64 {
        self.permutation_codec().uni_bits() - self.chain.automorphism_codec().uni_bits()
    }
}

impl<S: StabilizerChainLike> RCosetUniform<S> {
    fn permutation_codec(&self) -> PermutationUniform<S::Perm> {
        PermutationUniform::new(self.chain.len())
    }
}
