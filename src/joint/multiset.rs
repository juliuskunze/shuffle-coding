//! Aut coset codec for Vec, used for joint shuffle coding with multisets.
//! Faster than implementing `AutCanonizable` for `Vec` and using `AutRCosetUniform`.

use crate::autoregressive::chunked::{chunked_hashing_shuffle_codec, ChunkedHashingShuffleCodec};
use crate::autoregressive::multiset::{IIDVecSliceCodecs, VecPrefixingChain};
use crate::codec::{Codec, Message, OrdSymbol, Symbol, UniformCodec, IID};
use crate::joint::coset::{RCosetUniform, StabilizerChainLike};
use crate::joint::{JointShuffleCodec, RCosetUniformCodec};
use crate::permutable::{Complete, Perm, PermutableCodec, Permutation, PermutationUniform};
use itertools::Itertools;
use std::marker::PhantomData;
use std::ops::Range;

pub fn runs<T: Eq>(vec: &[T]) -> impl Iterator<Item=Range<usize>> + '_ {
    let mut start = 0;
    vec.chunk_by(|a, b| a == b).map(move |run| {
        let start_ = start;
        start += run.len();
        start_..start
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VecStabilizerChain<T: Ord> {
    canonized: Vec<T>,
}

impl<T: Ord> StabilizerChainLike for VecStabilizerChain<T> {
    type Perm = Perm;
    type Automorphism = Perm;

    fn coset_min_and_restore_aut(&self, mut labels: Self::Perm) -> (Self::Perm, Self::Automorphism) {
        let mut restore_aut = (0..self.canonized.len()).collect_vec();
        for run in runs(&self.canonized) {
            restore_aut[run.clone()].sort_unstable_by_key(|i| labels.0[*i]);
            labels.0[run].sort_unstable();
        }
        (Perm::from(labels), Perm::from(restore_aut))
    }

    fn coset_min(&self, mut labels: Self::Perm) -> Self::Perm {
        for run in runs(&self.canonized) {
            labels.0[run].sort_unstable();
        }
        labels
    }

    fn apply(coset_min: Self::Perm, restore_aut: Self::Automorphism) -> Self::Perm {
        &coset_min * &restore_aut.inverse()
    }

    fn len(&self) -> usize {
        self.canonized.len()
    }

    fn automorphism_codec(&self) -> impl UniformCodec<Symbol=Self::Automorphism> {
        VecAutomorphismUniform { canon: &self.canonized }
    }
}

impl<T: OrdSymbol> RCosetUniformCodec<Vec<T>> for VecRCosetUniform<T> {
    fn from_canonized(canonized: &Vec<T>) -> Self {
        Self { chain: VecStabilizerChain { canonized: canonized.clone() } }
    }
}

pub type VecRCosetUniform<T> = RCosetUniform<VecStabilizerChain<T>>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VecAutomorphismUniform<'a, T: Ord> {
    canon: &'a Vec<T>,
}

impl<'a, T: Ord> Codec for VecAutomorphismUniform<'a, T> {
    type Symbol = Perm;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        for run in runs(&self.canon).collect_vec().into_iter().rev() {
            let vals = x.0[run.clone()].iter();
            let perm = Perm::from(vals.map(|i| i - run.start).collect_vec());
            PermutationUniform::new(run.len()).push(m, &perm);
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let mut out = vec![0; self.canon.len()];
        for run in runs(&self.canon) {
            let perm: Perm = PermutationUniform::new(run.len()).pop(m);
            let start = run.start;
            let vals = perm.iter().map(|i| start + i);
            out[run].iter_mut().set_from(vals);
        }
        Perm::from(out)
    }

    fn bits(&self, _x: &Self::Symbol) -> Option<f64> { Some(self.uni_bits()) }
}

impl<'a, T: Ord> UniformCodec for VecAutomorphismUniform<'a, T> {
    fn uni_bits(&self) -> f64 {
        runs(&self.canon).map(|run| PermutationUniform::<Perm>::new(run.len()).uni_bits()).sum()
    }
}

pub type MultisetShuffleCodec<S, C> = JointShuffleCodec<C, VecRCosetUniform<S>>;

pub fn joint_multiset_shuffle_codec<S: OrdSymbol, C: PermutableCodec<Symbol=Vec<S>>>(ordered: C) -> MultisetShuffleCodec<S, C> {
    JointShuffleCodec { ordered, phantom: PhantomData }
}

pub type ChunkedIIDMultisetShuffleCodec<S> = ChunkedHashingShuffleCodec<VecPrefixingChain<<S as Codec>::Symbol>, IIDVecSliceCodecs<S>, Complete, <S as Codec>::Symbol>;

#[allow(unused)]
pub fn chunked_iid_multiset_shuffle_codec<S: Codec<Symbol: OrdSymbol + Default> + Symbol>(codec: IID<S>, chunk_sizes: Vec<usize>) -> ChunkedIIDMultisetShuffleCodec<S> {
    let IID { item, len } = codec;
    chunked_hashing_shuffle_codec(
        VecPrefixingChain::new(),
        IIDVecSliceCodecs::new(len, item),
        Complete,
        chunk_sizes,
        len,
    )
}
