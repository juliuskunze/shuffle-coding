//! Interleaved perm group coset codec based on a stabilizer chains.
//! Can be used for joint shuffle coding given a function to retrieve the automorphism group.
//! Reduces initial bits, but typically only marginally compared to autoregressive shuffle coding.
//! Mostly here for reference.

use crate::codec::{Codec, Message, MutDistribution, UniformCodec};
use crate::experimental::complete_joint::aut::{AutCanonizable, AutRCosetUniform, OrbitElementUniform, OrbitStabilizerInfo, PermutationGroup};
use crate::experimental::complete_joint::sparse_perm::SparsePerm;
use crate::joint::{JointShuffleCodec, RCosetUniformCodec};
use crate::permutable::Permutation;
use ftree::FenwickTree;
use std::iter::repeat_n;
use std::marker::PhantomData;

#[derive(Clone, Debug, Eq, PartialEq)]
struct AutomorphicOrbitStabilizer {
    orbit: OrbitElementUniform,
    stabilizer: PermutationGroup,
    element_to_min: SparsePerm,
}

fn automorphic_orbit_stabilizer(group: PermutationGroup, element: usize) -> AutomorphicOrbitStabilizer {
    let OrbitStabilizerInfo { orbit, stabilizer, from } =
        group.orbit_stabilizer(element);
    let orbit = OrbitElementUniform::new(&orbit);
    let min = orbit.min();
    let element_to_min = from[&min].inverse();
    assert_eq!(&element_to_min * element, min);
    let stabilizer = stabilizer.permuted(&element_to_min);
    AutomorphicOrbitStabilizer { orbit, stabilizer, element_to_min }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InterleavedAutRCosetUniform {
    pub group: PermutationGroup,
    pub stab_codec: FenwickTree<usize>,
}

impl Codec for InterleavedAutRCosetUniform {
    type Symbol = SparsePerm;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let mut indices = x.clone();
        let mut stab_codec = self.stab_codec.clone();
        let mut group = self.group.clone();
        let mut codecs = vec![];
        for i in 0..self.group.len {
            let element = &indices.inverse() * i;
            let AutomorphicOrbitStabilizer { orbit, stabilizer, element_to_min } = automorphic_orbit_stabilizer(group, element);
            let min = orbit.min();
            codecs.push((orbit, stab_codec.clone()));
            stab_codec.remove(&min, 1);
            group = stabilizer;
            indices = indices * element_to_min.inverse();
            assert_eq!(&indices * min, i);
        }

        for (orbit, stab) in codecs.iter().rev() {
            let element = orbit.pop(m);
            stab.push(m, &element);
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let mut group = self.group.clone();
        let mut stab_codec = self.stab_codec.clone();
        let mut elements = vec![];
        for _ in 0..self.group.len {
            let element = stab_codec.pop(m);
            let AutomorphicOrbitStabilizer { orbit, stabilizer, .. } =
                automorphic_orbit_stabilizer(group, element);
            orbit.push(m, &element);
            let min = orbit.min();
            assert!(!elements.contains(&min));
            elements.push(min);
            group = stabilizer;
            stab_codec.remove(&min, 1);
        }

        let indices = SparsePerm::from(elements).inverse();
        indices
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { Some(self.uni_bits()) }
}

impl UniformCodec for InterleavedAutRCosetUniform {
    fn uni_bits(&self) -> f64 {
        AutRCosetUniform::new(self.group.clone()).uni_bits()
    }
}

impl<P: AutCanonizable> RCosetUniformCodec<P> for InterleavedAutRCosetUniform {
    fn from_canonized(canonized: &P) -> Self {
        Self::new(canonized.automorphism_group())
    }
}

impl InterleavedAutRCosetUniform {
    pub fn new(group: PermutationGroup) -> Self {
        let stab_codec = FenwickTree::from_iter(repeat_n(1, group.len));
        Self { group, stab_codec }
    }
}

pub fn interleaved_coset_shuffle_codec<C: Codec<Symbol: AutCanonizable>>(codec: C) -> JointShuffleCodec<C, InterleavedAutRCosetUniform> {
    JointShuffleCodec { ordered: codec, phantom: PhantomData }
}


#[cfg(test)]
pub mod tests {
    use crate::codec::Codec;
    use crate::experimental::complete_joint::aut::AutCanonizable;
    use crate::experimental::graph::tests::small_digraphs;

    use super::*;

    #[test]
    fn test() {
        for x in small_digraphs() {
            InterleavedAutRCosetUniform::new(x.automorphism_group()).test_on_samples(100);
        }
    }
}
