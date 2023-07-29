use crate::ans::{Codec, Message, MutDistribution, MutCategorical, UniformCodec};
use crate::permutable::{GroupPermutable, OrbitStabilizerInfo, Permutation, PermutationGroup};
use crate::shuffle_ans::{OrbitElementUniform, RCosetUniform, ShuffleCodec};

#[derive(Clone, Debug)]
struct AutomorphicOrbitStabilizer {
    orbit: OrbitElementUniform,
    stabilizer: PermutationGroup,
    element_to_min: Permutation,
}

fn automorphic_orbit_stabilizer(group: PermutationGroup, element: usize) -> AutomorphicOrbitStabilizer {
    let OrbitStabilizerInfo { orbit, stabilizer, from } =
        group.orbit_stabilizer(element);
    let orbit = OrbitElementUniform::new(orbit);
    let min = orbit.min();
    let element_to_min = from[&min].inverse();
    assert_eq!(&element_to_min * element, min);
    let stabilizer = stabilizer.permuted(&element_to_min);
    AutomorphicOrbitStabilizer { orbit, stabilizer, element_to_min }
}

#[derive(Clone)]
pub struct InterleavedRCosetUniform {
    pub group: PermutationGroup,
    pub stab_codec: MutCategorical,
}

impl Codec for InterleavedRCosetUniform {
    type Symbol = Permutation;

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

        let indices = Permutation::from(elements).inverse();
        indices
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { Some(self.uni_bits()) }
}

impl UniformCodec for InterleavedRCosetUniform {
    fn uni_bits(&self) -> f64 {
        RCosetUniform::new(self.group.clone()).uni_bits()
    }
}

impl InterleavedRCosetUniform {
    pub fn new(group: PermutationGroup) -> Self {
        let stab_codec = MutCategorical::new((0..group.len).map(|i| (i, 1)), true);
        Self { group, stab_codec }
    }
}

pub fn coset_interleaved_shuffle_codec<C: Codec>(
    codec: C
) -> ShuffleCodec<C, fn(&C::Symbol) -> InterleavedRCosetUniform, InterleavedRCosetUniform> where C::Symbol: GroupPermutable {
    ShuffleCodec { ordered: codec, rcoset_for: |x| InterleavedRCosetUniform::new(x.automorphism_group()) }
}


#[cfg(test)]
pub mod tests {
    use crate::ans::Codec;
    use crate::graph_ans::tests::small_digraphs;
    use crate::permutable::GroupPermutable;
    use crate::shuffle_ans::coset_interleaved::InterleavedRCosetUniform;

    #[test]
    fn interleaved_rcoset_codec() {
        for x in small_digraphs() {
            InterleavedRCosetUniform::new(x.automorphism_group()).test_on_samples(100);
        }
    }
}