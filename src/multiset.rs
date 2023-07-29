use std::collections::BTreeMap;
use std::hash::Hash;

use itertools::Itertools;

use crate::ans::Symbol;
use crate::ans::UniformCodec;
use crate::permutable::{Automorphisms, GroupPermutable, Orbit, Orbits, Partial, Permutable, Permutation, PermutationGroup, Unordered};
use crate::shuffle_ans::PermutationUniform;

pub type Multiset<T> = Unordered<Vec<T>>;

pub trait OrdSymbol: Symbol + Ord + Hash {}

impl<T: Symbol + Ord + Hash> OrdSymbol for T {}

impl<T: Symbol> Permutable for Vec<T> {
    fn len(&self) -> usize { self.len() }

    fn swap(&mut self, i: usize, j: usize) {
        (self as &mut [T]).swap(i, j);
    }
}

impl<T: Symbol + Ord + Hash> GroupPermutable for Vec<T> {
    fn automorphism_group(&self) -> PermutationGroup {
        self.automorphisms().group
    }

    fn canon(&self) -> Permutation {
        Permutation::from(self.iter().enumerate().
            sorted_by(|(_, x), (_, y)| x.cmp(y)).
            map(|(i, _)| i).collect()).inverse()
    }

    fn orbits(&self) -> Orbits {
        let mut out = BTreeMap::new();
        for (i, o) in self.iter().enumerate() {
            out.entry(o).or_insert(Orbit::new()).insert(i);
        }
        out.into_values().collect()
    }

    fn automorphisms(&self) -> Automorphisms {
        let orbits = self.orbits();
        let canon = self.canon();
        let decanon = canon.inverse();
        let generators = orbits.iter().flat_map(
            |o| o.iter().map(|&e| Permutation::create_swap(self.len(), *o.first().unwrap(), e))).collect();
        let group = PermutationGroup::new(self.len(), generators);
        let bits = orbits.iter().map(|o| {
            let len = o.len();
            PermutationUniform { len }.uni_bits()
        }).sum();
        Automorphisms { group, canon, decanon, orbits, bits }
    }
}

impl<T: OrdSymbol> Partial for Vec<T> {
    type Complete = Vec<T>;
    type Slice = T;

    fn pop_slice(&mut self) -> Self::Slice { self.pop().unwrap() }
    fn push_slice(&mut self, slice: &Self::Slice) { self.push(slice.clone()) }
    fn empty(_size: usize) -> Self { vec![] }
    fn from_complete(complete: Vec<T>) -> Self { complete }
    fn into_complete(self) -> Vec<T> { self }
}

#[cfg(test)]
mod tests {
    use crate::ans::{Codec, IID, Message, Uniform, VecCodec};
    use crate::graph_ans::tests::with_sampled_symbols;
    use crate::permutable::{GroupPermutable, Orbit, Orbits, Permutation};
    use crate::shuffle_ans::{RCosetUniform, TestConfig};
    use crate::shuffle_ans::interleaved::{InterleavedShuffleCodec, OrbitPartitioner};
    use crate::shuffle_ans::test::{test_and_print_vec, test_shuffle_codecs};

    use super::*;

    pub fn test_multiset_codecs(codecs_and_vecs: impl IntoIterator<Item=(IID<Uniform>, Vec<usize>)>, config: &TestConfig) {
        let (codecs, permutables): (Vec<_>, Vec<_>) = codecs_and_vecs.into_iter().unzip();
        let unordered = &permutables.iter().map(|x| Unordered(x.clone())).collect();
        test_shuffle_codecs(&codecs, unordered, config);
        if config.interleaved {
            test_and_print_vec(codecs.iter().map(interleaved), unordered, &config.initial_message());
        }
        println!();
    }

    fn interleaved(c: &IID<Uniform>) -> impl Codec<Symbol=Multiset<usize>> + '_ {
        InterleavedShuffleCodec::<Vec<_>, _, _, _>::new(c.len, move |_| c.item.clone(), OrbitPartitioner)
    }

    #[test]
    fn multiset() {
        let v = vec![1, 1];
        assert_eq!(v.len(), 2);
        assert_eq!(&Permutation::create_swap(2, 1, 0) * &v, v);
        let chain = v.automorphism_group().lex_stabilizer_chain();
        assert_eq!(chain[0].0.len(), 2);
        assert_eq!(chain[1].0.len(), 1);
        assert_eq!(v.orbits(), Orbits::from([Orbit::from([0, 1])]));
        assert!(v.canon().is_identity());
        assert!(RCosetUniform::new(v.automorphism_group()).uni_bits().abs() < 1e-6);
    }

    #[test]
    fn min_codecs_with_vecs() {
        let vecs = [vec![0, 1]];
        test_multiset_codecs(vecs.map(|v|
            (IID::new(Uniform::new(*v.iter().max().unwrap() + 1), v.len()), v)), &TestConfig::test(0));
    }

    #[test]
    fn codecs_with_vecs() {
        let vecs = [vec![1, 1], vec![1, 0, 4, 2, 2, 1, 1]];
        test_multiset_codecs(vecs.map(|v|
            {
                let size = *v.iter().max().unwrap() + 1;
                (IID::new(Uniform::new(size), v.len()), v)
            }), &TestConfig::test(0));
    }

    #[test]
    fn sampled_codecs_small() {
        let codecs = (0..20).map(|i| {
            let size = i + 1;
            IID::new(Uniform::new(size), i * 2)
        });
        test_multiset_codecs(with_sampled_symbols(codecs), &TestConfig::test(0));
    }

    #[test]
    fn interleaved_sampled_codecs() {
        let (codecs, permutables): (Vec<_>, Vec<_>) = with_sampled_symbols((0..100).map(|i| {
            let size = i + 1;
            IID::new(Uniform::new(size), i * 2)
        })).into_iter().unzip();
        let fs_codec = VecCodec::new(codecs.iter().map(interleaved));
        let unordered = permutables.into_iter().map(|v| Unordered(v)).collect();
        let result = fs_codec.test(&unordered, &Message::random(0));
        assert!(result.enc_sec < 5., "enc_sec = {}", result.enc_sec);
        assert!(result.dec_sec < 5., "dec_sec = {}", result.dec_sec);
    }

    #[test]
    #[ignore = "Slow due to use of Vec group automorphisms."]
    fn slow_permutable_axioms() {
        let codecs = (0..20).map(|i| {
            let size = i + 1;
            IID::new(Uniform::new(size), i * 2)
        });
        for x in VecCodec::new(codecs).pop(&mut Message::zeros()) {
            x.test(0)
        };
    }
}
