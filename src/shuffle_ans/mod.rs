use std::fmt::{Debug, Formatter};

use clap::Parser;
use float_extras::f64::lgamma;

use crate::ans::{Codec, Message, Uniform, UniformCodec};
use crate::permutable::{Automorphism, GroupPermutable, Orbit, Permutable, Permutation, PermutationGroup, StabilizerChain, Unordered};

pub mod coset_interleaved;
pub mod interleaved;

#[derive(Clone, Debug)]
pub struct OrbitElementUniform {
    pub sorted_elements: Vec<usize>,
}

impl OrbitElementUniform {
    pub fn new(orbit: Orbit) -> Self {
        Self { sorted_elements: orbit.into_iter().collect() }
    }

    fn len(&self) -> usize { self.sorted_elements.len() }

    pub fn min(&self) -> usize { self.sorted_elements[0] }

    fn inner_codec(&self) -> Uniform {
        let size = self.len();
        Uniform::new(size)
    }
}

impl Codec for OrbitElementUniform {
    type Symbol = usize;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let index = self.sorted_elements.iter().position(|e| *e == *x).unwrap();
        self.inner_codec().push(m, &index);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let index = self.inner_codec().pop(m);
        self.sorted_elements[index]
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { Some(self.uni_bits()) }
}

impl UniformCodec for OrbitElementUniform {
    fn uni_bits(&self) -> f64 { self.inner_codec().uni_bits() }
}

/// Pop is the modern version of the Fisher-Yates shuffle, and push its inverse. Both are O(len).
#[derive(Clone)]
pub struct PermutationUniform {
    pub len: usize,
}

impl Codec for PermutationUniform {
    type Symbol = Permutation;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let mut p = Permutation::identity(self.len);
        let mut p_inv = Permutation::identity(self.len);
        let mut js = vec![];
        for i in (0..self.len).rev() {
            let xi = x * i;
            let j = &p_inv * xi;
            p_inv.swap(&p * i, xi);
            p.swap(i, j);
            js.push(j);
        }

        for (i, j) in js.iter().rev().enumerate() {
            let size = i + 1;
            Uniform::new(size).push(m, j);
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let mut p = Permutation::identity(self.len);

        for i in (0..self.len).rev() {
            let size = i + 1;
            let j = Uniform::new(size).pop(m);
            p.swap(i, j);
        }

        p
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { Some(self.uni_bits()) }
}

impl UniformCodec for PermutationUniform {
    fn uni_bits(&self) -> f64 {
        lgamma((self.len + 1) as f64) / 2f64.ln()
    }
}

#[derive(Clone)]
pub struct AutomorphismUniform {
    chain: StabilizerChain,
}

impl Codec for AutomorphismUniform {
    type Symbol = Automorphism;

    fn push(&self, m: &mut Message, x: &Automorphism) {
        assert_eq!(x.min_elements.len(), self.chain.len());
        for ((orbit, _), min_element) in self.chain.iter().zip(&x.min_elements).rev() {
            OrbitElementUniform::new(orbit.clone()).push(m, &min_element);
        }
    }

    fn pop(&self, m: &mut Message) -> Automorphism {
        let (min_elements, element_to_min) = self.chain.iter().map(|(orbit, from)| {
            let min_element = OrbitElementUniform::new(orbit.clone()).pop(m);
            (min_element, from[&min_element].inverse())
        }).unzip();

        Automorphism { min_elements, base_to_min: element_to_min }
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { Some(self.uni_bits()) }
}

impl UniformCodec for AutomorphismUniform {
    fn uni_bits(&self) -> f64 {
        self.chain.iter().map(|(orbit, _)| (orbit.len() as f64).log2()).sum()
    }
}

impl AutomorphismUniform {
    pub fn new(chain: StabilizerChain) -> Self { Self { chain } }

    #[allow(unused)]
    pub fn from_group(group: PermutationGroup) -> Self {
        Self::new(group.lex_stabilizer_chain())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RCosetUniform {
    chain: StabilizerChain,
}

impl Codec for RCosetUniform {
    type Symbol = Permutation;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let (coset_min, _) = self.chain.coset_min_and_restore_aut(x);
        let restore_aut = self.automorphism_codec().pop(m);
        let perm = restore_aut.apply_to(coset_min);
        self.permutation_codec().push(m, &perm);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let perm = self.permutation_codec().pop(m);
        let (coset_min, restore_aut) = self.chain.coset_min_and_restore_aut(&perm);
        self.automorphism_codec().push(m, &restore_aut);
        coset_min
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { Some(self.uni_bits()) }
}

impl UniformCodec for RCosetUniform {
    fn uni_bits(&self) -> f64 {
        self.permutation_codec().uni_bits() - self.automorphism_codec().uni_bits()
    }
}

impl RCosetUniform {
    pub fn new(group: PermutationGroup) -> Self {
        Self { chain: group.lex_stabilizer_chain() }
    }

    fn automorphism_codec(&self) -> AutomorphismUniform {
        AutomorphismUniform::new(self.chain.clone())
    }

    fn permutation_codec(&self) -> PermutationUniform {
        PermutationUniform { len: self.chain.len() }
    }
}

#[derive(Clone)]
pub struct ShuffleCodec<
    C: Codec,
    RCosetCFromP: Fn(&C::Symbol) -> RCosetC + Clone,
    RCosetC: Codec<Symbol=Permutation> = RCosetUniform>
    where C::Symbol: GroupPermutable {
    pub ordered: C,
    pub rcoset_for: RCosetCFromP,
}

impl<
    C: Codec,
    RCosetCFromP: Fn(&C::Symbol) -> RCosetC + Clone,
    RCosetC: UniformCodec<Symbol=Permutation>>
Codec for ShuffleCodec<C, RCosetCFromP, RCosetC> where C::Symbol: GroupPermutable {
    type Symbol = Unordered<C::Symbol>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let x = x.canonized();
        let coset_min = (self.rcoset_for)(&x).pop(m);
        let x = &coset_min * &x;
        self.ordered.push(m, &x)
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let (x, canon) = self.ordered.pop(m).canonized();
        let coset_min = canon.inverse();
        (self.rcoset_for)(&x).push(m, &coset_min);
        Unordered(x)
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.ordered.bits(x.to_ordered()).map(|bits| bits - (self.rcoset_for)(x.to_ordered()).uni_bits())
    }
}

/// Only for use of codec as symbol for parametrized models:
impl<
    C: Codec,
    RCosetCFromP: Fn(&C::Symbol) -> RCosetC + Clone,
    RCosetC: UniformCodec<Symbol=Permutation>>
Debug for ShuffleCodec<C, RCosetCFromP, RCosetC> where C::Symbol: GroupPermutable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnorderedCodec").finish()
    }
}

/// Only for use of codec as symbol for parametrized models:
impl<
    C: Codec,
    RCosetCFromP: Fn(&C::Symbol) -> RCosetC + Clone,
    RCosetC: UniformCodec<Symbol=Permutation>>
PartialEq<Self> for ShuffleCodec<C, RCosetCFromP, RCosetC> where C::Symbol: GroupPermutable {
    fn eq(&self, _: &Self) -> bool { true }
}

impl<
    C: Codec,
    RCosetCFromP: Fn(&C::Symbol) -> RCosetC + Clone,
    RCosetC: UniformCodec<Symbol=Permutation>>
Eq for ShuffleCodec<C, RCosetCFromP, RCosetC> where C::Symbol: GroupPermutable {}

pub fn shuffle_codec<C: Codec>(codec: C) -> ShuffleCodec<C, fn(&C::Symbol) -> RCosetUniform> where C::Symbol: GroupPermutable {
    ShuffleCodec { ordered: codec, rcoset_for: |x| RCosetUniform::new(x.automorphism_group()) }
}

#[derive(Clone, Debug, Parser)]
pub struct TestConfig {
    /// Test shuffle codec.
    #[clap(short = 's', long = "shuffle")]
    pub blockwise: bool,
    /// Test coset-interleaved shuffle codec.
    #[clap(hide = true, short, long)]
    pub coset_interleaved: bool,
    /// Test optimal-rate interleaved shuffle codec on uniform model.
    #[clap(skip)]
    pub interleaved: bool,

    #[clap(short, long, default_value = "3")]
    pub wl_iter: usize,

    #[clap(long)]
    pub wl_extra_half_iter: bool,

    /// Use a random message tail generator with the given seed, providing random initial bits. By default, no random generator is used, and initial bits will be 0.
    #[clap(long)]
    pub seed: Option<usize>,
    /// Verify permute group action and canon labelling axioms with the given seed. By default, no verification is performed.
    #[clap(hide = true, short, long)]
    pub axioms_seed: Option<usize>,
}

impl TestConfig {
    #[cfg(test)]
    pub fn test(seed: usize) -> Self {
        Self { axioms_seed: Some(seed), blockwise: true, coset_interleaved: true, interleaved: true, wl_iter: 3, wl_extra_half_iter: false, seed: Some(seed) }
    }

    pub fn initial_message(&self) -> Message {
        if let Some(seed) = self.seed { Message::random(seed) } else { Message::zeros() }
    }
}

#[cfg(test)]
pub mod test {
    use crate::ans::{Codec, CodecTestResults, VecCodec};
    use crate::benchmark::test_and_print;
    use crate::graph::Graph;
    use crate::graph_ans::tests::small_digraphs;
    use crate::permutable::{GroupPermutable, Permutable, Permutation};
    use crate::shuffle_ans::coset_interleaved::coset_interleaved_shuffle_codec;

    use super::*;

    pub fn test_and_print_vec<C: Codec>(codecs: impl IntoIterator<Item=C>, symbols: &Vec<C::Symbol>, initial: &Message) -> CodecTestResults {
        test_and_print(&VecCodec::new(codecs), symbols, initial)
    }

    pub fn test_shuffle_codecs<C: Codec>(
        codecs: &Vec<C>,
        unordered: &Vec<Unordered<C::Symbol>>,
        config: &TestConfig) where C::Symbol: GroupPermutable {
        if let Some(seed) = config.axioms_seed {
            for x in unordered {
                x.to_ordered().test(seed);
            }
        }
        if config.blockwise {
            test_and_print_vec(codecs.iter().cloned().map(shuffle_codec), unordered, &config.initial_message());
        }
        if config.coset_interleaved {
            test_and_print_vec(codecs.iter().cloned().map(coset_interleaved_shuffle_codec), unordered, &config.initial_message());
        }
    }

    #[test]
    fn permutation_codec() {
        PermutationUniform { len: 5 }.test(&Permutation::from(vec![0, 2, 1, 3, 4]), &Message::random(0));
        PermutationUniform { len: 9 }.test_on_samples(100);
    }

    #[test]
    fn automorphism_codec() {
        for graph in small_digraphs() {
            test_automorphism_codec(graph);
        }
    }

    fn test_automorphism_codec(graph: Graph) {
        let group = graph.automorphism_group();
        let chain = group.lex_stabilizer_chain();
        let labels = PermutationUniform { len: graph.len() }.sample(0);
        let (canonized_labels, restore_aut) = chain.coset_min_and_restore_aut(&labels);
        let labels_ = restore_aut.clone().apply_to(canonized_labels.clone());
        assert_eq!(labels_, labels.clone());

        let mut isomorphic_labels = labels.clone();
        for g in group.generators.clone() {
            isomorphic_labels = isomorphic_labels * g;
            let (canonized_labels_, _) = chain.coset_min_and_restore_aut(&isomorphic_labels);

            assert_eq!(&canonized_labels, &canonized_labels_);
        }

        let aut_codec = AutomorphismUniform::new(chain);
        aut_codec.test(&restore_aut, &Message::random(0));
        aut_codec.test_on_samples(1000);
    }

    #[test]
    fn rcoset_codec() {
        for x in small_digraphs() {
            RCosetUniform::new(x.automorphism_group()).test_on_samples(100);
        }
    }
}