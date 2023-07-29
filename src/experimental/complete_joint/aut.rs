//! Perm group coset codec based on a stabilizer chains.
//! Can be used for joint shuffle coding given an objects' automorphism group.
//! Used for graphs.

use crate::autoregressive::multiset::Orbit;
#[cfg(test)]
use crate::codec::{assert_bits_eq, EqSymbol};
use crate::codec::{Codec, Message, Uniform, UniformCodec};
use crate::experimental::complete_joint::sparse_perm::SparsePerm;
use crate::joint::coset::{RCosetUniform, StabilizerChainLike};
use crate::joint::{Canonizable, JointShuffleCodec, RCosetUniformCodec};
#[cfg(test)]
use crate::permutable::PermutationUniform;
use crate::permutable::{FHashMap, Orbits, Perm, Permutable, Permutation};
use itertools::Itertools;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::ops::Deref;

/// Ordered object with a canonical ordering and known automorphism group.
pub trait AutCanonizable: Permutable {
    /// Returns the automorphism group of the object.
    fn automorphism_group(&self) -> PermutationGroup {
        self.automorphisms().group
    }

    /// Fused method for getting automorphism group, canonical permutation and orbits all at once,
    /// allowing optimized implementations.
    fn automorphisms(&self) -> Automorphisms;

    #[cfg(test)]
    /// Any implementation should pass this test for any seed.
    fn test(&self, seed: usize)
    where
        Self: EqSymbol,
    {
        let (y, y_canon) = Canonizable::test_canon(self, seed);
        let y_aut = y.automorphisms();
        assert_eq!(y_aut.canon, y_canon);
        assert_eq!(y_aut.decanon, y_canon.inverse());
        let x_aut = self.automorphisms();
        assert_eq!(y.orbit_ids().len(), self.orbit_ids().len());
        assert_eq!(y_aut.orbit_ids.len(), x_aut.orbit_ids.len());
        let y_aut_bits = AutomorphismUniform::new(&y.automorphism_group().lex_stabilizer_chain()).uni_bits();
        let group = self.automorphism_group();
        let x_aut_bits = AutomorphismUniform::new(&group.lex_stabilizer_chain()).uni_bits();
        assert_bits_eq(y_aut_bits, x_aut_bits);
        assert_bits_eq(y_aut.bits, x_aut.bits);
        assert_bits_eq(x_aut.bits, x_aut_bits);
        // TODO test automorphism_group
    }
}

impl<T: AutCanonizable> Canonizable for T {
    fn canon(&self) -> Perm { self.automorphisms().canon }

    fn is_isomorphic(&self, other: &Self) -> bool
    where
        Self: Eq,
    {
        if self.len() != other.len() {
            return false;
        }

        if ISOMORPHISM_TEST_MAX_LEN.with_borrow(|l| *l) < self.len() {
            return true;
        }

        self.canonized() == other.canonized()
    }
}

thread_local! {
    static ISOMORPHISM_TEST_MAX_LEN: RefCell<usize> = RefCell::new(usize::MAX);
}

/// Runs the given function with isomorphism tests for graphs above this length disabled,
/// with is_isomorphic returning true for any such graphs of the same length.
/// Useful when testing unordered graph codecs where equality checks can become too slow.
pub fn with_isomorphism_test_max_len<R>(len: usize, f: impl FnOnce() -> R) -> R {
    ISOMORPHISM_TEST_MAX_LEN.with(|l| {
        let old = l.replace(len);
        let out = f();
        assert_eq!(l.replace(old), len);
        out
    })
}

impl<T: AutCanonizable> Orbits for T {
    type Id = i32;

    fn orbit_ids(&self) -> impl Deref<Target=Vec<<Self as Orbits>::Id>> {
        Box::new(self.automorphisms().orbit_ids)
    }
}

pub fn joint_shuffle_codec<C: Codec<Symbol: AutCanonizable>>(ordered: C) -> JointShuffleCodec<C, AutRCosetUniform> {
    JointShuffleCodec { ordered, phantom: PhantomData }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrbitElementUniform {
    pub sorted_elements: Vec<usize>,
}

impl OrbitElementUniform {
    pub fn new(orbit: &Orbit) -> Self {
        Self { sorted_elements: orbit.iter().sorted_unstable().cloned().collect() }
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AutomorphismUniform<'a> {
    chain: &'a StabilizerChain,
}

impl<'a> Codec for AutomorphismUniform<'a> {
    type Symbol = Automorphism;

    fn push(&self, m: &mut Message, x: &Automorphism) {
        assert_eq!(x.min_elements.len(), self.chain.len());
        for ((orbit, _), min_element) in self.chain.iter().zip(&x.min_elements).rev() {
            OrbitElementUniform::new(orbit).push(m, &min_element);
        }
    }

    fn pop(&self, m: &mut Message) -> Automorphism {
        let (min_elements, element_to_min) = self.chain.iter().map(|(orbit, from)| {
            let min_element = OrbitElementUniform::new(orbit).pop(m);
            (min_element, from[&min_element].inverse())
        }).unzip();

        Automorphism { min_elements, base_to_min: element_to_min }
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { Some(self.uni_bits()) }
}

impl<'a> UniformCodec for AutomorphismUniform<'a> {
    fn uni_bits(&self) -> f64 {
        self.chain.iter().map(|(orbit, _)| (orbit.len() as f64).log2()).sum()
    }
}

impl<'a> AutomorphismUniform<'a> {
    pub fn new(chain: &'a StabilizerChain) -> Self { Self { chain } }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StabilizerChain(Vec<(Orbit, FHashMap<usize, SparsePerm>)>);

impl Deref for StabilizerChain {
    type Target = Vec<(Orbit, FHashMap<usize, SparsePerm>)>;
    fn deref(&self) -> &Self::Target { &self.0 }
}

pub type AutRCosetUniform = RCosetUniform<StabilizerChain>;

impl<P: AutCanonizable> RCosetUniformCodec<P> for AutRCosetUniform {
    fn from_canonized(canonized: &P) -> Self {
        Self { chain: canonized.automorphism_group().lex_stabilizer_chain() }
    }
}

impl AutRCosetUniform {
    pub fn new(group: PermutationGroup) -> Self {
        Self { chain: group.lex_stabilizer_chain() }
    }
}

/// Automorphism that when applied to the min coset representative containing a permutation l
/// will recover l.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Automorphism {
    pub min_elements: Vec<usize>,
    pub base_to_min: Vec<SparsePerm>,
}

impl Automorphism {
    /// (a[n] * a[n-1] * ... a[1] * a[0])
    pub fn total(self) -> SparsePerm {
        let mut total = SparsePerm::identity(self.base_to_min.len());
        for element_to_min in self.base_to_min.into_iter().rev() {
            total = element_to_min * total;
        }
        total
    }
}

impl StabilizerChainLike for StabilizerChain {
    type Perm = SparsePerm;
    type Automorphism = Automorphism;

    fn coset_min_and_restore_aut(&self, mut labels: SparsePerm) -> (SparsePerm, Automorphism) {
        let mut min_elements = vec![];
        let mut base_to_min = vec![];

        for (base, (orbit, from)) in self.iter().enumerate() {
            let (min_label, min_element) = orbit.iter().
                map(|&e| (&labels * e, e)).
                min_by(|(l, _), (l_, _)| l.cmp(l_)).
                unwrap();
            let base_to_min_automorphism = from[&min_element].inverse();
            min_elements.push(min_element);
            base_to_min.push(base_to_min_automorphism.clone());
            // Unapply automorphism to get min coset representative, moving min -> base:
            let new_labels = labels * base_to_min_automorphism;
            assert_eq!(&new_labels * base, min_label);
            labels = new_labels;
        }
        (labels, Automorphism { min_elements, base_to_min })
    }

    /// Restore permutation from lexicographical minimum of its coset.
    fn apply(coset_min: Self::Perm, restore_aut: Self::Automorphism) -> Self::Perm {
        coset_min * restore_aut.total().inverse()
    }

    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn automorphism_codec(&self) -> impl UniformCodec<Symbol=Self::Automorphism> {
        AutomorphismUniform::new(self)
    }
}

pub struct OrbitStabilizerInfo {
    pub orbit: Orbit,
    pub stabilizer: PermutationGroup,
    /// Permutation from any orbit element to the original element.
    pub from: FHashMap<usize, SparsePerm>,
}

impl PermutationGroup {
    pub fn adjust_len(&mut self, len: usize) {
        for g in self.generators.iter_mut() {
            g.set_len(len);
        }
        self.len = len;
    }

    pub fn new(len: usize, generators: Vec<SparsePerm>) -> Self {
        for g in &generators {
            assert_eq!(len, g.len);
        }
        Self { len, generators }
    }

    /// Return an element's orbit and stabilizer group.
    /// See K. H. Rosen: Computational Group Theory, p. 79.
    pub fn orbit_stabilizer(&self, element: usize) -> OrbitStabilizerInfo {
        let mut orbit = vec![element];
        let mut from = FHashMap::default();
        from.insert(element, SparsePerm::identity(self.len));
        let mut generators = BTreeSet::new();
        let mut i = 0;
        while let Some(&e) = orbit.get(i) {
            let from_e = from[&e].clone();
            for g in self.generators.iter() {
                let new = g * e;
                let from_new = from_e.clone() * g.inverse();
                assert_eq!(&from_new * new, element);
                if let Some(from_new_) = from.get(&new) {
                    let stab_g = from_new * from_new_.inverse();
                    if !stab_g.is_identity() {
                        assert_eq!(&stab_g * element, element);
                        generators.insert(stab_g);
                    }
                } else {
                    from.insert(new, from_new);
                    orbit.push(new);
                }
            }

            i += 1;
        }

        let stabilizer = PermutationGroup::new(self.len, generators.into_iter().collect());
        OrbitStabilizerInfo { orbit: Orbit::from_iter(orbit), stabilizer, from }
    }

    /// Stabilizer chain for base 0..self.len.
    /// OPTIMIZE: Use Schreier-Sims algorithm to compute stabilizer chain more quickly.
    pub fn lex_stabilizer_chain(&self) -> StabilizerChain {
        // OPTIMIZE: avoid this clone
        let mut group = self.clone();
        let mut chain = vec![];
        for element in 0..self.len {
            let OrbitStabilizerInfo { orbit, stabilizer, from } = group.orbit_stabilizer(element);
            chain.push((orbit, from));
            group = stabilizer;
        }
        StabilizerChain(chain)
    }

    /// Conjugated group with the set of elements relabeled as given by the permutation.
    pub fn permuted(self, p: &SparsePerm) -> PermutationGroup {
        let generators = self.generators.into_iter().map(|g| p.clone() * g * p.inverse().clone()).collect();
        PermutationGroup::new(self.len, generators)
    }
}

#[derive(Clone, Debug)]
pub struct Automorphisms {
    pub group: PermutationGroup,
    pub canon: Perm,
    pub decanon: Perm,
    pub orbit_ids: Vec<i32>,
    pub bits: f64,
}

impl PartialEq for Automorphisms {
    fn eq(&self, other: &Self) -> bool {
        self.group == other.group && self.canon == other.canon
    }
}

impl Eq for Automorphisms {}

impl Permutable for Automorphisms {
    fn len(&self) -> usize {
        self.group.len
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.group.swap(i, j);
        self.canon.swap(i, j);
        self.decanon.swap(self.canon.apply(i), self.canon.apply(j));
        self.orbit_ids.swap(i, j);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PermutationGroup {
    pub len: usize,
    pub generators: Vec<SparsePerm>,
}

impl Permutable for PermutationGroup {
    fn len(&self) -> usize { self.len }

    fn swap(&mut self, i: usize, j: usize) {
        for g in self.generators.iter_mut() {
            g.swap(i, j);
        }
    }
}

/// Demo implementation for a generic canonization on product classes. Only for reference, unused. 
impl<A: AutCanonizable, B: AutCanonizable> Canonizable for (A, B) {
    fn canon(&self) -> Perm {
        let canon0 = self.0.canon();
        // Permute the fused object so that the new 0 is canonical:
        let new: (A, B) = &canon0 * self;
        // Permute the fused object to minimize 1's canonical labels while keeping new 0 canonical:
        let chain = new.0.automorphism_group().lex_stabilizer_chain();
        let (_, restore1_aut) = chain.coset_min_and_restore_aut(new.1.canon().iter().collect_vec().into());
        let min1_aut = restore1_aut.total().inverse();
        // The canonical fused object is &min1_aut * &new.
        // Return chain of perms that we applied to get it:
        (min1_aut * canon0.iter().collect_vec().into()).iter().collect_vec().into()
    }
}

#[cfg(test)]
mod tests {
    use crate::experimental::complete_joint::aut::AutCanonizable;
    use crate::experimental::graph::tests::small_digraphs;
    use crate::graph::Graph;
    use crate::joint::coset::StabilizerChainLike;
    use crate::permutable::Permutable;
    use itertools::Itertools;

    use super::*;

    #[test]
    fn automorphism_codec() {
        for graph in small_digraphs() {
            test_automorphism_codec(graph);
        }
    }

    fn test_automorphism_codec(graph: Graph) {
        let group = graph.automorphism_group();
        let chain = group.lex_stabilizer_chain();
        let labels: SparsePerm = PermutationUniform::new(graph.len()).sample(0);
        let (canonized_labels, restore_aut) = chain.coset_min_and_restore_aut(labels.clone());
        let labels_ = StabilizerChain::apply(canonized_labels.clone(), restore_aut.clone());
        assert_eq!(labels_, labels.clone());

        let mut isomorphic_labels = labels.clone();
        for g in group.generators.clone() {
            isomorphic_labels = isomorphic_labels * g;
            let (canonized_labels_, _) = chain.coset_min_and_restore_aut(isomorphic_labels.clone());

            assert_eq!(&canonized_labels, &canonized_labels_);
        }

        let aut_codec = AutomorphismUniform::new(&chain);
        aut_codec.test(&restore_aut, 0);
        aut_codec.test_on_samples(1000);
    }

    #[test]
    fn rcoset_codec() {
        for x in small_digraphs() {
            AutRCosetUniform::new(x.automorphism_group()).test_on_samples(100);
        }
    }

    #[test]
    fn orbit_stabilizer_trivial() {
        let trivial = PermutationGroup::new(3, vec![]);
        let r = trivial.orbit_stabilizer(2);
        assert_eq!(r.orbit.into_iter().collect_vec(), vec![2]);
        assert_eq!(r.stabilizer.generators, vec![]);
    }

    #[test]
    fn orbit_stabilizer() {
        let generators = vec![SparsePerm::from(vec![0, 1, 3, 2])];
        let g = PermutationGroup::new(4, generators);
        let r = g.orbit_stabilizer(2);
        assert_eq!(r.orbit, Orbit::from_iter([2, 3]));
        assert_eq!(r.stabilizer.generators, vec![]);
    }

    #[test]
    fn orbit_stabilizer2() {
        let generators = vec![SparsePerm::from(vec![0, 1, 3, 2]), SparsePerm::from(vec![1, 0, 2, 3])];
        let g = PermutationGroup::new(4, generators);
        let r = g.orbit_stabilizer(2);
        assert_eq!(r.orbit, Orbit::from_iter([2, 3]));
        assert_eq!(r.stabilizer.generators, vec![SparsePerm::from(vec![1, 0, 2, 3])]);
    }

    #[test]
    fn permuted_group() {
        let g = &PermutationGroup::new(4, vec![SparsePerm::from(vec![0, 1, 3, 2])]);
        let p = &SparsePerm::from(vec![3, 0, 1, 2]);
        let g_p = g.clone().permuted(p);

        for e in 0..3 {
            let orbit = g.orbit_stabilizer(e).orbit.into_iter().sorted_unstable().map(|x| p * x).collect_vec();
            let orbit_ = g_p.orbit_stabilizer(p * e).orbit.into_iter().sorted_unstable().collect_vec();
            assert_eq!(orbit, orbit_)
        }
    }
}
