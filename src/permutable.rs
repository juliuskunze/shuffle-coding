use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Debug;
use std::ops::{Deref, Mul};
use std::vec;

use itertools::Itertools;

use crate::ans::{assert_bits_eq, Codec, Symbol, UniformCodec};
use crate::shuffle_ans::{AutomorphismUniform, PermutationUniform, RCosetUniform};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Permutation {
    pub len: usize,
    pub indices: BTreeMap<usize, usize>,
}

impl Permutation {
    pub fn identity(len: usize) -> Self { Self { len, indices: BTreeMap::new() } }

    pub fn inverse(&self) -> Self {
        Self { len: self.len, indices: self.indices.iter().map(|(&x, &y)| (y, x)).collect() }
    }

    pub fn from(items: Vec<usize>) -> Self {
        let len = items.len();
        Self { len, indices: items.into_iter().enumerate().filter(|(i, x)| {
            assert!(*x < len);
            i != x
        }).map(|(i, x)| (i, x)).collect() }
    }

    pub fn to_iter(&self, len: usize) -> impl Iterator<Item=usize> + '_ {
        (0..len).map(move |i| self * i).into_iter()
    }

    pub fn is_identity(&self) -> bool { self.indices.is_empty() }

    fn normalize(&mut self) {
        self.indices.retain(|i, x| i != x);
    }

    pub fn set_len(&mut self, len: usize) {
        assert!(len >= self.len || self.indices.iter().all(|(&x, &y)| x < len && y < len));
        self.len = len;
    }

    pub fn create_swap(len: usize, i: usize, j: usize) -> Self {
        Self { len, indices: if i == j { BTreeMap::new() } else { BTreeMap::from_iter(vec![(i, j), (j, i)]) } }
    }
}

impl Permutable for Permutation {
    fn len(&self) -> usize {
        self.len
    }

    fn swap(&mut self, i: usize, j: usize) {
        if i == j { return; }
        let p = self as &Self;
        let pi = p * i;
        let pj = p * j;
        if j == pi { self.indices.remove(&j); } else { self.indices.insert(j, pi); }
        if i == pj { self.indices.remove(&i); } else { self.indices.insert(i, pj); }
    }
}

impl Mul<Permutation> for Permutation {
    type Output = Self;

    fn mul(mut self, mut rhs: Self) -> Self {
        assert_eq!(self.len, rhs.len);

        for x in rhs.indices.values_mut() {
            *x = self.indices.remove(x).unwrap_or(*x);
        }

        for (&i, &x) in self.indices.iter() {
            rhs.indices.entry(i).or_insert(x);
        }

        rhs.normalize();
        rhs
    }
}

impl Mul<usize> for &Permutation {
    type Output = usize;

    fn mul(self, rhs: usize) -> usize {
        *self.indices.get(&rhs).unwrap_or(&rhs)
    }
}

pub type Partition = BTreeSet<usize>;
pub type Cell = Partition;
pub type Orbit = Cell;
pub type Orbits = Vec<Orbit>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StabilizerChain(Vec<(Orbit, HashMap<usize, Permutation>)>);

impl Deref for StabilizerChain {
    type Target = Vec<(Orbit, HashMap<usize, Permutation>)>;
    fn deref(&self) -> &Self::Target { &self.0 }
}

/// Automorphism that when applied to the min coset representative containing a permutation l
/// will recover l.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Automorphism {
    pub min_elements: Vec<usize>,
    pub base_to_min: Vec<Permutation>,
}

impl Automorphism {
    /// (a[n] * a[n-1] * ... a[1] * a[0])
    pub fn total(self) -> Permutation {
        let mut total = Permutation::identity(self.min_elements.len());
        for element_to_min in self.base_to_min.into_iter().rev() {
            total = element_to_min * total;
        }
        total
    }

    /// Restore permutation from lexicographical minimum of its coset.
    pub fn apply_to(self, coset_min: Permutation) -> Permutation {
        coset_min * self.total().inverse()
    }
}

impl StabilizerChain {
    /// Returns a pair of:
    /// - The lexicographical minimum of the coset containing the given labels.
    /// Can be used as a canonical representative of that coset.
    /// - The automorphism to be applied to this representative to restore the original labels.
    ///
    /// In short: (h, a) where h = min{labels * AUT}, a = h * labels^-1
    pub fn coset_min_and_restore_aut(&self, labels: &Permutation) -> (Permutation, Automorphism) {
        let mut min_elements = vec![];
        let mut base_to_min = vec![];
        let mut labels = labels.clone();

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
}

pub struct OrbitStabilizerInfo {
    pub orbit: Orbit,
    pub stabilizer: PermutationGroup,
    /// Permutation from any orbit element to the original element.
    pub from: HashMap<usize, Permutation>,
}

#[derive(Clone, Debug)]
pub struct Automorphisms {
    pub group: PermutationGroup,
    pub canon: Permutation,
    pub decanon: Permutation,
    pub orbits: Orbits,
    pub bits: f64,
}

impl PartialEq for Automorphisms {
    fn eq(&self, other: &Self) -> bool {
        self.group == other.group && self.canon == other.canon
    }
}

impl Eq for Automorphisms {}

impl Automorphisms {
    pub fn orbit(&self, element: usize) -> Orbit {
        self.orbits.iter().find(|o| o.contains(&element)).unwrap().clone()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PermutationGroup {
    pub len: usize,
    pub generators: Vec<Permutation>,
}

impl PermutationGroup {
    pub fn adjust_len(&mut self, len: usize) {
        for g in self.generators.iter_mut() {
            g.set_len(len);
        }
        self.len = len;
    }

    pub fn new(len: usize, generators: Vec<Permutation>) -> Self {
        for g in &generators {
            assert_eq!(len, g.len);
        }
        Self { len, generators }
    }

    /// Return an element's orbit and stabilizer group.
    /// See K. H. Rosen: Computational Group Theory, p. 79.
    pub fn orbit_stabilizer(&self, element: usize) -> OrbitStabilizerInfo {
        let mut orbit = vec![element];
        let mut from = HashMap::from([(element, Permutation::identity(self.len))]);
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
    pub fn permuted(self, p: &Permutation) -> PermutationGroup {
        let generators = self.generators.into_iter().map(|g| p.clone() * g * p.inverse().clone()).collect();
        PermutationGroup::new(self.len, generators)
    }

    pub fn intersection(&self, _other: &PermutationGroup) -> PermutationGroup {
        println!("Warning: Intersection was called although it's not implemented yet.");
        PermutationGroup::new(self.len, vec![])
    }
}

impl<P: Permutable> Mul<&P> for &Permutation {
    type Output = P;

    fn mul(self, rhs: &P) -> P { P::permuted(rhs, self) }
}

pub trait Permutable: Symbol {
    fn len(&self) -> usize;

    fn swap(&mut self, i: usize, j: usize);

    fn permuted(&self, x: &Permutation) -> Self {
        self._permuted_from_swap(x)
    }

    fn _permuted_from_swap(&self, x: &Permutation) -> Self {
        assert_eq!(self.len(), x.len);
        let mut p = Permutation::identity(self.len());
        let mut p_inv = Permutation::identity(self.len());
        let mut out = self.clone();
        for i in (0..self.len()).rev() {
            let xi = x * i;
            let j = &p_inv * xi;
            let pi = &p * i;
            p_inv.swap(pi, xi);
            out.swap(pi, xi);
            p.swap(i, j);
        }

        out
    }

    fn shuffled(&self, seed: usize) -> Self {
        self.permuted(&PermutationUniform { len: self.len() }.sample(seed))
    }

    /// Any implementation should pass this test for any seed.
    fn test_left_group_action_axioms(&self, seed: usize) {
        let (p0, p1) = &PermutationUniform { len: self.len() }.samples(2, seed).into_iter().collect_tuple().unwrap();
        assert_eq!((p0 * self).len(), self.len());
        assert_eq!(&(&Permutation::identity(self.len()) * self), self);
        assert_eq!(p1 * &(p0 * self), &(p1.clone() * p0.clone()) * self);
    }
}

pub trait GroupPermutable: Permutable {
    fn automorphism_group(&self) -> PermutationGroup;
    fn canon(&self) -> Permutation;

    fn canonized(&self) -> (Self, Permutation) {
        let c = self.canon();
        (&c * self, c)
    }

    fn is_isomorphic(&self, other: &Self) -> bool {
        self.canonized().0 == other.canonized().0
    }

    fn orbits(&self) -> Orbits {
        println!("Warning: Orbits was called although it's not implemented yet.");
        Orbits::new()
    }

    /// Fused method for getting automorphism group, canonical permutation and orbits all at once,
    /// allowing optimized implementations.
    fn automorphisms(&self) -> Automorphisms {
        let group = self.automorphism_group();
        Automorphisms {
            group: group.clone(),
            canon: self.canon(),
            decanon: self.canon().inverse(),
            orbits: self.orbits(),
            bits: RCosetUniform::new(group).uni_bits(),
        }
    }

    /// Any implementation should pass this test for any seed.
    fn test(&self, seed: usize) {
        self.test_left_group_action_axioms(seed);

        let p = &PermutationUniform { len: self.len() }.sample(seed);

        // Canonical labelling axiom, expressed in 5 equivalent ways:
        let y = p * self;
        let y_canon = y.canon();
        assert_eq!(y.permuted(&y_canon), self.permuted(&self.canon()));
        assert_eq!(&y_canon * &y, &self.canon() * self);
        assert_eq!(y.canonized().0, self.canonized().0);
        assert!(y.is_isomorphic(self));
        assert_eq!(Unordered(y.clone()), Unordered(self.clone()));

        // Fused method allowing optimized implementations:
        let y_aut = y.automorphisms();
        assert_eq!(y_aut.canon, y_canon);
        assert_eq!(y_aut.decanon, y_canon.inverse());

        let x_aut = self.automorphisms();

        // misc
        assert_eq!(y.orbits().len(), self.orbits().len());
        assert_eq!(y_aut.orbits.len(), x_aut.orbits.len());

        let y_aut_bits = AutomorphismUniform::from_group(y.automorphism_group()).uni_bits();
        let group = self.automorphism_group();
        let x_aut_bits = AutomorphismUniform::from_group(group).uni_bits();
        assert_bits_eq(y_aut_bits, x_aut_bits);
        assert_bits_eq(y_aut.bits, x_aut.bits);
        assert_bits_eq(x_aut.bits, x_aut_bits);

        // TODO test automorphism_group
    }
}

pub trait GroupPermutableFromFused: Permutable {
    fn auts(&self) -> Automorphisms;
}

impl<T: GroupPermutableFromFused> GroupPermutable for T {
    fn automorphism_group(&self) -> PermutationGroup { self.auts().group }
    fn canon(&self) -> Permutation { self.auts().canon }
    fn orbits(&self) -> Orbits { self.auts().orbits }
    fn automorphisms(&self) -> Automorphisms { self.auts() }
}

pub trait Partial: Permutable {
    type Complete: Permutable;
    type Slice;

    fn pop_slice(&mut self) -> Self::Slice;
    fn push_slice(&mut self, slice: &Self::Slice);
    fn empty(len: usize) -> Self;
    fn from_complete(complete: Self::Complete) -> Self;
    fn into_complete(self) -> Self::Complete;
    fn last_(&self) -> usize { self.len() - 1 }
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

impl<P: GroupPermutable, Q: GroupPermutable> GroupPermutable for (P, Q) {
    fn automorphism_group(&self) -> PermutationGroup {
        self.0.automorphism_group().intersection(&self.1.automorphism_group())
    }

    fn canon(&self) -> Permutation {
        let canon0 = self.0.canon();
        // Permute the fused object so that the new 0 is canonized:
        let new = &canon0 * self;
        // Permute the fused object to minimize 1's canonical labels while keeping new 0 canonical:
        let chain = new.0.automorphism_group().lex_stabilizer_chain();
        let (_, restore1_aut) = chain.coset_min_and_restore_aut(&new.1.canon());
        let min1_aut = restore1_aut.total().inverse();
        // The canonized fused object is &min1_aut * &new.
        // Return chain of perms that we applied to get it:
        min1_aut * canon0
    }
}

impl<P: Partial, Q: Partial> Partial for (P, Q) {
    type Complete = (P::Complete, Q::Complete);
    type Slice = (P::Slice, Q::Slice);

    fn pop_slice(&mut self) -> Self::Slice {
        (self.0.pop_slice(), self.1.pop_slice())
    }

    fn push_slice(&mut self, slice: &Self::Slice) {
        self.0.push_slice(&slice.0);
        self.1.push_slice(&slice.1);
    }

    fn empty(len: usize) -> Self {
        (P::empty(len), Q::empty(len))
    }

    fn from_complete(complete: Self::Complete) -> Self {
        (P::from_complete(complete.0), Q::from_complete(complete.1))
    }

    fn into_complete(self) -> Self::Complete {
        (self.0.into_complete(), self.1.into_complete())
    }
}

#[derive(Clone, Debug)]
pub struct Unordered<P: Permutable>(pub P);

impl<P: Permutable> Unordered<P> {
    pub fn into_ordered(self) -> P { self.0 }
    pub fn to_ordered(&self) -> &P { &self.0 }
    pub fn len(&self) -> usize { self.to_ordered().len() }
}

impl<P: GroupPermutable> Unordered<P> {
    pub fn canonized(&self) -> P { self.to_ordered().canonized().0 }
}

impl<P: GroupPermutable> PartialEq<Self> for Unordered<P> {
    fn eq(&self, other: &Self) -> bool {
        self.to_ordered().is_isomorphic(other.to_ordered())
    }
}

impl<P: GroupPermutable> Eq for Unordered<P> {}

#[cfg(test)]
pub mod tests {
    use std::collections::BTreeMap;

    use itertools::Itertools;

    use crate::ans::Codec;
    use crate::shuffle_ans::PermutationUniform;

    use super::*;

    #[test]
    fn permutation_elements() {
        let p0 = Permutation { len: 5, indices: BTreeMap::from([(2, 3), (3, 4), (4, 2)]) };
        let p1 = Permutation { len: 5, indices: BTreeMap::from([(0, 1), (1, 3), (2, 0), (3, 2)]) };
        let p2 = p1 * p0;
        assert_eq!(p2.indices, BTreeMap::from([(0, 1), (1, 3), (3, 4), (4, 0)]));
    }

    #[test]
    fn permutation_axioms() {
        for i in 0..20 {
            let (p0, p1, p2) = PermutationUniform { len: i }.samples(3, i).into_iter().collect_tuple().unwrap();

            // Permutation group axioms:
            assert_eq!(Permutation::identity(i) * p0.clone(), p0);
            assert_eq!(p0.inverse() * p0.clone(), Permutation::identity(i));
            assert_eq!(p0.clone() * p0.inverse(), Permutation::identity(i));
            assert_eq!((p2.clone() * p1.clone()) * p0.clone(), p2.clone() * (p1.clone() * p0.clone()));

            // Compatibility with integer:
            assert_eq!(&Permutation::identity(i) * 2, 2);
            assert_eq!(&(p2.clone() * p1.clone()) * 2, &p2 * (&p1 * 2));
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
        let generators = vec![Permutation::from(vec![0, 1, 3, 2])];
        let g = PermutationGroup::new(4, generators);
        let r = g.orbit_stabilizer(2);
        assert_eq!(r.orbit, Orbit::from([2, 3]));
        assert_eq!(r.stabilizer.generators, vec![]);
    }

    #[test]
    fn orbit_stabilizer2() {
        let generators = vec![Permutation::from(vec![0, 1, 3, 2]), Permutation::from(vec![1, 0, 2, 3])];
        let g = PermutationGroup::new(4, generators);
        let r = g.orbit_stabilizer(2);
        assert_eq!(r.orbit, Orbit::from([2, 3]));
        assert_eq!(r.stabilizer.generators, vec![Permutation::from(vec![1, 0, 2, 3])]);
    }

    #[test]
    fn permuted_group() {
        let g = &PermutationGroup::new(4, vec![Permutation::from(vec![0, 1, 3, 2])]);
        let p = &Permutation::from(vec![3, 0, 1, 2]);
        let g_p = g.clone().permuted(p);

        for e in 0..3 {
            let orbit = g.orbit_stabilizer(e).orbit.iter().map(|x| p * *x).collect_vec();
            let orbit_ = g_p.orbit_stabilizer(p * e).orbit.into_iter().collect_vec();
            assert_eq!(orbit, orbit_)
        }
    }
}
