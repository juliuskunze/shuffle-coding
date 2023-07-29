use crate::permutable::{Permutable, Permutation};
use std::collections::BTreeMap;
use std::ops::Mul;

/// Permutation a given length, sparsely represented.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SparsePerm {
    pub len: usize,
    pub indices: BTreeMap<usize, usize>,
}

impl From<Vec<usize>> for SparsePerm {
    fn from(items: Vec<usize>) -> Self {
        let items = items.into_iter();
        let len = items.len();
        Self {
            len,
            indices: items.enumerate().filter(|(i, x)| {
                assert!(*x < len);
                i != x
            }).map(|(i, x)| (i, x)).collect(),
        }
    }
}

impl Permutation for SparsePerm {
    fn identity(len: usize) -> Self { Self { len, indices: BTreeMap::new() } }

    fn inverse(&self) -> Self {
        Self { len: self.len, indices: self.indices.iter().map(|(&x, &y)| (y, x)).collect() }
    }

    fn iter(&self) -> impl Iterator<Item=usize> + '_ {
        (0..self.len).map(move |i| self * i).into_iter()
    }

    fn apply(&self, i: usize) -> usize {
        *self.indices.get(&i).unwrap_or(&i)
    }

    fn is_identity(&self) -> bool { self.indices.is_empty() }

    fn create_swap(len: usize, i: usize, j: usize) -> Self {
        Self { len, indices: if i == j { BTreeMap::new() } else { BTreeMap::from_iter(vec![(i, j), (j, i)]) } }
    }
}

impl SparsePerm {
    fn normalize(&mut self) {
        self.indices.retain(|i, x| i != x);
    }

    pub fn set_len(&mut self, len: usize) {
        assert!(len >= self.len || self.indices.iter().all(|(&x, &y)| x < len && y < len));
        self.len = len;
    }
}

/// Permutations of a given length are ordered objects forming permutable class.
impl Permutable for SparsePerm {
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

impl Mul<SparsePerm> for SparsePerm {
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

impl Mul<usize> for &SparsePerm {
    type Output = usize;

    fn mul(self, i: usize) -> usize { self.apply(i) }
}

#[cfg(test)]
pub mod tests {
    use std::collections::BTreeMap;

    use itertools::Itertools;

    use super::*;
    use crate::codec::Codec;
    use crate::experimental::complete_joint::sparse_perm::SparsePerm;
    use crate::permutable::tests::test_perm_and_uniform;
    use crate::permutable::PermutationUniform;

    #[test]
    fn permutation_elements() {
        let p0 = SparsePerm { len: 5, indices: BTreeMap::from([(2, 3), (3, 4), (4, 2)]) };
        let p1 = SparsePerm { len: 5, indices: BTreeMap::from([(0, 1), (1, 3), (2, 0), (3, 2)]) };
        let p2 = p1 * p0;
        assert_eq!(p2.indices, BTreeMap::from([(0, 1), (1, 3), (3, 4), (4, 0)]));
    }

    #[test]
    fn permutation_axioms() {
        for i in 0..20 {
            let (p0, p1, p2) = PermutationUniform::<SparsePerm>::new(i).samples(3, i).into_iter().collect_tuple().unwrap();

            // Permutation group axioms:
            assert_eq!(SparsePerm::identity(i) * p0.clone(), p0);
            assert_eq!(p0.inverse() * p0.clone(), SparsePerm::identity(i));
            assert_eq!(p0.clone() * p0.inverse(), SparsePerm::identity(i));
            assert_eq!((p2.clone() * p1.clone()) * p0.clone(), p2.clone() * (p1.clone() * p0.clone()));

            // Compatibility with integer:
            assert_eq!(&SparsePerm::identity(i) * 2, 2);
            assert_eq!(&(p2.clone() * p1.clone()) * 2, &p2 * (&p1 * 2));
        }
    }

    #[test]
    fn sparse_perm_and_uniform() {
        test_perm_and_uniform::<SparsePerm>()
    }
}
