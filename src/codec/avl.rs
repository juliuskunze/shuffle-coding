use crate::codec::{Distribution, MutDistribution, OrdSymbol};
use std::mem;
use std::ops::{Deref, DerefMut};

/// Categorical allowing insertion and removal of mass.
/// Insert, remove, pmf, cdf and icdf all have a runtime of O(log #symbols).
/// Implemented via an order statistic tree.
#[derive(Clone, Debug, Default, Eq)]
pub struct AvlCategorical<S: OrdSymbol + Default = usize> {
    branches: Option<Box<(Self, Self)>>,
    count: usize,
    split: S,
    depth: usize,
}

impl<S: OrdSymbol + Default> Distribution for AvlCategorical<S> {
    type Symbol = S;

    fn norm(&self) -> usize {
        self.count
    }

    fn pmf(&self, x: &Self::Symbol) -> usize {
        if let Some(branches) = &self.branches {
            let (left, right) = branches.deref();
            if x < &self.split { left } else { right }.pmf(x)
        } else {
            if x == &self.split { self.count } else { 0 }
        }
    }

    fn cdf(&self, x: &Self::Symbol, i: usize) -> usize {
        if let Some(branches) = &self.branches {
            let (left, right) = branches.deref();
            if x < &self.split { left.cdf(x, i) } else { left.count + right.cdf(x, i) }
        } else {
            assert_eq!(&self.split, x, "Symbol {x:?} not found in distribution.");
            assert!(i < self.count);
            i
        }
    }

    fn icdf(&self, cf: usize) -> (Self::Symbol, usize) {
        if let Some(branches) = &self.branches {
            let (left, right) = branches.deref();
            if cf < left.count { left.icdf(cf) } else { right.icdf(cf - left.count) }
        } else {
            assert!(cf < self.count);
            (self.split.clone(), cf)
        }
    }
}

impl<S: OrdSymbol + Default> MutDistribution for AvlCategorical<S> {
    fn insert(&mut self, x: Self::Symbol, mass: usize) {
        if mass == 0 {
            return;
        }

        let is_left = x < self.split;
        if let Some(branches) = &mut self.branches {
            assert_ne!(self.count, 0);
            let (left, right) = branches.deref_mut();
            let tree = if is_left { left } else { right };
            tree.insert(x, mass);
        } else if x != self.split {
            if self.count == 0 {
                self.split = x;
            } else {
                let new = Self::leaf(x.clone(), mass);
                let prev = Self::leaf(self.split.clone(), self.count);
                self.branches = Some(Box::new(if is_left {
                    (new, prev)
                } else {
                    self.split = x;
                    (prev, new)
                }));
            }
        }
        self.count += mass;
        self.rebalance_and_update_depth();
    }

    fn remove(&mut self, x: &Self::Symbol, mass: usize) {
        assert!(mass <= self.count);
        self.count -= mass;

        if let Some(branches) = &mut self.branches {
            let (left, right) = branches.deref_mut();
            let is_left = x < &self.split;
            let tree = if is_left { left } else { right };
            tree.remove(x, mass);
            if tree.count == 0 {
                let (left, right) = branches.deref_mut();
                *self = mem::take(if is_left { right } else { left });
            } else {
                self.rebalance_and_update_depth();
            }
            assert_ne!(self.count, 0);
        }
    }
}

impl<S: OrdSymbol + Default> PartialEq for AvlCategorical<S> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<S: OrdSymbol + Default> AvlCategorical<S> {
    fn update_depth(&mut self) {
        self.depth = if let Some(branches) = &self.branches {
            let (left, right) = branches.deref();
            left.depth.max(right.depth) + 1
        } else { 0 };
    }

    fn update_depth_and_count(&mut self) {
        self.update_depth();
        let (l, r) = self.branches_mut();
        self.count = l.count + r.count;
    }

    fn rebalance_and_update_depth(&mut self) {
        let balance = self.balance_factor();
        if balance > 1 {
            let left = self.left();
            if left.balance_factor() < 0 {
                left.rotate_left();
            }
            self.rotate_right();
        } else if balance < -1 {
            let right = self.right();
            if right.balance_factor() > 0 {
                right.rotate_right();
            }
            self.rotate_left();
        } else {
            self.update_depth();
        }
    }


    fn balance_factor(&self) -> isize {
        if let Some(branches) = &self.branches {
            let (left, right) = branches.deref();
            left.depth as isize - right.depth as isize
        } else { 0 }
    }

    fn rotate_left(&mut self) { self.rotate(false) }
    fn rotate_right(&mut self) { self.rotate(true) }
    fn rotate(&mut self, right: bool) {
        // Variable names are according to the case of right rotation (right == true):
        let get_left = if right { Self::left } else { Self::right };
        let get_right = if right { Self::right } else { Self::left };
        let left = get_left(self);
        let left_right = mem::take(get_right(left));
        let mut left = mem::replace(left, left_right);
        let left_right = get_right(&mut left);
        *left_right = mem::take(self);
        left_right.update_depth_and_count();
        *self = left;
        self.update_depth_and_count();
    }

    fn left(&mut self) -> &mut Self {
        &mut self.branches_mut().0
    }

    fn right(&mut self) -> &mut Self {
        &mut self.branches_mut().1
    }

    fn branches_mut(&mut self) -> &mut (AvlCategorical<S>, AvlCategorical<S>) {
        self.branches.as_mut().unwrap().deref_mut()
    }

    #[allow(unused)]
    pub fn iter(&self) -> MutCategoricalIter<S> {
        let mut iter = MutCategoricalIter { stack: Vec::new() };
        iter.push_left_edge(self);
        iter
    }

    #[allow(unused)]
    pub fn max(&self) -> S {
        let mut a = self;
        while let Some(branch) = &a.branches {
            a = &branch.deref().1;
        }

        a.split.clone()
    }

    fn leaf(x: S, count: usize) -> Self {
        Self { branches: None, count, split: x, depth: 0 }
    }
}

impl<S: OrdSymbol + Default> FromIterator<(S, usize)> for AvlCategorical<S> {
    fn from_iter<I>(masses: I) -> Self
    where
        I: IntoIterator<Item=(S, usize)>,
    {
        let mut out: Option<Self> = None;
        for (x, count) in masses {
            if let Some(out) = &mut out {
                out.insert(x, count);
            } else {
                out = Some(Self::leaf(x, count));
            }
        }
        out.unwrap_or_else(Self::default)
    }
}

#[derive(Clone, Debug)]
pub struct MutCategoricalIter<'a, S: OrdSymbol + Default = usize> {
    stack: Vec<&'a AvlCategorical<S>>,
}

impl<'a, S: OrdSymbol + Default> MutCategoricalIter<'a, S> {
    fn push_left_edge(&mut self, mut node: &'a AvlCategorical<S>) {
        loop {
            if node.count == 0 {
                assert!(node.branches.is_none());
                break;
            }
            self.stack.push(node);
            if let Some(branch) = &node.branches {
                node = &branch.deref().0;
            } else {
                break;
            }
        }
    }
}

impl<'a, S: OrdSymbol + Default> Iterator for MutCategoricalIter<'a, S> {
    type Item = (S, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.stack.pop() {
            if let Some(branch) = &node.branches {
                self.push_left_edge(&branch.deref().1);
            } else {
                assert_ne!(node.count, 0);
                return Some((node.split.clone(), node.count.clone()));
            }
        }
        None
    }
}
