use crate::autoregressive::OrbitCodec;
use crate::codec::{Distribution, MutDistribution, OrdSymbol};
use crate::permutable::FHashMap;
use crate::permutable::Permutable;
use ftree::FenwickTree;
use itertools::Itertools;
use rayon::prelude::*;
use std::ops::Range;

pub trait OrbitsById {
    type Id: OrdSymbol;
    fn swap(&mut self, i: usize, j: usize, id_i: Self::Id, id_j: Self::Id);

    fn insert(&mut self, id: Self::Id, element: usize);

    fn remove(&mut self, id: &Self::Id, element: usize);

    fn index(&self, id: &Self::Id) -> &usize;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VecOrbits {
    pub indices: Vec<usize>,
    pub orbits: Vec<Range<usize>>,
}

impl VecOrbits {
    pub fn new(mut ids: Vec<usize>, len: usize) -> Self {
        let mut indices = (0..ids.len()).collect_vec();
        indices.par_sort_by_key(|&i| ids[i]);
        ids.par_sort_unstable();
        let ids_ = &ids;
        let orbits = ids.
            par_chunk_by(|a, b| a == b).
            map(|run| {
                let start = unsafe { run.as_ptr().offset_from(ids_.as_ptr()) } as usize;
                let len = indices[start..start + run.len()].partition_point(|i| *i < len);
                start..start + len
            }).
            collect();
        Self { indices, orbits }
    }

    fn orbit(&mut self, id: usize) -> &mut [usize] {
        &mut self.indices[self.orbits[id].clone()]
    }
}

impl OrbitsById for VecOrbits {
    type Id = usize;

    fn swap(&mut self, i: usize, j: usize, id_i: Self::Id, id_j: Self::Id) {
        *self.orbit(id_i).iter_mut().find(|x| **x == i).unwrap() = j;
        *self.orbit(id_j).iter_mut().find(|x| **x == j).unwrap() = i;
    }

    fn insert(&mut self, id: Self::Id, element: usize) {
        let end = &mut self.orbits[id].end;
        self.indices[*end] = element;
        *end += 1;
    }

    fn remove(&mut self, id: &Self::Id, element: usize) {
        let index = self.orbits[*id].start + self.orbit(*id).iter().position(|x| *x == element).unwrap();
        let end = &mut self.orbits[*id].end;
        *end -= 1;
        self.indices.swap(index, *end);
    }

    fn index(&self, id: &Self::Id) -> &usize {
        &self.indices[self.orbits[*id].start]
    }
}

/// Orbit codec data structure useful for autoregressive shuffle coding on multisets and graphs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixOrbitCodec<O: OrbitsById, D: MutDistribution<Symbol=O::Id>> {
    pub ids: Vec<O::Id>,
    pub len: usize,
    pub orbits: O,
    pub categorical: D,
}

impl<O, D> Permutable for PrefixOrbitCodec<O, D>
where
    O: OrbitsById + Clone,
    D: MutDistribution<Symbol=O::Id> + Clone,
{
    fn len(&self) -> usize { self.len }

    fn swap(&mut self, i: usize, j: usize) {
        assert!(i < self.len && j < self.len);
        let id_i = self.id(i);
        let id_j = self.id(j);
        if id_i == id_j {
            return;
        }
        self.orbits.swap(i, j, id_i, id_j);
        self.ids.swap(i, j);
    }
}

impl<O, D> PrefixOrbitCodec<O, D>
where
    O: OrbitsById + Clone,
    D: MutDistribution<Symbol=O::Id> + Clone,
{
    pub fn push_id(&mut self) {
        let element = self.len;
        let id = self.ids[element].clone();
        self.len += 1;

        self.orbits.insert(id.clone(), element);
        self.categorical.insert(id, 1);
    }

    pub fn pop_id(&mut self) {
        self.len -= 1;
        let element = self.len;
        let id = &self.ids[element];

        self.orbits.remove(&id, element);
        self.categorical.remove(id, 1);
    }
}

impl<O, D> Distribution for PrefixOrbitCodec<O, D>
where
    O: OrbitsById + Clone,
    D: MutDistribution<Symbol=O::Id> + Clone,
{
    type Symbol = O::Id;

    fn norm(&self) -> usize {
        self.categorical.norm()
    }
    fn pmf(&self, x: &Self::Symbol) -> usize { self.categorical.pmf(&x) }

    fn cdf(&self, x: &Self::Symbol, i: usize) -> usize {
        self.categorical.cdf(&x, i)
    }

    fn icdf(&self, cf: usize) -> (Self::Symbol, usize) {
        self.categorical.icdf(cf)
    }
}

impl<O, D> OrbitCodec for PrefixOrbitCodec<O, D>
where
    O: OrbitsById + Clone,
    D: MutDistribution<Symbol=O::Id> + Clone,
{
    fn id(&self, index: usize) -> O::Id {
        self.ids[index].clone()
    }

    fn index(&self, id: &O::Id) -> usize {
        *self.orbits.index(id)
    }
}

/// All possible orbit ids are known in advance, at creation time.
pub type FixPrefixOrbitCodec = PrefixOrbitCodec<VecOrbits, FenwickTree<usize>>;

impl FixPrefixOrbitCodec {
    pub fn new<OrbitId: OrdSymbol + Default>(ids: Vec<OrbitId>, len: usize) -> Self {
        let ids = Self::ranks(&ids);
        let orbits = VecOrbits::new(ids.clone(), len);
        let masses = orbits.orbits.iter().map(|p| p.len());
        let categorical = FenwickTree::from_iter(masses);
        Self { ids, len, orbits, categorical }
    }

    fn ranks<OrbitId: OrdSymbol + Default>(ids: &Vec<OrbitId>) -> Vec<usize> {
        let mut hashes = ids.clone();
        hashes.par_sort_unstable();
        hashes.dedup();
        let map: FHashMap<_, _> = hashes.into_iter().enumerate().map(|(i, h)| (h, i)).collect();
        ids.into_iter().map(|id| map[&id]).collect_vec()
    }
}
