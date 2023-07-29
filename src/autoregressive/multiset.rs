//! Autoregressive shuffle coding for multisets.
//! This is a high-performance implementation of Severo et al. 2020, more than 100x faster.

use crate::autoregressive::prefix_orbit::{OrbitsById, PrefixOrbitCodec};
use crate::autoregressive::{AutoregressiveShuffleCodec, InnerSliceCodecs, PrefixFn, PrefixingChain, UncachedPrefixFn, UnfusedAutoregressiveShuffleCodec};
use crate::codec::avl::AvlCategorical;
use crate::codec::{Codec, ConstantCodec, EqSymbol, OrdSymbol, Symbol, IID};
use crate::permutable::{FBuildHasher, FHashSet, Len, Permutable};
use itertools::Itertools;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::BuildHasher;
use std::iter::repeat_n;
use std::marker::PhantomData;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VecPrefixingChain<T: EqSymbol>(pub PhantomData<T>);

impl<T: EqSymbol> PrefixingChain for VecPrefixingChain<T> {
    type Prefix = Vec<T>;
    type Full = Vec<T>;
    type Slice = T;

    fn pop_slice(&self, prefix: &mut Vec<T>) -> Self::Slice {
        prefix.pop().unwrap()
    }

    fn push_slice(&self, prefix: &mut Vec<T>, slice: &Self::Slice) {
        prefix.push(slice.clone())
    }

    fn prefix(&self, full: Vec<T>) -> Vec<T> {
        full
    }

    fn full(&self, prefix: Vec<T>) -> Vec<T> {
        prefix
    }
}

impl<T: EqSymbol> VecPrefixingChain<T> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IIDVecSliceCodecs<S: Codec<Symbol: Eq> + Symbol> {
    len: usize,
    slice: S,
}

impl<S: Codec<Symbol: Eq> + Symbol> UncachedPrefixFn<VecPrefixingChain<S::Symbol>> for IIDVecSliceCodecs<S> {
    type Output = S;

    fn apply(&self, _: &Vec<S::Symbol>) -> Self::Output { self.slice.clone() }
}

impl<S: Codec<Symbol: Eq> + Symbol> Len for IIDVecSliceCodecs<S> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<S: Codec<Symbol: Eq> + Symbol> InnerSliceCodecs<VecPrefixingChain<S::Symbol>> for IIDVecSliceCodecs<S> {
    fn prefixing_chain(&self) -> VecPrefixingChain<S::Symbol> {
        VecPrefixingChain::new()
    }

    fn empty_prefix(&self) -> impl Codec<Symbol=Vec<S::Symbol>> {
        ConstantCodec(Vec::with_capacity(self.len))
    }
}

impl<S: Codec<Symbol: Eq> + Symbol> IIDVecSliceCodecs<S> {
    #[allow(unused)]
    pub fn new(len: usize, slice: S) -> Self {
        IIDVecSliceCodecs { len, slice }
    }
}

/// Not all possible orbit ids are known in advance. (Otherwise, use `FixPrefixOrbitCodec` instead.)
pub type DynPrefixOrbitCodec<OrbitId, Hasher = FBuildHasher> = PrefixOrbitCodec<MapOrbits<OrbitId, Hasher>, AvlCategorical<OrbitId>>;

impl<OrbitId, Hasher> DynPrefixOrbitCodec<OrbitId, Hasher>
where
    OrbitId: OrdSymbol + Default,
    Hasher: BuildHasher + Symbol + Default,
{
    pub fn new(ids: Vec<OrbitId>, len: usize) -> Self {
        let orbits = MapOrbits::new(&ids, len);
        let masses = orbits.orbits.iter().map(|(id, p)| (id.clone(), p.len()));
        let categorical = AvlCategorical::from_iter(masses);
        Self { ids, len, orbits, categorical }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MapOrbits<Id: OrdSymbol, Hasher: BuildHasher + Clone = FBuildHasher> {
    pub orbits: HashMap<Id, Orbit, Hasher>,
}

impl<Id: OrdSymbol, Hasher: BuildHasher + Clone + Default> OrbitsById for MapOrbits<Id, Hasher> {
    type Id = Id;

    fn swap(&mut self, i: usize, j: usize, id_i: Self::Id, id_j: Self::Id) {
        let pi = self.orbits.get_mut(&id_i).unwrap();
        assert!(pi.remove(&i));
        assert!(pi.insert(j));
        let pj = self.orbits.get_mut(&id_j).unwrap();
        assert!(pj.remove(&j));
        assert!(pj.insert(i));
    }

    fn insert(&mut self, id: Id, element: usize) {
        self.orbits.entry(id.clone()).or_default().insert(element);
    }

    fn remove(&mut self, id: &Id, element: usize) {
        let partition = self.orbits.get_mut(id).unwrap();
        partition.remove(&element);
        if partition.is_empty() {
            self.orbits.remove(id);
        }
    }

    fn index(&self, id: &Id) -> &usize {
        self.orbits[id].iter().next().unwrap()
    }
}

impl<Id: OrdSymbol, Hasher: BuildHasher + Symbol + Default> MapOrbits<Id, Hasher> {
    pub fn new(ids: &[Id], len: usize) -> Self {
        let mut indices = (0..len).collect_vec();
        indices.par_sort_unstable_by_key(|&i| &ids[i]);
        let orbits = indices.
            par_chunk_by(|i, j| ids[*i] == ids[*j]).
            map(|orbit| (ids[orbit[0]].clone(), orbit.iter().cloned().collect::<Orbit>())).
            collect();
        Self { orbits }
    }
}


#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct VecComplete {
    pub len: usize,
}

impl<S: OrdSymbol + Default> PrefixFn<VecPrefixingChain<S>> for VecComplete {
    type Output = DynPrefixOrbitCodec<S>;

    fn apply(&self, x: &Vec<S>) -> Self::Output {
        let mut ids = x.clone();
        ids.extend(repeat_n(S::default(), self.len - x.len()));
        DynPrefixOrbitCodec::new(ids, x.len())
    }

    fn update_after_pop_slice(&self, orbits: &mut Self::Output, _x: &Vec<S>, _slice: &S) {
        orbits.pop_id();
    }

    fn update_after_push_slice(&self, orbits: &mut Self::Output, x: &Vec<S>, slice: &S) {
        orbits.ids[x.len() - 1] = slice.clone();
        orbits.push_id();
    }

    fn swap(&self, orbits: &mut Self::Output, i: usize, j: usize) {
        orbits.swap(i, j)
    }
}

impl VecComplete {
    #[allow(unused)]
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}


#[allow(unused)]
pub fn iid_multiset_shuffle_codec<S: Codec<Symbol: OrdSymbol + Default> + Symbol>(codec: &IID<S>) -> UnfusedAutoregressiveShuffleCodec<VecPrefixingChain<S::Symbol>, IIDVecSliceCodecs<S>, VecComplete> {
    AutoregressiveShuffleCodec::new(IIDVecSliceCodecs::new(codec.len, codec.item.clone()), VecComplete::new(codec.len))
}

pub type Orbit = FHashSet<usize>;
