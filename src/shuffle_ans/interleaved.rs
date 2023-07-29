use std::fmt::Debug;
use std::marker::PhantomData;

use crate::ans::{Codec, Distribution, Message};
use crate::multiset::OrdSymbol;
use crate::permutable::{Automorphisms, Cell, GroupPermutable, Orbit, Partial, Permutable, Unordered};

#[derive(Clone)]
pub struct InterleavedShuffleCodec<P, D, DFromLen, Pa>
    where P: Partial + GroupPermutable,
          P::Complete: GroupPermutable,
          D: Codec<Symbol=P::Slice>,
          DFromLen: Fn(usize) -> D + Clone,
          Pa: Partitioner<P> {
    pub partial: PartialShuffleCodec<P, D, DFromLen, Pa>,
}

impl<P, D, DFromLen, Pa> Codec for InterleavedShuffleCodec<P, D, DFromLen, Pa> where
    P: Partial + GroupPermutable,
    P::Complete: GroupPermutable,
    D: Codec<Symbol=P::Slice>,
    DFromLen: Fn(usize) -> D + Clone,
    Pa: Partitioner<P> {
    type Symbol = Unordered<P::Complete>;

    fn push(&self, m: &mut Message, Unordered(x): &Self::Symbol) {
        self.partial.push(m, &Unordered(P::from_complete(x.clone())));
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        Unordered(self.partial.pop(m).0.into_complete())
    }

    fn bits(&self, Unordered(x): &Self::Symbol) -> Option<f64> {
        self.partial.bits(&Unordered(P::from_complete(x.clone())))
    }
}

#[derive(Clone, Debug)]
pub struct PartialShuffleCodec<P, D, DFromLen, Pa>
    where P: Partial + GroupPermutable,
          P::Complete: GroupPermutable,
          D: Codec<Symbol=P::Slice>,
          DFromLen: Fn(usize) -> D + Clone,
          Pa: Partitioner<P> {
    pub complete_len: usize,
    pub slice_len: usize,
    pub slice_codec_for_len: DFromLen,
    pub partitioner: Pa,
    pub phantom: PhantomData<P>,
}

impl<P, D, DFromLen, Pa> Codec for PartialShuffleCodec<P, D, DFromLen, Pa> where
    P: Partial + GroupPermutable,
    P::Complete: GroupPermutable,
    D: Codec<Symbol=P::Slice>,
    DFromLen: Fn(usize) -> D + Clone,
    Pa: Partitioner<P> {
    type Symbol = Unordered<P>;

    fn push(&self, m: &mut Message, Unordered(x): &Self::Symbol) {
        assert_eq!(self.len(), x.len());

        let mut x = x.clone();
        if self.slice_len < self.complete_len {
            let cells = &mut self.partitioner.cells(&x);
            for slice_len in self.slice_len..self.complete_len {
                let element = *cells.cell(&cells.pop(m)).first().unwrap();
                let last = x.last_();
                x.swap(element, last);
                cells.swap(element, last);
                let slice = x.pop_slice();
                self.partitioner.update_after_pop_slice(cells, &x, &slice);
                (self.slice_codec_for_len)(slice_len).push(m, &slice);
            }
        }
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let mut x = P::empty(self.complete_len);

        if self.slice_len < self.complete_len {
            let cells = &mut self.partitioner.cells(&x);
            for slice_len in (self.slice_len..self.complete_len).rev() {
                let slice = (self.slice_codec_for_len)(slice_len).pop(m);
                x.push_slice(&slice);
                self.partitioner.update_after_push_slice(cells, &x, &slice);
                cells.push(m, &cells.id(x.last_()));
            }
        }
        Unordered(x)
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> {
        None
    }
}

#[derive(Clone, Debug)]
pub struct OrbitPartitioner;

impl<P: GroupPermutable + Partial> Partitioner<P> for OrbitPartitioner {
    type CellId = Orbit;
    type CellDistribution = OrbitDistribution;

    fn cells(&self, x: &P) -> Self::CellDistribution {
        OrbitDistribution::new(x.automorphisms())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrbitDistribution {
    automorphisms: Automorphisms,
}

impl Distribution for OrbitDistribution {
    type Symbol = Orbit;

    fn norm(&self) -> usize { self.automorphisms.group.len }
    fn pmf(&self, orbit: &Orbit) -> usize { orbit.len() }

    fn cdf(&self, orbit: &Orbit, i: usize) -> usize {
        *self.canonized(orbit).iter().nth(i).unwrap()
    }

    fn icdf(&self, cf: usize) -> (Orbit, usize) {
        let some_element = &self.automorphisms.decanon * cf;
        let orbit = self.automorphisms.orbit(some_element);
        let i = self.canonized(&orbit).iter().position(|&x| x == cf).unwrap();
        (orbit, i)
    }
}

impl Permutable for OrbitDistribution {
    fn len(&self) -> usize {
        self.automorphisms.orbits.len()
    }

    fn swap(&mut self, _i: usize, _j: usize) {
        // Empty because result is never used.
    }
}

impl OrbitDistribution {
    fn new(automorphisms: Automorphisms) -> Self { Self { automorphisms } }

    fn canonized(&self, orbit: &Orbit) -> Orbit {
        Orbit::from_iter(orbit.iter().map(|&x| &self.automorphisms.canon * x))
    }
}

impl CellDistribution<Orbit> for OrbitDistribution {
    fn id(&self, element: usize) -> Orbit {
        self.automorphisms.orbit(element)
    }

    fn cell(&self, cell: &Orbit) -> Cell {
        cell.clone()
    }
}

impl<P, D, DFromLen, Pa> PartialShuffleCodec<P, D, DFromLen, Pa> where
    D: Codec<Symbol=P::Slice>,
    DFromLen: Clone + Fn(usize) -> D,
    P: GroupPermutable + Partial,
    P::Complete: GroupPermutable,
    Pa: Partitioner<P> {
    fn len(&self) -> usize {
        self.complete_len - self.slice_len
    }
}

impl<P, D, DFromLen, Pa> InterleavedShuffleCodec<P, D, DFromLen, Pa> where
    P: Partial + GroupPermutable,
    P::Complete: GroupPermutable,
    D: Codec<Symbol=P::Slice>,
    DFromLen: Fn(usize) -> D + Clone,
    Pa: Partitioner<P> {
    #[allow(unused)]
    pub fn new(len: usize, slice_codec_for_len: DFromLen, partitioner: Pa) -> Self {
        Self { partial: PartialShuffleCodec { complete_len: len, slice_len: 0, slice_codec_for_len, partitioner, phantom: Default::default() } }
    }
}

pub trait CellDistribution<CellId = usize>: Distribution<Symbol=CellId> + Debug {
    fn id(&self, element: usize) -> CellId;

    fn cell(&self, id: &CellId) -> Cell;
}

pub trait Partitioner<P: Partial>: Clone {
    type CellId: OrdSymbol + Default;
    type CellDistribution: CellDistribution<Self::CellId> + Permutable;

    fn cells(&self, x: &P) -> Self::CellDistribution;

    fn update_after_pop_slice(&self, cells: &mut Self::CellDistribution, x: &P, _slice: &P::Slice) {
        *cells = self.cells(x)
    }

    fn update_after_push_slice(&self, cells: &mut Self::CellDistribution, x: &P, _slice: &P::Slice) {
        *cells = self.cells(x)
    }
}