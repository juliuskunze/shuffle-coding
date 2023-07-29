//! Orbit codecs for complete autoregressive shuffle coding using `AutCanonizable::automorphisms()`
//! at every iteration. Very slow, mostly here for reference.

use crate::autoregressive::multiset::Orbit;
use crate::autoregressive::{OrbitCodec, PrefixingChain, UncachedPrefixFn};
use crate::codec::Distribution;
use crate::experimental::complete_joint::aut::{AutCanonizable, Automorphisms};
use crate::permutable::Complete;
use itertools::Itertools;

impl<P: PrefixingChain<Prefix: AutCanonizable>> UncachedPrefixFn<P> for Complete {
    type Output = CompleteOrbitCodec;

    fn apply(&self, x: &P::Prefix) -> Self::Output {
        CompleteOrbitCodec::new(x.automorphisms())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompleteOrbitCodec {
    automorphisms: Automorphisms,
}

impl Distribution for CompleteOrbitCodec {
    type Symbol = Orbit;

    fn norm(&self) -> usize { self.automorphisms.group.len }
    fn pmf(&self, orbit: &Orbit) -> usize { orbit.len() }

    fn cdf(&self, orbit: &Orbit, i: usize) -> usize {
        *self.canonized(orbit).iter().sorted_unstable().nth(i).unwrap()
    }

    fn icdf(&self, cf: usize) -> (Orbit, usize) {
        let some_element = &self.automorphisms.decanon * cf;
        let orbit = self.automorphisms.orbit(some_element);
        let i = self.canonized(&orbit).iter().sorted_unstable().position(|&x| x == cf).unwrap();
        (orbit, i)
    }
}

impl CompleteOrbitCodec {
    fn new(automorphisms: Automorphisms) -> Self { Self { automorphisms } }

    fn canonized(&self, orbit: &Orbit) -> Orbit {
        Orbit::from_iter(orbit.iter().map(|&x| &self.automorphisms.canon * x))
    }
}

impl OrbitCodec for CompleteOrbitCodec {
    fn id(&self, index: usize) -> Orbit {
        self.automorphisms.orbit(index)
    }

    fn index(&self, id: &Orbit) -> usize { *id.iter().next().unwrap() }
}

impl Automorphisms {
    fn orbit(&self, index: usize) -> Orbit {
        let id = self.orbit_ids[index];
        self.orbit_ids.iter().enumerate().filter(|(_, &id_)| id_ == id).map(|(i, _)| i).collect()
    }
}
