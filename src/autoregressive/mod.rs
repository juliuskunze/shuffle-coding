//! Autoregressive shuffle coding.
use crate::codec::{Codec, Message, Symbol};
use crate::permutable::{Len, Permutable, Unordered};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Deref;

pub mod graph;
pub mod multiset;
pub mod prefix_orbit;
pub mod chunked;

/// A prefixing chain on a permutable class.
/// Allows to pop or push a slice onto a prefix to obtain a new prefix.
/// Also allows to convert a full object to a prefix and back.
pub trait PrefixingChain {
    type Prefix: Permutable;
    type Full: Permutable;
    type Slice;

    fn pop_slice(&self, prefix: &mut Self::Prefix) -> Self::Slice;
    fn push_slice(&self, prefix: &mut Self::Prefix, slice: &Self::Slice);
    fn prefix(&self, full: Self::Full) -> Self::Prefix;
    fn full(&self, prefix: Self::Prefix) -> Self::Full;
}

/// A function defined on prefixes. If the prefix is the result of a pop or push operation
/// the output can be computed based on the original prefix and the popped/pushed slice, for efficiency.
pub trait PrefixFn<P: PrefixingChain> {
    type Output;

    fn apply(&self, x: &P::Prefix) -> Self::Output;
    fn update_after_pop_slice(&self, image: &mut Self::Output, x: &P::Prefix, _slice: &P::Slice);
    fn update_after_push_slice(&self, image: &mut Self::Output, x: &P::Prefix, _slice: &P::Slice);
    fn swap(&self, _image: &mut Self::Output, _i: usize, _j: usize);
}

/// Uncached implementation of PrefixFn, recomputing the output from scratch after each push/pop.
pub trait UncachedPrefixFn<P: PrefixingChain> {
    type Output;
    fn apply(&self, x: &P::Prefix) -> Self::Output;
}

impl<P: PrefixingChain, F: UncachedPrefixFn<P>> PrefixFn<P> for F {
    type Output = F::Output;

    fn apply(&self, x: &P::Prefix) -> Self::Output {
        self.apply(x)
    }

    fn update_after_pop_slice(&self, image: &mut Self::Output, x: &P::Prefix, _slice: &P::Slice) {
        *image = self.apply(x)
    }

    fn update_after_push_slice(&self, image: &mut Self::Output, x: &P::Prefix, _slice: &P::Slice) {
        *image = self.apply(x)
    }

    /// The result of swap is only used to update the image passed into update_after_pop/push_slice.
    /// The value of image argument is unused here, so we don't need to implement swap.
    fn swap(&self, _image: &mut Self::Output, _i: usize, _j: usize) {}
}

pub trait InnerSliceCodecs<P: PrefixingChain<Prefix: Symbol>>: Len {
    fn prefixing_chain(&self) -> P;
    fn empty_prefix(&self) -> impl Codec<Symbol=P::Prefix>;
}

/// Autoregressive model for permutable objects
/// represented by a function returning a codec for a slice given an adjacent prefix.
pub trait SliceCodecs<P: PrefixingChain<Prefix: Symbol>>: PrefixFn<P, Output: Codec<Symbol=P::Slice>> + InnerSliceCodecs<P> {}
impl<P: PrefixingChain<Prefix: Symbol, Full: Symbol>, S: PrefixFn<P, Output: Codec<Symbol=P::Slice>> + InnerSliceCodecs<P>> SliceCodecs<P> for S {}

/// A function returning a codec for the orbits of a given prefix.
pub trait OrbitCodecs<P: PrefixingChain>: PrefixFn<P, Output: OrbitCodec> {}
impl<P: PrefixingChain, O: PrefixFn<P, Output: OrbitCodec>> OrbitCodecs<P> for O {}

/// A codec for orbits of a permutable object, represented as an orbit id of type Self::Symbol.
pub trait OrbitCodec: Codec {
    /// Returns the orbit id corresponding to the given index from 0..self.len().
    fn id(&self, index: usize) -> Self::Symbol;

    /// Returns an element from the orbit with the given orbit id.
    fn index(&self, id: &Self::Symbol) -> usize;

    /// Push an orbit, given one of its elements.
    fn push_element(&self, m: &mut Message, index: usize) {
        self.push(m, &self.id(index));
    }

    /// Pop an orbit, and return one of its elements.
    fn pop_element(&self, m: &mut Message) -> usize {
        self.index(&self.pop(m))
    }
}

pub trait PrefixSliceCodec<P: PrefixingChain>: Permutable {
    type SliceCodec: Codec<Symbol=P::Slice>;

    fn slice_codec(&self) -> impl Deref<Target=Self::SliceCodec>;
    fn prefix(&self) -> impl Deref<Target=P::Prefix>;
}

pub trait SliceCodecPrefixingChain<P: PrefixingChain<Prefix: Symbol>>: PrefixingChain<Prefix: PrefixSliceCodec<P>, Full=P::Prefix, Slice=P::Slice> + InnerSliceCodecs<P> {
    fn push_slice_from_prefix(&self, m: &mut Message, prefix: &mut Self::Prefix) -> Self::Slice {
        let slice = self.pop_slice(prefix);
        prefix.slice_codec().push(m, &slice);
        slice
    }

    fn pop_slice_to_prefix(&self, m: &mut Message, prefix: &mut Self::Prefix) -> Self::Slice {
        let slice = prefix.slice_codec().pop(m);
        self.push_slice(prefix, &slice);
        slice
    }
}

/// The core of autoregressive shuffle coding: A autoregressive codec for prefixes of unordered objects.
/// Written non-recursively for efficiency.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AutoregressivePrefixShuffleCodec<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecPrefixingChain<P>, O: OrbitCodecs<P>> {
    /// Autoregressive model, coding slices given an adjacent prefix.
    pub slice_codecs: S,
    /// Codecs for the orbits of the prefixes of unordered objects.
    pub orbit_codecs: O,
    pub phantom: PhantomData<P>,
}

impl<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecPrefixingChain<P>, O: OrbitCodecs<P>> Codec for AutoregressivePrefixShuffleCodec<P, S, O> {
    type Symbol = Unordered<P::Prefix>;

    fn push(&self, m: &mut Message, Unordered(x): &Self::Symbol) {
        assert_eq!(self.slice_codecs.len(), x.len());
        let prefix = &mut self.slice_codecs.prefix(x.clone());
        let orbit_codec = &mut self.orbit_codecs.apply(&prefix.prefix());
        for _i in 0..self.slice_codecs.len() {
            let index = orbit_codec.pop_element(m);
            let last_index = prefix.len() - 1;
            prefix.swap(index, last_index);
            self.orbit_codecs.swap(orbit_codec, index, last_index);
            let slice = &self.slice_codecs.push_slice_from_prefix(m, prefix);
            self.orbit_codecs.update_after_pop_slice(orbit_codec, &prefix.prefix(), slice);
        }
        self.slice_codecs.empty_prefix().push(m, &prefix.prefix());
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let p = self.slice_codecs.empty_prefix().pop(m);
        let mut prefix = self.slice_codecs.prefix(p);
        let orbit_codec = &mut self.orbit_codecs.apply(&prefix.prefix());
        for _ in 0..self.slice_codecs.len() {
            let slice = &self.slice_codecs.pop_slice_to_prefix(m, &mut prefix);
            self.orbit_codecs.update_after_push_slice(orbit_codec, &prefix.prefix(), slice);
            orbit_codec.push_element(m, prefix.prefix().len() - 1);
        }
        Unordered(self.slice_codecs.full(prefix))
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { None }
}

/// Implements autoregressive shuffle coding as a wrapper around `AutoregressivePrefixShuffleCodec`,
/// mapping prefixes of full length to unordered objects and back.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AutoregressiveShuffleCodec<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecPrefixingChain<P>, O: OrbitCodecs<P>> {
    pub prefix: AutoregressivePrefixShuffleCodec<P, S, O>,
}

impl<P: PrefixingChain<Prefix: Symbol, Full: Symbol>, S: SliceCodecPrefixingChain<P>, O: OrbitCodecs<P>> Codec for AutoregressiveShuffleCodec<P, S, O> {
    type Symbol = Unordered<P::Full>;

    fn push(&self, m: &mut Message, Unordered(x): &Self::Symbol) {
        self.prefix.push(m, &Unordered(self.prefix.slice_codecs.prefixing_chain().prefix(x.clone())));
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        Unordered(self.prefix.slice_codecs.prefixing_chain().full(self.prefix.pop(m).0))
    }

    fn bits(&self, Unordered(x): &Self::Symbol) -> Option<f64> {
        self.prefix.bits(&Unordered(self.prefix.slice_codecs.prefixing_chain().prefix(x.clone())))
    }
}

impl<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecPrefixingChain<P>, O: OrbitCodecs<P>> AutoregressiveShuffleCodec<P, S, O> {
    #[allow(unused)]
    pub fn fused(slice_codecs: S, orbit_codecs: O) -> Self {
        Self { prefix: AutoregressivePrefixShuffleCodec { slice_codecs, orbit_codecs, phantom: PhantomData } }
    }
}

/// SliceCodecs is not directly implementing codec to avoid blanket implementation conflict 
/// of `impl<S: SliceCodecs> Codec for S` with `impl<D: Distribution> Codec for D`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Autoregressive<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecs<P>>(pub PrefixAutoregressive<P, S>);

impl<P: PrefixingChain<Prefix: Symbol, Full: Symbol>, S: SliceCodecs<P>> Codec for Autoregressive<P, S> {
    type Symbol = P::Full;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.0.push(m, &self.0.slices.prefixing_chain().prefix(x.clone()))
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        self.0.slices.prefixing_chain().full(self.0.pop(m))
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.0.bits(&self.0.slices.prefixing_chain().prefix(x.clone()))
    }
}

impl<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecs<P>> Len for Autoregressive<P, S> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecs<P>> Autoregressive<P, S> {
    pub fn new(slice_codecs: S) -> Self {
        let len = slice_codecs.len();
        Autoregressive(PrefixAutoregressive::new(slice_codecs, len))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixAutoregressive<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecs<P>> {
    pub slices: S,
    pub len: usize,
    pub phantom: PhantomData<P>,
}

impl<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecs<P>> Codec for PrefixAutoregressive<P, S> {
    type Symbol = P::Prefix;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        let prefix = &mut x.clone();
        let slice_codec = &mut self.slices.apply(prefix);
        for _ in 0..self.len {
            let slice = self.slices.prefixing_chain().pop_slice(prefix);
            self.slices.update_after_pop_slice(slice_codec, prefix, &slice);
            slice_codec.push(m, &slice);
        }
        self.slices.empty_prefix().push(m, prefix);
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        let mut prefix = self.slices.empty_prefix().pop(m);
        let slice_codec = &mut self.slices.apply(&prefix);
        for _ in 0..self.len {
            let slice = slice_codec.pop(m);
            self.slices.prefixing_chain().push_slice(&mut prefix, &slice);
            self.slices.update_after_push_slice(slice_codec, &prefix, &slice);
        }
        prefix
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { None }
}

impl<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecs<P>> Len for PrefixAutoregressive<P, S> {
    fn len(&self) -> usize { self.len }
}

impl<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecs<P>> PrefixAutoregressive<P, S> {
    pub fn new(slices: S, len: usize) -> Self {
        PrefixAutoregressive { slices, len, phantom: PhantomData }
    }
}


#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SliceCodecAndPrefix<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecs<P>> {
    pub prefix: P::Prefix,
    pub slices: S,
    pub slice_codecs: S::Output,
}

impl<P: PrefixingChain<Prefix: Symbol> + Clone, S: SliceCodecs<P, Output: Clone> + Clone> Permutable for SliceCodecAndPrefix<P, S> {
    fn len(&self) -> usize {
        self.prefix.len()
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.prefix.swap(i, j);
        self.slices.swap(&mut self.slice_codecs, i, j);
    }
}

impl<P: PrefixingChain<Prefix: Symbol> + Clone, S: SliceCodecs<P, Output: Clone> + Clone> PrefixSliceCodec<P> for SliceCodecAndPrefix<P, S> {
    type SliceCodec = S::Output;

    fn slice_codec(&self) -> impl Deref<Target=Self::SliceCodec> {
        &self.slice_codecs
    }

    fn prefix(&self) -> impl Deref<Target=P::Prefix> {
        &self.prefix
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnfusedSliceCodecPrefixingChain<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecs<P>> {
    pub slices: S,
    pub phantom: PhantomData<P>,
}
impl<P: PrefixingChain<Prefix: Symbol> + Clone, S: SliceCodecs<P, Output: Clone> + Clone> PrefixingChain for UnfusedSliceCodecPrefixingChain<P, S> {
    type Prefix = SliceCodecAndPrefix<P, S>;
    type Full = P::Prefix;
    type Slice = P::Slice;

    fn pop_slice(&self, prefix: &mut Self::Prefix) -> Self::Slice {
        let slice = self.prefixing_chain().pop_slice(&mut prefix.prefix);
        self.slices.update_after_pop_slice(&mut prefix.slice_codecs, &prefix.prefix, &slice);
        slice
    }

    fn push_slice(&self, prefix: &mut Self::Prefix, slice: &Self::Slice) {
        self.prefixing_chain().push_slice(&mut prefix.prefix, slice);
        self.slices.update_after_push_slice(&mut prefix.slice_codecs, &prefix.prefix, slice);
    }

    fn prefix(&self, full: Self::Full) -> Self::Prefix {
        let slice_codecs = self.slices.apply(&full);
        SliceCodecAndPrefix { prefix: full, slices: self.slices.clone(), slice_codecs }
    }

    fn full(&self, prefix: Self::Prefix) -> Self::Full {
        prefix.prefix
    }
}

impl<P: PrefixingChain<Prefix: Symbol> + Clone, S: SliceCodecs<P, Output: Clone> + Clone> InnerSliceCodecs<P> for UnfusedSliceCodecPrefixingChain<P, S> {
    fn prefixing_chain(&self) -> P {
        self.slices.prefixing_chain()
    }

    fn empty_prefix(&self) -> impl Codec<Symbol=P::Prefix> {
        self.slices.empty_prefix()
    }
}

impl<P: PrefixingChain<Prefix: Symbol> + Clone, S: SliceCodecs<P, Output: Clone> + Clone> SliceCodecPrefixingChain<P> for UnfusedSliceCodecPrefixingChain<P, S> {}

impl<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecs<P> + Clone> Len for UnfusedSliceCodecPrefixingChain<P, S> {
    fn len(&self) -> usize {
        self.slices.len()
    }
}

pub type UnfusedAutoregressiveShuffleCodec<P, S, O> = AutoregressiveShuffleCodec<P, UnfusedSliceCodecPrefixingChain<P, S>, O>;
impl<P, S, O> UnfusedAutoregressiveShuffleCodec<P, S, O>
where
    P: PrefixingChain<Prefix: Symbol> + Clone,
    S: SliceCodecs<P, Output: Clone> + Clone,
    O: OrbitCodecs<P>,
{
    #[allow(unused)]
    pub fn new(slice_codecs: S, orbit_codecs: O) -> Self {
        AutoregressiveShuffleCodec::fused(UnfusedSliceCodecPrefixingChain { slices: slice_codecs, phantom: PhantomData }, orbit_codecs)
    }
}
