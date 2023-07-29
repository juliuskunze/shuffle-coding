//! Chunked autoregressive shuffle coding, a variant of autoregressive shuffle coding.

use crate::autoregressive::graph::{GraphPrefix, GraphPrefixingChain};
use crate::autoregressive::prefix_orbit::FixPrefixOrbitCodec;
use crate::autoregressive::{AutoregressiveShuffleCodec, InnerSliceCodecs, PrefixAutoregressive, PrefixFn, PrefixSliceCodec, PrefixingChain, SliceCodecPrefixingChain, SliceCodecs, UncachedPrefixFn};
use crate::codec::{Codec, Message, OrdSymbol, Symbol};
use crate::graph::{ColorRefinement, EdgeType};
use crate::permutable::{Hashing, Len, Permutable, PermutableCodec};
use itertools::Itertools;
use std::cell::RefCell;
use std::ops::{Deref, Range};

/// Prefix type forming the basis for chunked shuffle coding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkedPrefix<Prefix: Permutable> {
    pub inner: Prefix,
    pub len: usize,
    /// Index of the last chunk included by the prefix.
    pub chunk_index: usize,
}

/// Would be Vec<P::Slice>, but our fused implementation does not require storing slices,
/// so we keep it empty for performance.
pub type Chunk = ();

impl<Prefix: Permutable> ChunkedPrefix<Prefix> {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkedPrefixingChain<P: PrefixingChain> {
    pub inner: P,
    /// Prefix sizes at chunk boundaries.
    /// The first element is 0 and the last the full object's length.
    pub prefix_sizes: Vec<usize>,
}

impl<P: Permutable> Permutable for ChunkedPrefix<P> {
    fn len(&self) -> usize {
        self.len
    }

    fn swap(&mut self, i: usize, j: usize) {
        assert!(i < self.len());
        assert!(j < self.len());
        self.inner.swap(i, j)
    }
}

impl<P: PrefixingChain> PrefixingChain for ChunkedPrefixingChain<P> {
    type Prefix = ChunkedPrefix<P::Prefix>;
    type Full = P::Full;
    type Slice = Chunk;

    fn pop_slice(&self, _prefix: &mut Self::Prefix) -> Self::Slice {
        unimplemented!("Unused, fused into ChunkedSliceCodecPrefixingChain::push_slice_from_prefix.");
    }

    fn push_slice(&self, _prefix: &mut Self::Prefix, _chunk: &Self::Slice) {
        unimplemented!("Unused, fused into ChunkedSliceCodecPrefixingChain::pop_slice_to_prefix.");
    }

    fn prefix(&self, full: Self::Full) -> Self::Prefix {
        let len = full.len();
        assert_eq!(len, self.prefix_sizes.last().copied().unwrap_or(0));
        let chunk_index = self.prefix_sizes.len().saturating_sub(1);
        ChunkedPrefix { inner: self.inner.prefix(full), len, chunk_index }
    }

    fn full(&self, prefix: Self::Prefix) -> Self::Full {
        self.inner.full(prefix.inner)
    }
}

impl<P: PrefixingChain> ChunkedPrefixingChain<P> {
    pub fn new(inner: P, chunk_sizes: Vec<usize>) -> Self {
        assert!(chunk_sizes.iter().all(|&size| size > 0));
        let prefix_sizes = chunk_sizes.iter().scan(0, |acc, &size| {
            *acc += size;
            Some(*acc)
        }).collect_vec();
        Self { inner, prefix_sizes }
    }

    /// The range of the chunk coded (if any) at the slice at `prefix.len - previous as usize`.
    fn chunk_range(&self, prefix: &ChunkedPrefix<P::Prefix>, previous: bool) -> Option<Range<usize>> {
        if self.has_chunk(prefix, previous) {
            let index = prefix.chunk_index - previous as usize;
            Some(self.prefix_sizes[index]..self.prefix_sizes[index + 1])
        } else {
            None
        }
    }

    /// Returns whether a chunk is coded at the slice at `prefix.len - previous as usize`.
    fn has_chunk(&self, prefix: &ChunkedPrefix<P::Prefix>, previous: bool) -> bool {
        if previous {
            // The first slice is always empty (never a chunk boundary). In its place, the empty 
            // prefix is used, allowing for more optimized codecs for the first prefix/chunk:
            prefix.chunk_index > 0 && prefix.len - 1 == self.prefix_sizes[prefix.chunk_index - 1]
        } else {
            prefix.len == self.prefix_sizes[prefix.chunk_index]
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkAutoregressive<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecs<P>> {
    pub chain: ChunkedPrefixingChain<P>,
    pub slices: S,
    /// RefCell to avoid cloning on every push/pop by allowing for mutable borrows:
    pub prefix: RefCell<ChunkedPrefix<P::Prefix>>,
    pub slice_codec: RefCell<S::Output>,
}

impl<P: PrefixingChain<Prefix: Symbol, Slice: Symbol> + Clone, S: SliceCodecs<P, Output: Clone> + Clone> PrefixSliceCodec<ChunkedPrefixingChain<P>> for ChunkAutoregressive<P, S> {
    type SliceCodec = ChunkAutoregressive<P, S>;

    fn slice_codec(&self) -> impl Deref<Target=Self::SliceCodec> {
        self
    }

    fn prefix(&self) -> impl Deref<Target=ChunkedPrefix<P::Prefix>> {
        self.prefix.borrow()
    }
}

impl<P: PrefixingChain<Prefix: Symbol, Slice: Symbol> + Clone, S: SliceCodecs<P, Output: Clone> + Clone> Permutable for ChunkAutoregressive<P, S> {
    fn len(&self) -> usize {
        self.prefix.borrow().len
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.prefix.borrow_mut().swap(i, j);
        self.slices.swap(&mut self.slice_codec.borrow_mut(), i, j)
    }
}

impl<P: PrefixingChain<Prefix: Symbol, Slice: Symbol>, S: SliceCodecs<P>> Codec for ChunkAutoregressive<P, S> {
    type Symbol = Chunk;

    fn push(&self, _m: &mut Message, _chunk: &Self::Symbol) {
        unimplemented!("Unused, fused into ChunkedSliceCodecPrefixingChain::push_slice_from_prefix.");
    }

    fn pop(&self, _m: &mut Message) -> Self::Symbol {
        unimplemented!("Unused, fused into ChunkedSliceCodecPrefixingChain::pop_slice_to_prefix.");
    }

    fn bits(&self, _: &Self::Symbol) -> Option<f64> { None }
}

impl<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecs<P>> ChunkAutoregressive<P, S> {
    pub fn new(chain: ChunkedPrefixingChain<P>, slices: S, prefix: ChunkedPrefix<P::Prefix>, first_slice_codec: S::Output) -> Self {
        Self { chain, slices, prefix: RefCell::new(prefix), slice_codec: RefCell::new(first_slice_codec) }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkedSliceCodecPrefixingChain<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecs<P>> {
    pub prefixing_chain: ChunkedPrefixingChain<P>,
    pub empty: EmptyChunkedPrefixCodec<PrefixAutoregressive<P, S>>,
    pub slices: S,
}

impl<P: PrefixingChain<Prefix: Symbol, Slice: Symbol> + Clone, S: SliceCodecs<P, Output: Clone> + Clone> PrefixingChain for ChunkedSliceCodecPrefixingChain<P, S> {
    type Prefix = ChunkAutoregressive<P, S>;
    type Full = ChunkedPrefix<P::Prefix>;
    type Slice = Chunk;

    fn pop_slice(&self, _prefix: &mut Self::Prefix) -> Self::Slice {
        unimplemented!("Unused, fused into Self::push_slice_from_prefix.");
    }

    fn push_slice(&self, _prefix: &mut Self::Prefix, _chunk: &Self::Slice) {
        unimplemented!("Unused, fused into Self::pop_slice_to_prefix.");
    }

    fn prefix(&self, full: Self::Full) -> Self::Prefix {
        let first_slice_codec = self.slices.apply(&full.inner);
        ChunkAutoregressive::new(self.prefixing_chain.clone(), self.slices.clone(), full, first_slice_codec)
    }

    fn full(&self, prefix: Self::Prefix) -> Self::Full {
        prefix.prefix.into_inner()
    }
}

impl<P: PrefixingChain<Prefix: Symbol, Slice: Symbol> + Clone, S: SliceCodecs<P, Output: Clone> + Clone> SliceCodecPrefixingChain<ChunkedPrefixingChain<P>> for ChunkedSliceCodecPrefixingChain<P, S> {
    fn push_slice_from_prefix(&self, m: &mut Message, prefix: &mut Self::Prefix) {
        let p = &mut prefix.prefix.borrow_mut();
        if let Some(range) = prefix.chain.chunk_range(p, true) {
            p.chunk_index -= 1;
            let slice_codec = &mut prefix.slice_codec.borrow_mut();
            for _ in range {
                let slice = prefix.slices.prefixing_chain().pop_slice(&mut p.inner);
                prefix.slices.update_after_pop_slice(slice_codec, &mut p.inner, &slice);
                slice_codec.push(m, &slice);
            }
        }
        p.len -= 1;
    }

    fn pop_slice_to_prefix(&self, m: &mut Message, prefix: &mut Self::Prefix) {
        let p = &mut prefix.prefix.borrow_mut();
        if let Some(range) = prefix.chain.chunk_range(p, false) {
            p.chunk_index += 1;
            let slice_codec = &mut prefix.slice_codec.borrow_mut();
            for _ in range {
                let slice = slice_codec.pop(m);
                prefix.slices.prefixing_chain().push_slice(&mut p.inner, &slice);
                prefix.slices.update_after_push_slice(slice_codec, &mut p.inner, &slice);
            }
        }
        p.len += 1;
    }
}

impl<P: PrefixingChain<Prefix: Symbol, Slice: Symbol> + Clone, S: SliceCodecs<P, Output: Clone> + Clone> InnerSliceCodecs<ChunkedPrefixingChain<P>> for ChunkedSliceCodecPrefixingChain<P, S> {
    #[inline]
    fn prefixing_chain(&self) -> ChunkedPrefixingChain<P> {
        // TODO remove clone:
        self.prefixing_chain.clone()
    }

    #[inline]
    fn empty_prefix(&self) -> impl Codec<Symbol=ChunkedPrefix<P::Prefix>> {
        // TODO remove clone:
        self.empty.clone()
    }
}

impl<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecs<P>> Len for ChunkedSliceCodecPrefixingChain<P, S> {
    fn len(&self) -> usize {
        self.slices.len()
    }
}

impl<P: PrefixingChain<Prefix: Symbol>, S: SliceCodecs<P> + Clone> ChunkedSliceCodecPrefixingChain<P, S> {
    pub fn new(prefixing_chain: ChunkedPrefixingChain<P>, slices: S) -> Self {
        let first_prefix = PrefixAutoregressive::new(slices.clone(), prefixing_chain.prefix_sizes.first().copied().unwrap_or(0));
        let empty = EmptyChunkedPrefixCodec { first_prefix };
        Self { prefixing_chain, empty, slices }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EmptyChunkedPrefixCodec<F: PermutableCodec<Symbol: Permutable>> {
    pub first_prefix: F,
}

impl<F: PermutableCodec<Symbol: Permutable>> Codec for EmptyChunkedPrefixCodec<F> {
    type Symbol = ChunkedPrefix<F::Symbol>;

    fn push(&self, m: &mut Message, x: &Self::Symbol) {
        self.first_prefix.push(m, &x.inner)
    }

    fn pop(&self, m: &mut Message) -> Self::Symbol {
        ChunkedPrefix { inner: self.first_prefix.pop(m), len: 0, chunk_index: 0 }
    }

    fn bits(&self, x: &Self::Symbol) -> Option<f64> {
        self.first_prefix.bits(&x.inner)
    }
}

impl<N: OrdSymbol + Default, E: OrdSymbol, Ty: EdgeType> UncachedPrefixFn<ChunkedPrefixingChain<GraphPrefixingChain<N, E, Ty>>> for ColorRefinement {
    type Output = FixPrefixOrbitCodec;

    fn apply(&self, x: &ChunkedPrefix<GraphPrefix<N, E, Ty>>) -> Self::Output {
        let mut prefix = GraphPrefixingChain::new().prefix(x.inner.graph.clone());
        prefix.len = x.len;
        let ids = self.convolutions(&prefix.graph, &|i| prefix.node_label_or_index(i));
        FixPrefixOrbitCodec::new(ids, x.len())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkedIncompleteOrbitCodecs<P: PrefixingChain, H: Hashing<P::Prefix, Hash=Vec<Hash>>, Hash: OrdSymbol + Default> {
    prefixing_chain: ChunkedPrefixingChain<P>,
    hashing: H,
    len: usize,
}

impl<P: PrefixingChain, H: Hashing<P::Prefix, Hash=Vec<Hash>>, Hash: OrdSymbol + Default> PrefixFn<ChunkedPrefixingChain<P>> for ChunkedIncompleteOrbitCodecs<P, H, Hash> {
    type Output = FixPrefixOrbitCodec;

    fn apply(&self, x: &ChunkedPrefix<P::Prefix>) -> Self::Output {
        let hash = self.hashing.apply(&x.inner);
        let hash_len = hash.len();
        assert_eq!(hash_len, self.prefixing_chain.prefix_sizes.get(x.chunk_index).copied().unwrap_or(0));
        FixPrefixOrbitCodec::new(hash, x.len)
    }

    fn update_after_pop_slice(&self, image: &mut Self::Output, x: &ChunkedPrefix<P::Prefix>, _: &()) {
        if self.prefixing_chain.has_chunk(x, false) {
            *image = self.apply(x);
        } else {
            image.pop_id();
            // TODO debug_assert_eq!(image, &self.apply(x));
        }
    }

    fn update_after_push_slice(&self, image: &mut Self::Output, x: &ChunkedPrefix<P::Prefix>, _: &()) {
        if self.prefixing_chain.has_chunk(x, true) {
            *image = self.apply(x);
        } else {
            image.push_id();
            // TODO debug_assert_eq!(image, &self.apply(x));
        }
    }

    fn swap(&self, image: &mut Self::Output, i: usize, j: usize) {
        image.swap(i, j)
    }
}

impl<P: PrefixingChain, H: Hashing<P::Prefix, Hash=Vec<Hash>>, Hash: OrdSymbol + Default> Len for ChunkedIncompleteOrbitCodecs<P, H, Hash> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<P: PrefixingChain, H: Hashing<P::Prefix, Hash=Vec<Hash>>, Hash: OrdSymbol + Default> ChunkedIncompleteOrbitCodecs<P, H, Hash> {
    pub fn new(prefixing_chain: ChunkedPrefixingChain<P>, hashing: H, len: usize) -> Self {
        Self { hashing, len, prefixing_chain }
    }
}

pub type ChunkedHashingShuffleCodec<P, S, H, Hash> = AutoregressiveShuffleCodec<ChunkedPrefixingChain<P>, ChunkedSliceCodecPrefixingChain<P, S>, ChunkedIncompleteOrbitCodecs<P, H, Hash>>;

pub fn chunked_hashing_shuffle_codec<P, S, H, Hash>(prefixing_chain: P, slices: S, hashing: H, chunk_sizes: Vec<usize>, len: usize) -> ChunkedHashingShuffleCodec<P, S, H, Hash>
where
    P: PrefixingChain<Prefix: Symbol, Slice: Symbol> + Clone,
    S: SliceCodecs<P, Output: Clone> + Clone,
    H: Hashing<P::Prefix, Hash=Vec<Hash>>,
    Hash: OrdSymbol + Default,
{
    assert_eq!(chunk_sizes.iter().sum::<usize>(), len);
    let chain = ChunkedPrefixingChain::new(prefixing_chain, chunk_sizes);
    let slices = ChunkedSliceCodecPrefixingChain::new(chain.clone(), slices);
    let orbits = ChunkedIncompleteOrbitCodecs::new(chain, hashing, len);
    AutoregressiveShuffleCodec::fused(slices, orbits)
}

pub fn uniform_chunk_sizes(num_chunks: usize, len: usize) -> Vec<usize> {
    geom_chunk_sizes(num_chunks, len, 1.)
}

pub fn geom_chunk_sizes(num_chunks: usize, len: usize, base: f64) -> Vec<usize> {
    let sizes = (0..num_chunks.min(len)).map(|i| base.powi(i as i32)).collect_vec();
    let scale = len as f64 / sizes.iter().sum::<f64>();
    let mut sizes = sizes.iter().map(|s| 1.max((*s * scale).ceil() as usize)).collect_vec();
    let mut extra = sizes.iter().sum::<usize>() - len;
    while extra != 0 {
        for i in (0..sizes.len()).rev() {
            if extra == 0 {
                break;
            }
            if sizes[i] > 1 {
                extra -= 1;
                sizes[i] -= 1;
            }
        }
    }
    sizes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_chunk_sizes() {
        assert_eq!(uniform_chunk_sizes(3, 0), vec![]);
        assert_eq!(uniform_chunk_sizes(3, 1), vec![1]);
        assert_eq!(uniform_chunk_sizes(3, 2), vec![1, 1]);
        assert_eq!(uniform_chunk_sizes(3, 3), vec![1, 1, 1]);
        assert_eq!(uniform_chunk_sizes(3, 4), vec![2, 1, 1]);
        assert_eq!(uniform_chunk_sizes(3, 5), vec![2, 2, 1]);
        assert_eq!(uniform_chunk_sizes(3, 6), vec![2, 2, 2]);
        assert_eq!(uniform_chunk_sizes(3, 7), vec![3, 2, 2]);
    }

    #[test]
    fn test_geom_chunk_sizes() {
        assert_eq!(geom_chunk_sizes(3, 0, 0.5), vec![]);
        assert_eq!(geom_chunk_sizes(3, 1, 0.5), vec![1]);
        assert_eq!(geom_chunk_sizes(3, 2, 0.5), vec![1, 1]);
        assert_eq!(geom_chunk_sizes(3, 3, 0.5), vec![1, 1, 1]);
        assert_eq!(geom_chunk_sizes(3, 4, 0.5), vec![2, 1, 1]);
        assert_eq!(geom_chunk_sizes(3, 5, 0.5), vec![3, 1, 1]);
        assert_eq!(geom_chunk_sizes(3, 6, 0.5), vec![4, 1, 1]);
        assert_eq!(geom_chunk_sizes(3, 7, 0.5), vec![4, 2, 1]);
        assert_eq!(geom_chunk_sizes(3, 8, 0.5), vec![5, 2, 1]);
        assert_eq!(geom_chunk_sizes(3, 9, 0.5), vec![6, 2, 1]);
        assert_eq!(geom_chunk_sizes(3, 10, 0.5), vec![6, 3, 1]);
        assert_eq!(geom_chunk_sizes(3, 11, 0.5), vec![7, 3, 1]);
        assert_eq!(geom_chunk_sizes(3, 12, 0.5), vec![7, 4, 1]);
        assert_eq!(geom_chunk_sizes(3, 13, 0.5), vec![8, 4, 1]);
        assert_eq!(geom_chunk_sizes(3, 14, 0.5), vec![8, 4, 2]);

        assert_eq!(geom_chunk_sizes(3, 13, 1. / 3.), vec![9, 3, 1]);
    }
}
