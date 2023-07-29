use crate::codec::{EqSymbol, OrdSymbol};
use crate::joint::Canonizable;
use crate::permutable::{Orbits, Perm, Permutable, Permutation, Unordered};
use itertools::Itertools;
use rayon::prelude::*;
use std::ops::Deref;

pub type Multiset<T> = Unordered<Vec<T>>;

impl<T: EqSymbol> Permutable for Vec<T> {
    fn len(&self) -> usize { self.len() }

    fn swap(&mut self, i: usize, j: usize) {
        self.as_mut_slice().swap(i, j);
    }

    fn permuted(&self, x: &impl Permutation) -> Self {
        let mut out = self.clone();
        for i in 0..self.len() {
            out[x.apply(i)] = self[i].clone();
        }
        out
    }
}

impl<T: OrdSymbol> Canonizable for Vec<T> {
    fn canon(&self) -> Perm {
        let mut out = (0..self.len()).collect_vec();
        out.par_sort_unstable_by_key(|i| &self[*i]);
        Perm::from(out).inverse()
    }

    fn canonized(&self) -> Self {
        let mut out = self.clone();
        out.par_sort_unstable();
        out
    }
}

impl<T: OrdSymbol> Orbits for Vec<T> {
    type Id = T;

    fn orbit_ids(&self) -> impl Deref<Target=Vec<<Self as Orbits>::Id>> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autoregressive::chunked::{geom_chunk_sizes, uniform_chunk_sizes};
    use crate::autoregressive::multiset::iid_multiset_shuffle_codec;
    use crate::bench::{test_and_print, TestConfig};
    use crate::codec::tests::test_and_print_vec;
    use crate::codec::{Categorical, Codec, Independent, Message, Uniform, IID, MAX_SIZE};
    use crate::experimental::autoregressive::joint::autoregressive_joint_multiset_shuffle_codec;
    use crate::experimental::graph::tests::with_sampled_symbols;
    use crate::joint::multiset::{chunked_iid_multiset_shuffle_codec, joint_multiset_shuffle_codec};
    use crate::permutable::{Len, Unordered};
    use std::fmt::Debug;
    use std::fs;
    use std::str::FromStr;

    impl<C: Codec> Len for IID<C> {
        fn len(&self) -> usize { self.len }
    }

    pub fn test_multiset_codecs(codecs_and_vecs: impl IntoIterator<Item=(IID<Uniform>, Vec<usize>)>, config: &TestConfig) {
        let (codecs, permutables): (Vec<IID<_>>, Vec<_>) = codecs_and_vecs.into_iter().unzip();
        let unordered = &permutables.iter().map(|x| Unordered(x.clone())).collect();
        test_and_print_vec(codecs.iter().map(iid_multiset_shuffle_codec), unordered, config.seeds());
        test_and_print_vec(codecs.iter().map(|c| chunked_iid_multiset_shuffle_codec(c.clone(), uniform_chunk_sizes(5, c.len))), unordered, config.seeds());
        test_and_print_vec(codecs.iter().map(|c| chunked_iid_multiset_shuffle_codec(c.clone(), geom_chunk_sizes(5, c.len, 2.))), unordered, config.seeds());
        test_and_print_vec(codecs.iter().cloned().map(joint_multiset_shuffle_codec), unordered, config.seeds());
        test_and_print_vec(codecs.iter().cloned().map(autoregressive_joint_multiset_shuffle_codec), unordered, config.seeds());
        println!();
    }

    #[test]
    fn multiset() {
        let v = vec![1, 1];
        assert_eq!(v.len(), 2);
        assert_eq!(&Perm::create_swap(2, 1, 0) * &v, v);
        assert!(v.canon().is_identity());
    }

    #[test]
    fn min_codecs_with_vecs() {
        let vecs = [vec![0, 1]];
        test_multiset_codecs(vecs.map(|v|
            (IID::new(Uniform::new(*v.iter().max().unwrap() + 1), v.len()), v)), &TestConfig::test(0));
    }

    #[test]
    fn codecs_with_vecs() {
        let vecs = [vec![1, 1], vec![1, 0, 4, 2, 2, 1, 1]];
        test_multiset_codecs(vecs.map(|v| {
            let size = *v.iter().max().unwrap() + 1;
            (IID::new(Uniform::new(size), v.len()), v)
        }), &TestConfig::test(0));
    }

    #[test]
    fn sampled_codecs_small() {
        let codecs = (0..20).map(|i| {
            let size = i + 1;
            IID::new(Uniform::new(size), i * 2)
        });
        test_multiset_codecs(with_sampled_symbols(codecs), &TestConfig::test(0));
    }

    #[test]
    fn autoregressive_sampled_codecs() {
        let (codecs, permutables): (Vec<_>, Vec<_>) = with_sampled_symbols((0..100).map(|i| {
            let size = i + 1;
            IID::new(Uniform::new(size), i * 2)
        })).into_iter().unzip();
        let fs_codec = Independent::new(codecs.iter().map(iid_multiset_shuffle_codec));
        let unordered = permutables.into_iter().map(|v| Unordered(v)).collect();
        let result = fs_codec.test(&unordered, 0);
        assert!(result.enc_sec < 5., "enc_sec = {}", result.enc_sec);
        assert!(result.dec_sec < 5., "dec_sec = {}", result.dec_sec);
    }

    #[test]
    fn slow_permutable_axioms() {
        let codecs = (0..10).map(|i| {
            let size = i + 1;
            IID::new(Uniform::new(size), i * 2)
        });
        for x in Independent::new(codecs).pop(&mut Message::zeros()) {
            x.test_canon(0);
        };
    }

    #[test]
    #[ignore = "long-running"]
    fn benchmark_multiset() {
        fn read<S: FromStr<Err: Debug>>(name: &str, size: usize) -> Vec<S> {
            let file_name = format!("multiset-data/{name}.txt");
            let out: Vec<S> = fs::read_to_string(file_name).unwrap().split(", ").map(|s| s.parse().unwrap()).collect();
            assert_eq!(out.len(), size);
            out
        }

        let probs = read::<f64>("probs", 1024);
        let lens = [1, 10, 100, 1_000, 10_000, 100_000];
        let seeds = |len| {
            print!("{} ", len);
            0..if len <= 100_000 { 100 } else { 10 }
        };
        let multisets_and_iid_codecs = lens.into_iter().map(|len| (
            Unordered(read::<usize>(&len.to_string(), len)),
            IID::new(Categorical::from_iter(probs.iter().map(|p| ((p * (MAX_SIZE as f64)) as usize).max(1)).collect_vec()), len))
        ).collect_vec();

        println!("ordered:");
        for (m, iid) in &multisets_and_iid_codecs {
            test_and_print(iid, m.to_ordered(), seeds(m.len()));
            println!();
        }

        println!("joint:");
        for (m, iid) in &multisets_and_iid_codecs {
            test_and_print(&joint_multiset_shuffle_codec(iid.clone()), m, seeds(m.len()));
            println!();
        }

        println!("ar_joint:");
        for (m, iid) in &multisets_and_iid_codecs {
            test_and_print(&autoregressive_joint_multiset_shuffle_codec(iid.clone()), m, seeds(m.len()));
            println!();
        }

        println!("ar_full:");
        for (m, iid) in &multisets_and_iid_codecs {
            test_and_print(&iid_multiset_shuffle_codec(iid), m, seeds(m.len()));
            println!();
        }

        let chunks = 10;
        for chunks in [1, chunks] {
            println!("ar_chunks={chunks}_uniform:");
            for (m, iid) in &multisets_and_iid_codecs {
                test_and_print(&chunked_iid_multiset_shuffle_codec(iid.clone(), uniform_chunk_sizes(chunks, iid.len)), m, seeds(m.len()));
                println!();
            }
        }

        println!("ar_chunks={chunks}_exp_optbase:");
        let optimal_bases = [1.00, 0.18, 0.49, 0.82, 0.91, 0.93, 0.93, 0.93];
        for ((m, iid), base) in multisets_and_iid_codecs.iter().zip(optimal_bases) {
            test_and_print(&chunked_iid_multiset_shuffle_codec(iid.clone(), geom_chunk_sizes(chunks, iid.len, base)), m, seeds(m.len()));
            println!();
        }

        let last_base = *optimal_bases.last().unwrap();
        println!("ar_chunks={chunks}_exp{last_base}:");
        for (m, iid) in &multisets_and_iid_codecs {
            test_and_print(&chunked_iid_multiset_shuffle_codec(iid.clone(), geom_chunk_sizes(chunks, iid.len, last_base)), m, seeds(m.len()));
            println!();
        }
    }
}
