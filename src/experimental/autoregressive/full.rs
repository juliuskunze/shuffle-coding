//! Incomplete full autoregressive shuffle coding on graphs based on cached color refinement.
use crate::autoregressive::graph::{GraphPrefix, GraphPrefixingChain};
use crate::autoregressive::multiset::DynPrefixOrbitCodec;
use crate::autoregressive::prefix_orbit::{OrbitsById, PrefixOrbitCodec};
use crate::autoregressive::{OrbitCodec, PrefixFn, PrefixingChain};
use crate::codec::{Codec, MutDistribution, OrdSymbol};
use crate::experimental::graph::GraphIID;
use crate::graph::{ColorRefinement, EdgeType, Graph};
use crate::permutable::graph::EdgeIndicesCodec;
use crate::permutable::{Len, Permutable, Unordered};
use itertools::{Either, Itertools};
use std::collections::BTreeMap;
use std::mem;

impl ColorRefinement {
    fn update_around_slice<N: OrdSymbol + Default, E: OrdSymbol, Ty: EdgeType>(&self, x: &GraphPrefix<N, E, Ty>, orbits: &mut DynPrefixOrbitCodec<Vec<usize>>, (_, Unordered(edges)): &<GraphPrefixingChain<N, E, Ty> as PrefixingChain>::Slice, index: usize) {
        assert_eq!(x.len, orbits.len);
        let mut neighbors = edges.iter().map(|((i, j), _)| {
            assert_eq!(*i, index);
            *j
        }).collect_vec();
        neighbors.push(index);
        self.update_around::<N, E, Ty, _>(&x.graph, orbits, neighbors);
    }

    pub fn update_around<N: OrdSymbol + Default, E: OrdSymbol, Ty: EdgeType, M: OrdSymbol>(&self, x: &Graph<M, E, Ty>, orbits: &mut DynPrefixOrbitCodec<Vec<usize>>, indices: Vec<usize>) {
        assert!(!Ty::is_directed());
        // Add elements at distance 0.
        let mut new_ids_by_distance = vec![BTreeMap::from_iter(indices.into_iter().map(|i| (i, orbits.id(i))))];

        // Add neighbors using breadth-first search at distances >0.
        for _ in 0..self.convs {
            let mut neighbors_ = BTreeMap::new();
            for p in new_ids_by_distance.last().unwrap().keys() {
                neighbors_.extend(x.nodes[*p].1.keys().cloned().
                    filter(|i| new_ids_by_distance.iter().all(|ids| !ids.contains_key(i))).
                    map(|i| (i, orbits.id(i))))
            }
            new_ids_by_distance.push(neighbors_);
        }

        // Calculate updated hashes at distance 0 from the node labels and optionally the multiset of neighboring edge labels.
        for (&i, new_id) in new_ids_by_distance[0].iter_mut() {
            new_id[0] = self.init_node(x, i, &|i| if i < orbits.len {
                Either::Left(&x.nodes[i].0)
            } else {
                Either::Right(i)
            });
        }

        // Calculate updated hashes at distances >0.
        for iter in 0..self.convs {
            let indices = new_ids_by_distance.iter().take(iter + 2).enumerate().flat_map(
                |(update_i, x)| x.keys().map(move |i| (update_i, *i))).collect_vec();
            for (update_i, i) in indices {
                let prev_hash = |i: usize| {
                    for new_id in new_ids_by_distance.iter() {
                        if let Some(h) = new_id.get(&i) {
                            return h.clone()[iter];
                        }
                    }
                    return orbits.id(i)[iter];
                };

                let self_hash = prev_hash(i).clone();
                let neighbor_hashes = x.nodes[i].1.iter().map(|(ne, e)| (prev_hash(*ne), e)).sorted_unstable().collect_vec();
                let new_hash = Self::get_hash((self_hash, neighbor_hashes));
                new_ids_by_distance[update_i].get_mut(&i).unwrap()[iter + 1] = new_hash;
            }
        }

        // Apply the updates.
        for (i, new_id) in new_ids_by_distance.into_iter().flat_map(|x| x) {
            orbits.update_id(i, new_id);
        }
    }
}

impl<O, D> PrefixOrbitCodec<O, D>
where
    O: OrbitsById + Clone,
    D: MutDistribution<Symbol=O::Id> + Clone,
{
    fn update_id(&mut self, element: usize, new_id: O::Id) -> O::Id {
        let old_id = mem::replace(&mut self.ids[element], new_id.clone());
        if self.len <= element {
            return old_id;
        }

        self.orbits.remove(&old_id, element);
        self.orbits.insert(new_id.clone(), element);

        self.categorical.remove(&old_id, 1);
        self.categorical.insert(new_id, 1);
        old_id
    }
}

impl<N: OrdSymbol + Default, E: OrdSymbol, Ty: EdgeType> PrefixFn<GraphPrefixingChain<N, E, Ty>> for ColorRefinement {
    type Output = DynPrefixOrbitCodec<Vec<usize>>;

    fn apply(&self, x: &GraphPrefix<N, E, Ty>) -> Self::Output {
        let mut hashes = self.init(&x.graph, &|i| x.node_label_or_index(i));
        let mut ids = hashes.iter().map(|h| vec![h.clone()]).collect_vec();
        for _ in 0..self.convs {
            hashes = Self::convolve(&x.graph, &hashes);

            for (id, hash) in ids.iter_mut().zip_eq(&hashes) {
                id.push(hash.clone());
            }
        }

        DynPrefixOrbitCodec::new(ids, x.len)
    }

    fn update_after_pop_slice(&self, orbits: &mut Self::Output, x: &GraphPrefix<N, E, Ty>, slice: &<GraphPrefixingChain<N, E, Ty> as PrefixingChain>::Slice) {
        if Ty::is_directed() {
            *orbits = PrefixFn::<GraphPrefixingChain<N, E, Ty>>::apply(self, x);
            return;
        }

        orbits.pop_id();
        assert_eq!(orbits.len, x.len);
        self.update_around_slice(x, orbits, &slice, x.len);
    }

    fn update_after_push_slice(&self, orbits: &mut Self::Output, x: &GraphPrefix<N, E, Ty>, slice: &<GraphPrefixingChain<N, E, Ty> as PrefixingChain>::Slice) {
        if Ty::is_directed() {
            *orbits = PrefixFn::<GraphPrefixingChain<N, E, Ty>>::apply(self, x);
            return;
        }

        orbits.push_id();
        self.update_around_slice(x, orbits, &slice, x.len - 1);
    }

    fn swap(&self, orbits: &mut Self::Output, i: usize, j: usize) {
        orbits.swap(i, j)
    }
}

impl<NodeC: Codec, EdgeC: Codec<Symbol: OrdSymbol>, IndicesC: EdgeIndicesCodec<Ty=Ty>, Ty: EdgeType> Len for GraphIID<NodeC, EdgeC, IndicesC> {
    fn len(&self) -> usize {
        self.nodes.len
    }
}

#[cfg(test)]
pub mod tests {
    use crate::autoregressive::multiset::DynPrefixOrbitCodec;
    use crate::bench::datasets::dataset;
    use crate::bench::TestConfig;
    use crate::codec::Codec;
    use crate::experimental::graph::tests::{plain_erdos_renyi, test_graph_shuffle_codecs};
    use crate::graph::{ColorRefinement, UnGraph, Undirected};
    use crate::joint::Canonizable;
    use crate::permutable::Permutable;
    use clap::Parser;
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use std::io;
    use std::io::Write;
    use timeit::timeit_loops;

    #[test]
    fn test_5() {
        for g in dataset("as").unlabelled_graphs() {
            for edge_label_init in [false, true] {
                let shuffled = g.shuffled(0);
                let hasher = ColorRefinement::new(5, edge_label_init);
                let sec = timeit_loops!(1, {assert!(
                        hasher.maybe_isomorphic(&g, &shuffled))});
                dbg!(sec);
                assert!(sec < 2.);
            }
        }
    }

    #[test]
    fn false_positive() {
        let cycle7 = UnGraph::plain(7, [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 4)]);
        let cycle4_cycle3 = UnGraph::plain(7, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0)]);
        assert!(!cycle7.is_isomorphic(&cycle4_cycle3));
        assert!(ColorRefinement::new(7, false).maybe_isomorphic(&cycle7, &cycle4_cycle3));
    }

    #[test]
    fn test_orbits() {
        let orbits = DynPrefixOrbitCodec::<usize>::new(
            vec![0, 4, 6, 1, 1, 3, 3, 4, 5, 3, 2134, 123, 1, 23, 123, 124, 123, 53, 456, 45, 674, 56, 34, 5, 324, 64,
                 567, 5, 8, 56, 63, 4, 3, 4, 5, 45, 4, 54, 5, 45, 4, 54, 54, 5, 45, 34, 23, 42, 3, 213, 42, 34, 23, 423, 42, 1, 3, 4], 27);
        orbits.test_on_samples(5000);
    }

    #[test]
    fn minor() {
        assert!(vec![0, 2] < vec![1]);
        assert!(vec![0, 2] < vec![1, 1]);
    }

    fn generate_erdos_renyi_graph(num_nodes: usize, num_edges: usize, seed: u64) -> UnGraph {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut graph = UnGraph::plain_empty(num_nodes);
        let density = num_edges as f64 / (num_nodes as f64 * (num_nodes - 1) as f64);
        assert!(density < 0.1);
        for i in 0..num_edges {
            let mut retry = 0;
            'retry: loop {
                let edge = (rng.gen_range(0..num_nodes), rng.gen_range(0..num_nodes));
                if edge.0 != edge.1 && graph.insert_plain_edge(edge) {
                    break 'retry;
                }

                retry += 1;
                assert!(retry < 10, "Failed to generate edge {i}");
            }
        }
        graph
    }

    #[test]
    #[ignore = "long-running"]
    fn large() {
        let config = TestConfig::parse_from(["", "--ar", "--ap"]);
        for num_nodes in [10_000, 100_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000] {
            let num_edges = num_nodes * 50;
            let c = plain_erdos_renyi::<Undirected>(num_nodes, 0.1, false);
            let graph = generate_erdos_renyi_graph(num_nodes, num_edges, 0);
            println!("{} {} ", graph.nodes.len(), graph.num_edges());
            io::stdout().flush().unwrap();
            test_graph_shuffle_codecs([(c, graph)], &config);
            println!();
        }
    }
}
