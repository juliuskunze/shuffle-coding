use crate::autoregressive::chunked::{chunked_hashing_shuffle_codec, uniform_chunk_sizes};
use crate::autoregressive::graph::{GraphPrefix, GraphPrefixingChain, PolyaUrnSliceCodecs};
use crate::autoregressive::{Autoregressive, AutoregressiveShuffleCodec, SliceCodecs};
use crate::bench::datasets::{Dataset, SourceInfo};
use crate::bench::param_codec::{AutoregressiveErdosReyniGraphDatasetParamCodec, AutoregressivePolyaUrnGraphDatasetParamCodec, CategoricalParamCodec, EmptyParamCodec, ErdosRenyiParamCodec, GraphDatasetParamCodec, ParametrizedIndependent, PolyaUrnParamCodec, UniformParamCodec, WrappedParametrizedIndependent};
use crate::codec::{Bernoulli, Categorical, Codec, CodecTestResults, EqSymbol,
                   Independent, OrdSymbol, Uniform, UniformCodec};
use crate::experimental::autoregressive::joint::JointSliceCodecs;
use crate::experimental::complete_joint::aut::{joint_shuffle_codec, with_isomorphism_test_max_len, AutCanonizable};
use crate::experimental::complete_joint::graph::EdgeLabelledGraph;
use crate::experimental::complete_joint::interleaved::interleaved_coset_shuffle_codec;
use crate::experimental::graph::{erdos_renyi_indices, num_all_edge_indices, polya_urn, EmptyCodec, ErdosRenyiSliceCodecs, GraphIID};
use crate::graph::{ColorRefinement, EdgeType, Graph, UnGraph, Undirected};
use crate::joint::incomplete::cr_joint_shuffle_codec;
use crate::permutable::{Complete, Perm, Permutable, PermutableCodec, PermutationUniform, Unordered};
use clap::Parser;
use itertools::Itertools;
use rayon::prelude::ParallelSliceMut;
use rayon::ThreadPoolBuilder;
use std::collections::HashMap;
use std::io::{stdout, Write};
use std::marker::PhantomData;
use std::ops::{Deref, Range};
use timeit::timeit_loops;

#[macro_use]
pub mod datasets;
pub mod param_codec;

#[derive(Clone, Debug, Parser, PartialEq)]
pub struct Config {
    /// Ignore node and edge labels for all graph datasets, only use graph structure.
    #[arg(long)]
    pub nolabels: bool,

    /// Report the following graph dataset statistics: Permutation and symmetry bits, and time needed for retrieving automorphisms from nauty.
    #[clap(long)]
    pub symstats: bool,
    /// Report the following graph dataset statistics: graphs total_nodes total_edges selfloops edge_prob node_entropy node_labels edge_entropy edge_labels.
    #[arg(long)]
    pub stats: bool,

    /// Additionally test ordered codecs without shuffle coding.
    #[clap(short, long)]
    pub ordered: bool,
    /// Additionally test codecs for only the parameters (no data).
    #[clap(long)]
    pub param: bool,
    /// Do not test joint codecs for data and parameters. These are tested by default.
    #[clap(long)]
    pub no_parametrized: bool,
    /// Additionally test codecs for only the data (no parameters).
    #[clap(short, long)]
    pub unparametrized: bool,

    /// Shuffle coding test configuration.
    #[clap(flatten)]
    pub shuffle: TestConfig,

    /// Use Erdős–Rényi model with uniform node/edge labels.
    #[clap(long = "eru", default_value_t = false)]
    pub uniform_er: bool,
    /// Use Pólya urn model allowing redraws and self-loops.
    #[clap(long = "pur", default_value_t = false)]
    pub redraw_pu: bool,

    /// Add custom dataset source via '<name>:<index file URL>'. The defaults are
    /// TU:https://raw.githubusercontent.com/chrsmrrs/datasets/gh-pages/_docs/datasets.md,
    /// SZIP:https://raw.githubusercontent.com/juliuskunze/szip-graphs/main/datasets.md, and
    /// REC:https://raw.githubusercontent.com/juliuskunze/rec-graphs/main/datasets.md
    /// Custom index files must be in the same format as the defaults, and refer to zip files of
    /// datasets that follow the TUDataset format described at https://chrsmrrs.github.io/datasets/docs/format/,
    /// except that no graph labels are required. Node/edge/graph attributes are ignored.
    #[clap(name = "source", long, value_parser = parse_index)]
    pub sources: Vec<SourceInfo>,

    #[clap(short, long)]
    /// Number of threads to use for parallelism. By default, the number of logical CPUs is used.
    pub threads: Option<usize>,
}

impl Deref for Config {
    type Target = TestConfig;
    fn deref(&self) -> &Self::Target { &self.shuffle }
}

pub struct Model {
    pub name: &'static str,
    pub is_autoregressive: bool,
}

impl Config {
    pub fn pu_models_original(&self) -> Vec<bool> {
        let mut original = vec![];
        if self.pu {
            original.push(false);
        }
        if self.redraw_pu {
            original.push(true);
        }
        original
    }

    pub fn models(&self) -> Vec<Model> {
        let mut models = vec![];
        if self.uniform_er {
            models.push(Model { name: "ERu", is_autoregressive: false });
        }
        if self.er {
            models.push(Model { name: "ER", is_autoregressive: false });
        }
        if self.pu {
            models.push(Model { name: "PU", is_autoregressive: false });
        }
        if self.redraw_pu {
            models.push(Model { name: "PUr", is_autoregressive: false });
        }
        if self.ae {
            models.push(Model { name: "AE", is_autoregressive: true });
        }
        if self.ap {
            models.push(Model { name: "AP", is_autoregressive: true });
        }
        models
    }
}

fn parse_index(s: &str) -> Result<SourceInfo, String> {
    let (name, index_url) = s.splitn(2, ':').collect_tuple().expect("Expected '<name>:<index file URL>'.");
    Ok(SourceInfo::new(name, index_url))
}

pub fn print_symmetry(permutables: &Vec<impl AutCanonizable>) {
    let perm_bits = permutables.iter().map(|x| PermutationUniform::<Perm>::new(x.len()).uni_bits()).sum::<f64>();
    let mut symmetry_bits = 0.;
    let symmetry_sec = timeit_loops!(1, { symmetry_bits = permutables.iter().map(|x| x.automorphisms().bits).sum::<f64>(); });
    print_flush!("{perm_bits} {symmetry_bits} {symmetry_sec} ");
}

pub struct Benchmark {
    pub datasets: Vec<Dataset>,
    pub config: Config,
}

impl Benchmark {
    pub fn timed_run(&self) {
        let time = timeit_loops!(1, { self.run(); });
        println!("Finished in {time:.1}s.");
    }

    pub fn run(&self) {
        if let Some(threads) = self.config.threads {
            ThreadPoolBuilder::new().num_threads(threads).
                build_global().unwrap();
        }
        self.print_headers();
        for dataset in &self.datasets {
            DatasetBenchmark { dataset, config: &self.config }.run();
        }
    }

    fn print_headers(&self) {
        print!("src ctg dataset labels ");

        if self.config.symstats {
            print!("permutation symmetry sym_sec ");
        }
        if self.config.stats {
            print!("graphs total_nodes total_edges selfloops edge_prob node_entropy node_labels edge_entropy edge_labels ")
        }

        let print_codec = |codec: &str| {
            print!("{codec} net enc dec ");
            if self.config.seeds > 1 {
                print!("std stdnet stdenc stddec ")
            }
        };
        let cr = format!("cr{}", self.config.convs);

        for Model { name, is_autoregressive } in self.config.models() {
            if self.config.param {
                print_codec(&format!("par_{name}"));
            }

            let print_codec_p = |prefix: &str| {
                if self.config.unparametrized {
                    print_codec(&format!("n_{prefix}_{name}"));
                }
                if !self.config.no_parametrized {
                    print_codec(&format!("{prefix}_{name}"));
                }
            };

            let print_complete_cr = |mode: &str| {
                if self.config.complete {
                    print_codec_p(&format!("c{mode}"));
                }
                if !self.config.no_incomplete {
                    print_codec_p(&format!("{cr}{mode}"));
                }
            };

            if self.config.ordered {
                print_codec_p("ord")
            }
            if self.config.shuffle.joint {
                print_complete_cr("j");
            }
            if self.config.shuffle.interleaved_coset_joint {
                print_codec_p("ij");
            }
            if self.config.joint_ar {
                print_complete_cr("ja");
            }
            if is_autoregressive {
                if self.config.full_ar {
                    print_complete_cr("fa");
                }
                if self.config.ar {
                    print_complete_cr(&format!("a{}", self.config.chunks));
                }
            }
        }
        println!();
    }
}

pub struct DatasetBenchmark<'a> {
    pub dataset: &'a Dataset,
    pub config: &'a Config,
}

impl DatasetBenchmark<'_> {
    fn er_indices(&self) -> impl Fn(Vec<usize>, bool) -> ErdosRenyiParamCodec<Undirected> + '_ + Clone {
        |nums_nodes, loops| ErdosRenyiParamCodec::new(nums_nodes, loops)
    }
    fn pu_indices(&self, original: bool) -> impl Fn(Vec<usize>, bool) -> PolyaUrnParamCodec<Undirected> + '_ + Clone {
        move |nums_nodes, loops| PolyaUrnParamCodec::new(nums_nodes, loops || original, original)
    }

    fn verify_and_print_stats(&self, stats: &DatasetStats) {
        if !self.config.stats { return; }
        assert_eq!(self.dataset.num_graphs, stats.num_graphs);
        assert_eq!(format!("{:.2}", self.dataset.avg_nodes), format!("{:.2}", stats.avg_nodes()));
        assert_eq!(format!("{:.2}", self.dataset.avg_edges), format!("{:.2}", stats.avg_edges()));

        let (node_entropy, node_labels) = if let Some(n) = &stats.node_label { (n.entropy(), n.len()) } else { (0., 0) };
        let (edge_entropy, edge_labels) = if let Some(n) = &stats.edge_label { (n.entropy(), n.len()) } else { (0., 0) };
        print_flush!("{} {} {} {} {} {} {} {} {} ", stats.num_graphs, stats.total_nodes, stats.total_edges, stats.loops, stats.edge.prob(), node_entropy, node_labels, edge_entropy, edge_labels);
    }

    pub fn run(&self) {
        print_flush!("{} {} {} ", self.dataset.source.name, self.dataset.category[..3].to_owned(), self.dataset.name);

        if self.config.nolabels || !(self.dataset.has_node_labels || self.dataset.has_edge_labels) {
            print_flush!("none ");
            let graphs = self.sorted(self.dataset.unlabelled_graphs());
            self.verify_and_print_stats(&DatasetStats::unlabelled(&graphs));

            for uniform_er_only in [true, false] {
                self.run_models(&graphs, EmptyParamCodec::default(), EmptyParamCodec::default(), |graphs| {
                    let DatasetStats { loops, edge, .. } = DatasetStats::unlabelled(&graphs);
                    (loops, edge, EmptyCodec::default(), EmptyCodec::default())
                }, uniform_er_only);
            }
        } else if let Some(graphs) = self.dataset.edge_labelled_graphs() {
            print_flush!("edges ");
            let graphs = self.sorted(graphs.clone());
            self.verify_and_print_stats(&DatasetStats::edge_labelled(&graphs));

            self.run_models(&graphs, UniformParamCodec::default(), UniformParamCodec::default(), |graphs| {
                let DatasetStats { loops, edge, node_label, edge_label, .. } = DatasetStats::edge_labelled(&graphs);
                let node_label = Uniform::new(node_label.unwrap().len());
                let edge_label = Uniform::new(edge_label.unwrap().len());
                (loops, edge, node_label, edge_label)
            }, true);

            self.run_models(&graphs, CategoricalParamCodec, CategoricalParamCodec, |graphs| {
                let DatasetStats { loops, edge, node_label, edge_label, .. } = DatasetStats::edge_labelled(&graphs);
                (loops, edge, node_label.unwrap(), edge_label.unwrap())
            }, false);
        } else if let Some(graphs) = self.dataset.node_labelled_graphs() {
            print_flush!("nodes ");
            let graphs = self.sorted(graphs.clone());
            self.verify_and_print_stats(&DatasetStats::node_labelled(&graphs));

            self.run_models(&graphs, UniformParamCodec::default(), EmptyParamCodec::default(), |graphs| {
                let DatasetStats { loops, edge, node_label, .. } = DatasetStats::node_labelled(&graphs);
                let node_label = Uniform::new(node_label.unwrap().len());
                (loops, edge, node_label, EmptyCodec::default())
            }, true);

            self.run_models(&graphs, CategoricalParamCodec, EmptyParamCodec::default(), |graphs| {
                let DatasetStats { loops, edge, node_label, .. } = DatasetStats::node_labelled(&graphs);
                (loops, edge, node_label.unwrap(), EmptyCodec::default())
            }, false);
        }
        println!()
    }

    fn run_models<
        N: OrdSymbol + Default,
        E: OrdSymbol,
        NodeParamC: Codec<Symbol: Codec<Symbol=N> + Eq> + Clone,
        EdgeParamC: Codec<Symbol: Codec<Symbol=E> + Eq> + Clone
    >(&self,
      graphs: &Vec<UnGraph<N, E>>,
      node: NodeParamC, edge: EdgeParamC,
      infer: impl Fn(&Vec<UnGraph<N, E>>) -> (bool, Bernoulli, NodeParamC::Symbol, EdgeParamC::Symbol) + Clone,
      uniform_er_only: bool)
    where
        UnGraph<N, E>: AutCanonizable,
        GraphPrefix<N, E, Undirected>: AutCanonizable,
    {
        if if uniform_er_only { self.config.uniform_er } else { self.config.er } {
            self.run_joint_model(ParametrizedIndependent {
                param_codec: GraphDatasetParamCodec { node: node.clone(), edge: edge.clone(), indices: self.er_indices() },
                infer: |graphs| {
                    let (loops, has_edge, node, edge) = infer.clone()(&graphs);
                    Independent::new(graphs.iter().map(|x| GraphIID::new(
                        x.len(), erdos_renyi_indices(x.len(), has_edge.clone(), loops), node.clone(), edge.clone())).collect_vec())
                },
            }, graphs);
        }
        if uniform_er_only {
            return;
        }
        for original in self.config.pu_models_original() {
            self.run_joint_model(ParametrizedIndependent {
                param_codec: GraphDatasetParamCodec { node: node.clone(), edge: edge.clone(), indices: self.pu_indices(original) },
                infer: |graphs| {
                    let (loops, _, node, edge) = infer(&graphs);
                    Independent::new(graphs.iter().map(|x| {
                        let edge_indices = polya_urn(x.len(), x.num_edges(), loops || original, original);
                        GraphIID::new(x.len(), edge_indices, node.clone(), edge.clone())
                    }).collect_vec())
                },
            }, graphs);
        }

        if self.config.ae {
            self.run_autoregressive_model(ParametrizedIndependent {
                param_codec: AutoregressiveErdosReyniGraphDatasetParamCodec { joint: GraphDatasetParamCodec { node: node.clone(), edge: edge.clone(), indices: self.er_indices() } },
                infer: |graphs| {
                    let (loops, has_edge, node, edge) = infer(&graphs);
                    Independent::new(graphs.iter().map(|x| {
                        Autoregressive::new(ErdosRenyiSliceCodecs {
                            len: x.len(),
                            has_edge: has_edge.clone(),
                            node: node.clone(),
                            edge: edge.clone(),
                            loops,
                            phantom: PhantomData,
                        })
                    }).collect_vec())
                },
            }, graphs)
        }

        if self.config.ap {
            self.run_autoregressive_model(ParametrizedIndependent {
                param_codec: AutoregressivePolyaUrnGraphDatasetParamCodec { node: node.clone(), edge: edge.clone(), phantom: PhantomData },
                infer: |graphs| {
                    let (loops, _, node, edge) = infer(&graphs);
                    Independent::new(graphs.iter().map(|x| {
                        Autoregressive::new(PolyaUrnSliceCodecs {
                            len: x.len(),
                            node: node.clone(),
                            edge: edge.clone(),
                            loops,
                            phantom: PhantomData,
                        })
                    }).collect_vec())
                },
            }, graphs)
        }
    }

    fn run_autoregressive_model<N: OrdSymbol + Default, E: OrdSymbol, Ty: EdgeType, S: SliceCodecs<GraphPrefixingChain<N, E, Ty>, Output: Clone> + EqSymbol>(
        &self,
        codec: ParametrizedIndependent<impl Codec<Symbol=Independent<Autoregressive<GraphPrefixingChain<N, E, Ty>, S>>> + Clone, impl Fn(&Vec<Graph<N, E, Ty>>) -> Independent<Autoregressive<GraphPrefixingChain<N, E, Ty>, S>> + Clone>,
        graphs: &Vec<Graph<N, E, Ty>>)
    where
        Graph<N, E, Ty>: AutCanonizable,
        GraphPrefix<N, E, Ty>: AutCanonizable,
    {
        with_isomorphism_test_max_len(self.config.isomorphism_test_max_len, || {
            self.run_joint_model(codec.clone(), graphs);

            let param = (codec.infer)(&graphs);
            let unordered = graphs.iter().cloned().map(Unordered).collect_vec();
            let to_ordered = Box::new(|x: Vec<Unordered<_>>| x.into_iter().map(|x| x.into_ordered()).collect());

            if self.config.full_ar {
                if self.config.complete {
                    let codec = WrappedParametrizedIndependent {
                        parametrized_codec: codec.clone(),
                        data_to_inner: to_ordered.clone(),
                        codec_from_inner: Box::new(move |x| Independent::new(x.codecs.iter().cloned().map(|c| AutoregressiveShuffleCodec::new(c.0.slices, Complete)))),
                    };
                    if self.config.unparametrized {
                        test_and_print(&(codec.codec_from_inner)(param.clone()), &unordered, self.config.seeds());
                    }
                    if !self.config.no_parametrized {
                        test_and_print(&codec, &unordered, self.config.seeds());
                    }
                }
                if !self.config.no_incomplete {
                    let cr = self.config.color_refinement_max_local_info();
                    let codec = WrappedParametrizedIndependent {
                        parametrized_codec: codec.clone(),
                        data_to_inner: to_ordered.clone(),
                        codec_from_inner: Box::new(move |x| Independent::new(x.codecs.iter().cloned().map(|c| AutoregressiveShuffleCodec::new(c.0.slices, cr.clone())))),
                    };
                    if self.config.unparametrized {
                        test_and_print(&(codec.codec_from_inner)(param.clone()), &unordered, self.config.seeds());
                    }
                    if !self.config.no_parametrized {
                        test_and_print(&codec, &unordered, self.config.seeds());
                    }
                }
            } else if self.config.ar {
                if self.config.complete {
                    let chunks = self.config.chunks;
                    let codec = WrappedParametrizedIndependent {
                        parametrized_codec: codec.clone(),
                        data_to_inner: to_ordered.clone(),
                        codec_from_inner: Box::new(move |x| Independent::new(x.codecs.iter().cloned().map(|c| {
                            let chunk_sizes = uniform_chunk_sizes(chunks.clone(), c.0.len);
                            chunked_hashing_shuffle_codec(GraphPrefixingChain::new(), c.0.slices, Complete, chunk_sizes, c.0.len)
                        }))),
                    };
                    if self.config.unparametrized {
                        test_and_print(&(codec.codec_from_inner)(param.clone()), &unordered, self.config.seeds());
                    }
                    if !self.config.no_parametrized {
                        test_and_print(&codec, &unordered, self.config.seeds());
                    }
                }
                if !self.config.no_incomplete {
                    let cr = self.config.color_refinement();
                    let chunks = self.config.chunks;
                    let codec = WrappedParametrizedIndependent {
                        parametrized_codec: codec.clone(),
                        data_to_inner: to_ordered.clone(),
                        codec_from_inner: Box::new(move |x| Independent::new(x.codecs.iter().cloned().map(|c| {
                            let chunk_sizes = uniform_chunk_sizes(chunks.clone(), c.0.len);
                            chunked_hashing_shuffle_codec(GraphPrefixingChain::new(), c.0.slices, cr.clone(), chunk_sizes, c.0.len)
                        }))),
                    };
                    if self.config.unparametrized {
                        test_and_print(&(codec.codec_from_inner)(param.clone()), &unordered, self.config.seeds());
                    }
                    if !self.config.no_parametrized {
                        test_and_print(&codec, &unordered, self.config.seeds());
                    }
                }
            }
        })
    }

    fn run_joint_model<N: OrdSymbol + Default, E: OrdSymbol, Ty: EdgeType, C: PermutableCodec<Symbol=Graph<N, E, Ty>> + EqSymbol>(
        &self,
        codec: ParametrizedIndependent<impl Codec<Symbol=Independent<C>> + Clone, impl Fn(&Vec<C::Symbol>) -> Independent<C> + Clone>,
        graphs: &Vec<C::Symbol>)
    where
        Graph<N, E, Ty>: AutCanonizable,
        GraphPrefix<N, E, Ty>: AutCanonizable,
    {
        with_isomorphism_test_max_len(self.config.isomorphism_test_max_len, || {
            let param = (codec.infer)(&graphs);
            if self.config.param {
                test_and_print(&codec.param_codec, &param, self.config.seeds());
            }
            if self.config.ordered {
                if !self.config.no_parametrized {
                    test_shuffled_and_print(&codec, &graphs, self.config.seeds());
                }
                if self.config.unparametrized {
                    test_shuffled_and_print(&param, &graphs, self.config.seeds());
                }
            }

            let unordered = graphs.iter().cloned().map(Unordered).collect_vec();
            let to_ordered = Box::new(|x: Vec<Unordered<_>>| x.into_iter().map(|x| x.into_ordered()).collect());

            if self.config.joint {
                if self.config.complete {
                    let codec = WrappedParametrizedIndependent {
                        parametrized_codec: codec.clone(),
                        data_to_inner: to_ordered.clone(),
                        codec_from_inner: Box::new(|x| Independent::new(x.codecs.iter().cloned().map(joint_shuffle_codec))),
                    };
                    if self.config.unparametrized {
                        test_and_print(&(codec.codec_from_inner)(param.clone()), &unordered, self.config.seeds());
                    }
                    if !self.config.no_parametrized {
                        test_and_print(&codec, &unordered, self.config.seeds());
                    }
                }
                if !self.config.no_incomplete {
                    let cr = self.config.color_refinement();
                    let codec = WrappedParametrizedIndependent {
                        parametrized_codec: codec.clone(),
                        data_to_inner: to_ordered.clone(),
                        codec_from_inner: Box::new(move |x| Independent::new(x.codecs.iter().cloned().map(|c| cr_joint_shuffle_codec(c, cr.clone())))),
                    };
                    if self.config.unparametrized {
                        test_and_print(&(codec.codec_from_inner)(param.clone()), &unordered, self.config.seeds());
                    }
                    if !self.config.no_parametrized {
                        test_and_print(&codec, &unordered, self.config.seeds());
                    }
                }
            }
            if self.config.interleaved_coset_joint {
                let codec = WrappedParametrizedIndependent {
                    parametrized_codec: codec.clone(),
                    data_to_inner: to_ordered.clone(),
                    codec_from_inner: Box::new(|x| Independent::new(x.codecs.iter().cloned().map(interleaved_coset_shuffle_codec))),
                };
                if self.config.unparametrized {
                    test_and_print(&(codec.codec_from_inner)(param.clone()), &unordered, self.config.seeds());
                }
                if !self.config.no_parametrized {
                    test_and_print(&codec, &unordered, self.config.seeds());
                }
            }

            if self.config.joint_ar {
                if self.config.complete {
                    let codec = WrappedParametrizedIndependent {
                        parametrized_codec: codec.clone(),
                        data_to_inner: to_ordered.clone(),
                        codec_from_inner: Box::new(move |x| Independent::new(x.codecs.iter().cloned().map(|c| AutoregressiveShuffleCodec::new(JointSliceCodecs::new(c), Complete)))),
                    };
                    if self.config.unparametrized {
                        test_and_print(&(codec.codec_from_inner)(param.clone()), &unordered, self.config.seeds());
                    }
                    if !self.config.no_parametrized {
                        test_and_print(&codec, &unordered, self.config.seeds());
                    }
                }

                if !self.config.no_incomplete {
                    let cr = self.config.color_refinement_max_local_info();
                    let codec = WrappedParametrizedIndependent {
                        parametrized_codec: codec.clone(),
                        data_to_inner: to_ordered.clone(),
                        codec_from_inner: Box::new(move |x| Independent::new(x.codecs.iter().cloned().map(|c| AutoregressiveShuffleCodec::new(JointSliceCodecs::new(c), cr.clone())))),
                    };
                    if self.config.unparametrized {
                        test_and_print(&(codec.codec_from_inner)(param.clone()), &unordered, self.config.seeds());
                    }
                    if !self.config.no_parametrized {
                        test_and_print(&codec, &unordered, self.config.seeds());
                    }
                }
            }
        })
    }

    fn sorted<N: EqSymbol, E: EqSymbol, Ty: EdgeType>(&self, mut graphs: Vec<Graph<N, E, Ty>>) -> Vec<Graph<N, E, Ty>>
    where
        Graph<N, E, Ty>: AutCanonizable,
    {
        graphs.par_sort_unstable_by_key(|x| -(x.len() as isize));
        if self.config.symstats {
            print_symmetry(&graphs);
        }
        graphs
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DatasetStats {
    pub num_graphs: usize,
    pub total_nodes: usize,
    pub total_edges: usize,
    pub loops: bool,
    pub edge: Bernoulli,
    pub node_label: Option<Categorical>,
    pub edge_label: Option<Categorical>,
}

impl DatasetStats {
    pub fn unlabelled<N: EqSymbol, E: EqSymbol, Ty: EdgeType>(graphs: &Vec<Graph<N, E, Ty>>) -> Self {
        let num_graphs = graphs.len();
        let loops = graphs.iter().any(|x| x.has_selfloops());
        let total_nodes = graphs.iter().map(|x| x.len()).sum::<usize>();
        let total_edges = graphs.iter().map(|x| x.num_edges()).sum::<usize>();
        let total_possible_edges = graphs.iter().map(|x| num_all_edge_indices::<Ty>(x.len(), loops)).sum::<usize>();
        let edge = Bernoulli::new(total_edges, total_possible_edges);
        Self { num_graphs, total_nodes, total_edges, loops, edge, node_label: None, edge_label: None }
    }

    pub fn node_labelled<E: EqSymbol, Ty: EdgeType>(graphs: &Vec<Graph<usize, E, Ty>>) -> Self {
        Self { node_label: Some(Self::node_label_dist(graphs)), ..Self::unlabelled(graphs) }
    }

    pub fn edge_labelled<Ty: EdgeType>(graphs: &Vec<EdgeLabelledGraph<Ty>>) -> Self {
        Self { edge_label: Some(Self::edge_label_dist(graphs)), ..Self::node_labelled(graphs) }
    }

    fn edge_label_dist<Ty: EdgeType>(graphs: &Vec<EdgeLabelledGraph<Ty>>) -> Categorical {
        Self::dist(&graphs.iter().flat_map(|x| x.edge_labels()).counts())
    }

    fn node_label_dist<E: EqSymbol, Ty: EdgeType>(graphs: &Vec<Graph<usize, E, Ty>>) -> Categorical {
        Self::dist(&graphs.iter().flat_map(|x| x.node_labels()).counts())
    }

    fn dist(counts: &HashMap<usize, usize>) -> Categorical {
        let masses = (0..counts.keys().max().unwrap() + 1).map(|x| counts.get(&x).cloned().unwrap_or_default()).collect_vec();
        Categorical::from_iter(masses)
    }

    fn avg_nodes(&self) -> f64 {
        self.total_nodes as f64 / self.num_graphs as f64
    }

    fn avg_edges(&self) -> f64 {
        self.total_edges as f64 / self.num_graphs as f64
    }
}

/// Shuffles each symbol before testing, which matters for testing stochastic ordered codecs. 
pub fn test_shuffled_and_print<P: Eq + Permutable>(codec: &impl Codec<Symbol=Vec<P>>, symbol: &Vec<P>, seeds: Range<usize>) -> f64 {
    print_results(seeds.map(|seed| codec.test(&symbol.iter().map(|g| g.shuffled(seed)).collect_vec(), seed)).collect_vec())
}

pub fn test_and_print<S: EqSymbol>(codec: &impl Codec<Symbol=S>, symbol: &S, seeds: Range<usize>) -> f64 {
    print_results(seeds.map(|seed| codec.test(symbol, seed)).collect_vec())
}

pub fn print_results(results: Vec<CodecTestResults>) -> f64 {
    if results.len() == 1 {
        let CodecTestResults { bits, amortized_bits, enc_sec, dec_sec } = results.first().unwrap();
        print_flush!("{bits} {amortized_bits} {enc_sec} {dec_sec} ");
        return *bits as f64;
    }

    let len = results.len() as f64;
    let bits = results.iter().map(|x| x.bits).sum::<usize>() as f64 / len;
    let amortized_bits = results.iter().map(|x| x.amortized_bits).sum::<f64>() / len;
    let enc_sec = results.iter().map(|x| x.enc_sec).sum::<f64>() / len;
    let dec_sec = results.iter().map(|x| x.dec_sec).sum::<f64>() / len;
    let bit_std = (results.iter().map(|x| (x.bits as f64 - bits).powi(2)).sum::<f64>() / len).sqrt();
    let amortized_bit_std = (results.iter().map(|x| (x.amortized_bits - amortized_bits).powi(2)).sum::<f64>() / len).sqrt();
    let enc_sec_std = (results.iter().map(|x| (x.enc_sec - enc_sec).powi(2)).sum::<f64>() / len).sqrt();
    let dec_sec_std = (results.iter().map(|x| (x.dec_sec - dec_sec).powi(2)).sum::<f64>() / len).sqrt();
    print_flush!("{bits} {amortized_bits} {enc_sec} {dec_sec} {bit_std} {amortized_bit_std} {enc_sec_std} {dec_sec_std} ");
    bits
}

#[cfg(test)]
mod tests {
    use crate::bench::datasets::dataset;
    use crate::bench::{Benchmark, Config};
    use clap::Parser;

    #[test]
    fn test() {
        Benchmark {
            datasets: vec!(dataset("MUTAG")),
            config: Config::parse_from(["", "--stats", "--er"]),
        }.timed_run();
    }
}

#[derive(Clone, Debug, Parser, PartialEq)]
pub struct TestConfig {
    /// Run joint shuffle coding.
    #[clap(short, long)]
    pub joint: bool,

    /// Run chunked autoregressive shuffle coding.
    #[clap(short, long)]
    pub ar: bool,

    /// Number of color refinement convolutions used for incomplete shuffle coding.
    #[clap(long, default_value = "4")]
    pub convs: usize,

    /// Number of chunks used for chunked autoregressive shuffle coding.
    #[clap(long, default_value = "16")]
    pub chunks: usize,

    /// Experimental.
    /// Base for geometric series of chunk sizes used for chunked autoregressive shuffle coding.
    /// Sizes will be roughly proportional to base^index.
    /// The default of 1 leads to roughly uniform chunk sizes.
    /// Lower values lead to later chunks being exponentially smaller.
    #[clap(hide = true, long, default_value = "1.")]
    pub chunks_base: f64,

    /// Seed for the message tail random generator providing initial bits.
    #[clap(long, default_value = "0")]
    pub seed: usize,

    /// Number of seeds to test with.
    #[clap(long, default_value = "1")]
    pub seeds: usize,

    /// If some, verify permute group action and canon labelling axioms with the given seed.
    #[cfg(test)]
    #[clap(skip)]
    pub axioms_seed: Option<usize>,

    /// Maximum number of graph vertices to verify isomorphism of unordered objects after decoding,
    /// using nauty. By default, isomorphism is not verified. Can be very slow for large graphs.
    #[clap(short, long, default_value = "0")]
    pub isomorphism_test_max_len: usize,

    /// Use joint Erdős–Rényi model.
    #[clap(long)]
    pub er: bool,

    /// Use joint Pólya urn model.
    #[clap(long)]
    pub pu: bool,

    /// Use autoregressive Erdős–Rényi model.
    #[clap(long)]
    pub ae: bool,

    /// Use autoregressive model approximating Pólya urn.
    #[clap(long)]
    pub ap: bool,

    /// Experimental. Do not use incomplete variant of shuffle coding based on color refinement.
    #[clap(long)]
    pub no_incomplete: bool,

    /// Experimental. Use complete variant of shuffle coding based on nauty. Can be very slow.
    #[clap(short, long)]
    pub complete: bool,

    /// Experimental.
    /// Run joint shuffle coding with an interleaved coset codec based on nauty. Can be very slow.
    #[clap(hide = true, long)]
    pub interleaved_coset_joint: bool,

    /// Experimental. Run joint autoregressive shuffle coding. Can be very slow for more convs.
    #[clap(long)]
    pub joint_ar: bool,

    /// Experimental. Run full autoregressive shuffle coding. Can be very slow for more convs.
    #[clap(long)]
    pub full_ar: bool,
}

impl TestConfig {
    #[cfg(test)]
    pub const fn test(seed: usize) -> Self {
        Self {
            axioms_seed: Some(seed),
            joint: true,
            interleaved_coset_joint: true,
            full_ar: true,
            ar: true,
            joint_ar: true,
            complete: true,
            no_incomplete: false,
            er: true,
            pu: true,
            ae: true,
            ap: true,
            convs: 1,
            chunks: 3,
            chunks_base: 1.,
            seed,
            seeds: 1,
            isomorphism_test_max_len: usize::MAX,
        }
    }

    pub fn seeds(&self) -> Range<usize> {
        self.seed..self.seed + self.seeds
    }

    pub fn color_refinement(&self) -> ColorRefinement {
        ColorRefinement::new(self.convs, false)
    }

    pub fn color_refinement_max_local_info(&self) -> ColorRefinement {
        ColorRefinement::new(self.convs, true)
    }
}
