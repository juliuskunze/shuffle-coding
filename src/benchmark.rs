use std::collections::HashMap;
use std::fmt::Debug;
use std::io::{stdout, Write};
use std::ops::Deref;

use clap::Parser;
use itertools::Itertools;
use petgraph::Undirected;

use crate::ans::{Bernoulli, Categorical, Codec, CodecTestResults, Message, Symbol, Uniform, UniformCodec, VecCodec};
use crate::datasets::{Dataset, SourceInfo};
use crate::graph::{AsEdges, EdgeType, Graph};
use crate::graph_ans::{EmptyCodec, erdos_renyi_indices, GraphIID, num_all_edge_indices, polya_urn};
use crate::labelled_graph::EdgeLabelledGraph;
use crate::param_ans::{CategoricalParamCodec, EmptyParamCodec, ErdosRenyiParamCodec, GraphDatasetParamCodec, ParametrizedIndependent, PolyaUrnParamCodec, UniformParamCodec, WrappedParametrizedIndependent};
use crate::permutable::{GroupPermutable, Permutable, Unordered};
use crate::shuffle_ans::{PermutationUniform, shuffle_codec, TestConfig};
use crate::shuffle_ans::coset_interleaved::coset_interleaved_shuffle_codec;

#[derive(Clone, Debug, Parser)]
pub struct Config {
    /// Ignore node and edge labels for all graph datasets, only use graph structure.
    #[arg(long)]
    pub plain: bool,

    /// Report the following graph dataset statistics: Permutation and symmetry bits, and time needed for retrieving automorphisms from nauty.
    #[clap(long)]
    pub symstats: bool,
    /// Report the following graph dataset statistics: graphs total_nodes total_edges selfloops edge_prob node_entropy node_labels edge_entropy edge_labels.
    #[arg(long)]
    pub stats: bool,

    /// Use Erdős–Rényi model with categorical node/edge labels.
    #[arg(long, default_value_t = false)]
    pub er: bool,
    /// Use Erdős–Rényi model with uniform node/edge labels.
    #[arg(long = "eru", default_value_t = false)]
    pub uniform_er: bool,
    /// Use Pólya Urn model with categorical node/edge labels.
    #[arg(long, default_value_t = false)]
    pub pu: bool,
    /// Use Pólya Urn model with categorical node/edge labels, allowing redraws and self-loops.
    #[arg(long = "pur", default_value_t = false)]
    pub redraw_pu: bool,
    /// Do not test ordered codecs. These are tested by default.
    #[clap(short = 'o', long)]

    pub no_ordered: bool,
    /// Do not test codecs for only the parameters (no data). These are tested by default.
    #[clap(short = 'p', long)]
    pub no_param: bool,
    /// Do not test joint codecs for data and parameters. These are tested by default.
    #[clap(short, long)]
    pub no_parametrized: bool,
    /// Test codecs for only the data (no parameters). By default, these are not tested.
    #[clap(short, long)]
    pub unparametrized: bool,

    /// Shuffle codec test configuration.
    #[structopt(flatten)]
    pub shuffle: TestConfig,

    /// Add custom dataset source via '<name>:<index file URL>'. The defaults are
    /// TU:https://raw.githubusercontent.com/chrsmrrs/datasets/gh-pages/_docs/datasets.md and
    /// SZIP:https://raw.githubusercontent.com/juliuskunze/szip-graphs/main/datasets.md.
    /// Custom index files must be in the same format as the defaults, and refer to zip files of
    /// datasets that follow the TUDataset format described at https://chrsmrrs.github.io/datasets/docs/format/,
    /// except that no graph labels are required. Node/edge/graph attributes are ignored.
    #[arg(name = "source", long, value_parser = parse_index)]
    pub sources: Vec<SourceInfo>,
}

impl Deref for Config {
    type Target = TestConfig;
    fn deref(&self) -> &Self::Target { &self.shuffle }
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

    pub fn models(&self) -> Vec<&'static str> {
        let mut models = vec![];
        if self.er {
            models.push("ER");
        }
        if self.uniform_er {
            models.push("ERu");
        }
        if self.pu {
            models.push("PU");
        }
        if self.redraw_pu {
            models.push("PUr");
        }
        models
    }
}

fn parse_index(s: &str) -> Result<SourceInfo, String> {
    let (name, index_url) = s.splitn(2, ':').collect_tuple().expect("Expected '<name>:<index file URL>'.");
    Ok(SourceInfo::new(name, index_url, None))
}

pub fn print_symmetry(permutables: &Vec<impl GroupPermutable>) {
    let perm_bits = permutables.iter().map(|x| PermutationUniform { len: x.len() }.uni_bits()).sum::<f64>();
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

        let print_codec = |codec: &str| print!("{codec} {codec}_net {codec}_enc {codec}_dec ");
        let print_codec_p = |codec: &str| {
            if self.config.unparametrized {
                print_codec(&format!("n_{codec}"));
            }
            if !self.config.no_parametrized {
                print_codec(&format!("{codec}"));
            }
        };
        for codec in self.config.models() {
            if !self.config.no_param {
                print_codec(&format!("par_{codec}"));
            }

            if !self.config.no_ordered {
                print_codec_p(&format!("ord_{codec}"))
            }

            if self.config.shuffle.blockwise {
                print_codec_p(&format!("{codec}"));
            }
            if self.config.shuffle.coset_interleaved {
                print_codec_p(&format!("c_{codec}"));
            }
            if codec == "ER" {
                if self.config.shuffle.interleaved {
                    print_codec_p(&format!("i_{codec}"));
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

        let (node_entropy, node_labels) = if let Some(n) = &stats.node_label { (n.entropy(), n.masses.len()) } else { (0., 0) };
        let (edge_entropy, edge_labels) = if let Some(n) = &stats.edge_label { (n.entropy(), n.masses.len()) } else { (0., 0) };
        print_flush!("{} {} {} {} {} {} {} {} {} ", stats.num_graphs, stats.total_nodes, stats.total_edges, stats.loops, stats.edge.prob(), node_entropy, node_labels, edge_entropy, edge_labels);
    }

    pub fn run(&self) {
        print_flush!("{} {} {} ", self.dataset.source.name, self.dataset.category[..3].to_owned(), self.dataset.name);

        if self.config.plain || !(self.dataset.has_node_labels || self.dataset.has_edge_labels) {
            print_flush!("none ");
            let graphs = self.sorted(self.dataset.unlabelled_graphs());
            self.verify_and_print_stats(&DatasetStats::unlabelled(&graphs));
            let er = || self.run_model(ParametrizedIndependent {
                param_codec: GraphDatasetParamCodec { node: EmptyParamCodec::default(), edge: EmptyParamCodec::default(), indices: self.er_indices() },
                infer: |graphs| {
                    let DatasetStats { loops, edge, .. } = DatasetStats::unlabelled(&graphs);
                    VecCodec::new(graphs.iter().map(|x| {
                        GraphIID::new(x.len(), erdos_renyi_indices(x.len(), edge.clone(), loops), EmptyCodec::default(), EmptyCodec::default())
                    }).collect_vec())
                },
            }, &graphs);
            if self.config.uniform_er { er() }
            if self.config.er { er() }
            for original in self.config.pu_models_original() {
                self.run_model(ParametrizedIndependent {
                    param_codec: GraphDatasetParamCodec { node: EmptyParamCodec::default(), edge: EmptyParamCodec::default(), indices: self.pu_indices(original) },
                    infer: |graphs| {
                        let DatasetStats { loops, .. } = DatasetStats::unlabelled(&graphs);
                        VecCodec::new(graphs.iter().map(|x| {
                            let edge_indices = polya_urn(x.len(), x.num_edges(), loops || original, original);
                            GraphIID::new(x.len(), edge_indices, EmptyCodec::default(), EmptyCodec::default())
                        }).collect_vec())
                    },
                }, &graphs);
            }
        } else if let Some(graphs) = self.dataset.edge_labelled_graphs() {
            print_flush!("edges ");
            let graphs = self.sorted(graphs.clone());
            self.verify_and_print_stats(&DatasetStats::edge_labelled(&graphs));

            if self.config.er {
                self.run_model(ParametrizedIndependent {
                    param_codec: GraphDatasetParamCodec { node: CategoricalParamCodec, edge: CategoricalParamCodec, indices: self.er_indices() },
                    infer: |graphs| {
                        let DatasetStats { loops, edge, node_label, edge_label, .. } = DatasetStats::edge_labelled(&graphs);
                        VecCodec::new(graphs.iter().map(|x| GraphIID::new(
                            x.len(), erdos_renyi_indices(x.len(), edge.clone(), loops), node_label.clone().unwrap(), edge_label.clone().unwrap())).collect_vec())
                    },
                }, &graphs);
            }
            if self.config.uniform_er {
                self.run_model(ParametrizedIndependent {
                    param_codec: GraphDatasetParamCodec { node: UniformParamCodec::default(), edge: UniformParamCodec::default(), indices: self.er_indices() },
                    infer: |graphs| {
                        let DatasetStats { loops, edge, node_label, edge_label, .. } = DatasetStats::edge_labelled(&graphs);
                        let size = node_label.unwrap().masses.len();
                        let node_label = Uniform::new(size);
                        let size = edge_label.unwrap().masses.len();
                        let edge_label = Uniform::new(size);
                        VecCodec::new(graphs.iter().map(|x| GraphIID::new(
                            x.len(), erdos_renyi_indices(x.len(), edge.clone(), loops), node_label.clone(), edge_label.clone())).collect_vec())
                    },
                }, &graphs);
            }
            for original in self.config.pu_models_original() {
                self.run_model(ParametrizedIndependent {
                    param_codec: GraphDatasetParamCodec { node: CategoricalParamCodec, edge: CategoricalParamCodec, indices: self.pu_indices(original) },
                    infer: |graphs| {
                        let DatasetStats { loops, node_label, edge_label, .. } = DatasetStats::edge_labelled(&graphs);
                        VecCodec::new(graphs.iter().map(|x| {
                            let edge_indices = polya_urn(x.len(), x.num_edges(), loops || original, original);
                            GraphIID::new(x.len(), edge_indices, node_label.clone().unwrap(), edge_label.clone().unwrap())
                        }).collect_vec())
                    },
                }, &graphs);
            }
        } else if let Some(graphs) = self.dataset.node_labelled_graphs() {
            print_flush!("nodes ");
            let graphs = self.sorted(graphs.clone());
            self.verify_and_print_stats(&DatasetStats::node_labelled(&graphs));
            if self.config.er {
                self.run_model(ParametrizedIndependent {
                    param_codec: GraphDatasetParamCodec { node: CategoricalParamCodec, edge: EmptyParamCodec::default(), indices: self.er_indices() },
                    infer: |graphs| {
                        let DatasetStats { loops, edge, node_label, .. } = DatasetStats::node_labelled(&graphs);
                        VecCodec::new(graphs.iter().map(|x| {
                            let edge_indices = erdos_renyi_indices(x.len(), edge.clone(), loops);
                            GraphIID::new(x.len(), edge_indices, node_label.clone().unwrap(), EmptyCodec::default())
                        }).collect_vec())
                    },
                }, &graphs);
            }
            if self.config.uniform_er {
                self.run_model(ParametrizedIndependent {
                    param_codec: GraphDatasetParamCodec { node: UniformParamCodec::default(), edge: EmptyParamCodec::default(), indices: self.er_indices() },
                    infer: |graphs| {
                        let DatasetStats { loops, edge, node_label, .. } = DatasetStats::node_labelled(&graphs);
                        let size = node_label.unwrap().masses.len();
                        let node_label = Uniform::new(size);
                        VecCodec::new(graphs.iter().map(|x| {
                            let edge_indices = erdos_renyi_indices(x.len(), edge.clone(), loops);
                            GraphIID::new(x.len(), edge_indices, node_label.clone(), EmptyCodec::default())
                        }).collect_vec())
                    },
                }, &graphs);
            }
            for original in self.config.pu_models_original() {
                self.run_model(ParametrizedIndependent {
                    param_codec: GraphDatasetParamCodec { node: CategoricalParamCodec, edge: EmptyParamCodec::default(), indices: self.pu_indices(original) },
                    infer: |graphs| {
                        let DatasetStats { loops, node_label, .. } = DatasetStats::node_labelled(&graphs);
                        VecCodec::new(graphs.iter().map(|x| {
                            let edge_indices = polya_urn(x.len(), x.num_edges(), loops || original, original);
                            GraphIID::new(x.len(), edge_indices, node_label.clone().unwrap(), EmptyCodec::default())
                        }).collect_vec())
                    },
                }, &graphs);
            }
        }
        println!()
    }

    fn run_model<C: Codec<Symbol=impl GroupPermutable> + Symbol>(
        &self,
        codec: ParametrizedIndependent<VecCodec<C>, impl Codec<Symbol=VecCodec<C>>, impl Fn(&Vec<C::Symbol>) -> VecCodec<C> + Clone>,
        graphs: &Vec<C::Symbol>) {
        let param = (codec.infer)(&graphs);
        if !self.config.no_param {
            test_and_print(&codec.param_codec, &param, &self.config.initial_message());
        }
        if !self.config.no_ordered {
            if !self.config.no_parametrized {
                test_and_print(&codec, &graphs, &self.config.initial_message());
            }
            if self.config.unparametrized {
                test_and_print(&param, &graphs, &self.config.initial_message());
            }
        }
        if !self.config.blockwise && !self.config.coset_interleaved {
            return;
        }

        let unordered = graphs.iter().cloned().map(Unordered).collect_vec();
        let to_ordered = Box::new(|x: Vec<Unordered<_>>| x.into_iter().map(|x| x.into_ordered()).collect());
        let from_ordered = Box::new(|x: Vec<_>| x.iter().cloned().map(Unordered).collect());
        if self.config.blockwise {
            let codec = WrappedParametrizedIndependent {
                parametrized_codec: codec.clone(),
                data_to_inner: to_ordered.clone(),
                data_from_inner: from_ordered.clone(),
                from_inner_codec: Box::new(|x| VecCodec::new(x.codecs.iter().cloned().map(shuffle_codec))),
            };
            if self.config.unparametrized {
                test_and_print(&(codec.from_inner_codec)(param.clone()), &unordered, &self.config.initial_message());
            }
            if !self.config.no_parametrized {
                test_and_print(&codec, &unordered, &self.config.initial_message());
            }
        }
        if self.config.coset_interleaved {
            let codec = WrappedParametrizedIndependent {
                parametrized_codec: codec,
                data_to_inner: to_ordered,
                data_from_inner: from_ordered,
                from_inner_codec: Box::new(|x| VecCodec::new(x.codecs.iter().cloned().map(coset_interleaved_shuffle_codec))),
            };
            if self.config.unparametrized {
                test_and_print(&(codec.from_inner_codec)(param), &unordered, &self.config.initial_message());
            }
            if !self.config.no_parametrized {
                test_and_print(&codec, &unordered, &self.config.initial_message());
            }
        }
    }

    fn sorted<N: Symbol, E: Symbol, Ty: EdgeType>(&self, mut graphs: Vec<Graph<N, E, Ty>>) -> Vec<Graph<N, E, Ty>>
        where Graph<N, E, Ty>: GroupPermutable + AsEdges<N, E> {
        graphs.sort_by_key(|x| -(x.len() as isize));
        if self.config.symstats {
            print_symmetry(&graphs);
        }
        graphs
    }
}

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
    pub fn unlabelled<N: Symbol, E: Symbol, Ty: EdgeType>(graphs: &Vec<Graph<N, E, Ty>>) -> Self where Graph<N, E, Ty>: AsEdges<N, E> {
        let num_graphs = graphs.len();
        let loops = graphs.iter().any(|x| x.has_selfloops());
        let total_nodes = graphs.iter().map(|x| x.len()).sum::<usize>();
        let total_edges = graphs.iter().map(|x| x.num_edges()).sum::<usize>();
        let total_possible_edges = graphs.iter().map(|x| num_all_edge_indices::<Ty>(x.len(), loops)).sum::<usize>();
        let edge = Bernoulli::new(total_edges, total_possible_edges);
        Self { num_graphs, total_nodes, total_edges, loops, edge, node_label: None, edge_label: None }
    }

    pub fn node_labelled<E: Symbol, Ty: EdgeType>(graphs: &Vec<Graph<usize, E, Ty>>) -> Self where Graph<usize, E, Ty>: AsEdges<usize, E> {
        Self { node_label: Some(Self::node_label_dist(graphs)), ..Self::unlabelled(graphs) }
    }

    pub fn edge_labelled<Ty: EdgeType>(graphs: &Vec<EdgeLabelledGraph<Ty>>) -> Self where EdgeLabelledGraph<Ty>: AsEdges<usize, usize> {
        Self { edge_label: Some(Self::edge_label_dist(graphs)), ..Self::node_labelled(graphs) }
    }

    fn edge_label_dist<Ty: EdgeType>(graphs: &Vec<EdgeLabelledGraph<Ty>>) -> Categorical where EdgeLabelledGraph<Ty>: AsEdges<usize, usize> {
        Self::dist(&graphs.iter().flat_map(|x| x.edge_labels()).counts())
    }

    fn node_label_dist<E: Symbol, Ty: EdgeType>(graphs: &Vec<Graph<usize, E, Ty>>) -> Categorical where Graph<usize, E, Ty>: AsEdges<usize, E> {
        Self::dist(&graphs.iter().flat_map(|x| x.node_labels()).counts())
    }

    fn dist(counts: &HashMap<usize, usize>) -> Categorical {
        let masses = (0..counts.keys().max().unwrap() + 1).map(|x| counts.get(&x).cloned().unwrap_or_default()).collect_vec();
        Categorical::new(masses)
    }

    fn avg_nodes(&self) -> f64 {
        self.total_nodes as f64 / self.num_graphs as f64
    }

    fn avg_edges(&self) -> f64 {
        self.total_edges as f64 / self.num_graphs as f64
    }
}

pub fn test_and_print<S: Symbol>(codec: &impl Codec<Symbol=S>, symbol: &S, initial: &Message) -> CodecTestResults {
    let out = codec.test(symbol, initial);
    let CodecTestResults { bits, amortized_bits, enc_sec, dec_sec } = &out;
    print_flush!("{bits} {amortized_bits} {enc_sec} {dec_sec} ");
    out
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use crate::benchmark::{Benchmark, Config};
    use crate::datasets::dataset;

    #[test]
    fn test() {
        Benchmark {
            datasets: vec!(dataset("MUTAG")),
            config: Config::parse_from(["", "--stats", "--er"]),
        }.timed_run();
    }
}