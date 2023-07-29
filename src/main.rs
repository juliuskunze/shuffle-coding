use clap::Parser;
use itertools::Itertools;
use shuffle_coding::bench::datasets::{datasets_from, download, DatasetInfo, SourceInfo};
use shuffle_coding::bench::{Benchmark, Config};

/// Benchmark for graph codecs and shuffle coding. By default,
/// TU (https://chrsmrrs.github.io/datasets/), SZIP (https://github.com/juliuskunze/szip-graphs) and 
/// REC (https://github.com/juliuskunze/rec-graphs) graph datasets are available.
#[derive(Clone, Debug, Parser, PartialEq)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Each argument can be a dataset, a category, an index name (like "TU" or "SZIP") or an
    /// end-inclusive range "[dataset]..[dataset]" where both lower and upper bound are optional
    /// (".." will select all datasets, for example).
    /// If multiple dataset specifiers are given, the benchmark will run on all of them in sequence.
    #[arg(required = true)]
    datasets: Vec<String>,

    #[structopt(flatten)]
    config: Config,
}

impl Args {
    fn benchmark(self) -> Benchmark {
        Benchmark { datasets: download(&self.datasets()), config: self.config }
    }

    fn datasets(&self) -> Vec<DatasetInfo> {
        let all_datasets = datasets_from(self.indices());
        self.datasets.iter().flat_map(|filter| Self::filtered(&all_datasets, filter)).collect()
    }

    fn indices(&self) -> Vec<SourceInfo> {
        SourceInfo::defaults().into_iter().chain(self.config.sources.clone()).collect()
    }

    fn filtered(infos: &Vec<DatasetInfo>, filter: &str) -> Vec<DatasetInfo> {
        let from_index = infos.iter().filter(|d| d.source.name == filter).cloned().collect_vec();
        if !from_index.is_empty() { return from_index; }

        let from_category = infos.iter().filter(|d| d.category.as_str() == filter).cloned().collect_vec();
        if !from_category.is_empty() { return from_category; }

        if let Some((start, end)) = filter.split("..").collect_tuple() {
            let start = if start == "" { 0 } else { infos.iter().position(|x| x.name == start).expect(&format!("Dataset '{start}' not found.")) };
            let end = if end == "" { infos.len() - 1 } else { infos.iter().position(|x| x.name == end).expect(&format!("Dataset '{end}' not found.")) };
            return infos[start..=end].into_iter().cloned().collect();
        }

        vec![infos.iter().find(|x| x.name == filter).expect(&format!("'{filter}' dataset/category/index not found.")).clone()]
    }
}

fn main() {
    Args::parse().benchmark().timed_run();
}
