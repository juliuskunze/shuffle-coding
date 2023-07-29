use crate::experimental::complete_joint::graph::{EdgeLabelledGraph, NodeLabelledGraph};
use crate::graph::{DiGraph, EdgeIndex, UnGraph, Undirected};
use crate::permutable::Permutable;
use crate::permutable::{FHashMap, FHashSet};
use downloader::{Download, Downloader, Verification};
use itertools::Itertools;
use std::fs;
use std::fs::File;
use std::io::{stdout, BufRead, BufReader, Write};
use std::iter::repeat_n;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::string::ToString;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use zip_extract;

macro_rules! print_flush {
    ( $($t:tt)* ) => {
        {
            print!($($t)*);
            stdout().flush().unwrap();
        }
    }
}

/// Refers to a datasets source index file similar to the [TUDatasets index](https://raw.githubusercontent.com/chrsmrrs/datasets/gh-pages/_docs/datasets.md).
/// The index file in turn must refer to zip archives of datasets following the [TUDatasets format specification](https://chrsmrrs.github.io/datasets/docs/format/),
/// except that graph labels are not required.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceInfo {
    pub name: String,
    pub index_url: String,
}

impl SourceInfo {
    pub fn new(name: &str, index_url: &str) -> Self {
        Self { name: name.to_string(), index_url: index_url.to_string() }
    }

    pub fn defaults() -> Vec<Self> {
        vec![
            Self::new(
                "TU",
                "https://raw.githubusercontent.com/chrsmrrs/datasets/gh-pages/_docs/datasets.md"),
            Self::new(
                "SZIP",
                "https://raw.githubusercontent.com/juliuskunze/szip-graphs/main/datasets.md"),
            Self::new(
                "REC",
                "https://raw.githubusercontent.com/juliuskunze/rec-graphs/main/datasets.md"),
        ]
    }
}


#[derive(Clone, Debug, PartialEq)]
pub struct DatasetInfo {
    pub name: String,
    pub zip_url: String,
    pub cache: PathBuf,
    pub source: SourceInfo,
    pub category: String,
    pub has_node_labels: bool,
    pub has_edge_labels: bool,
    pub has_temporal_node_labels: bool,
    pub num_graphs: usize,
    pub avg_nodes: f64,
    pub avg_edges: f64,
}

impl DatasetInfo {
    fn zip_download(&self) -> Option<Download> {
        if self.is_downloaded() {
            return None;
        }
        Some(Download::new(&self.zip_url).file_name(&self.zip_path()).verify(Arc::new(
            |zip, _| match Self::unzip(&zip) {
                Ok(_) => Verification::Ok,
                Err(_) => Verification::Failed,
            })))
    }

    fn unzip(zip: &PathBuf) -> Result<(), ()> {
        let dir = zip.parent().ok_or(())?;
        let dir = dir.join(zip.file_stem().ok_or(())?);
        let file = File::open(&zip).ok().ok_or(())?;
        zip_extract::extract(file, &dir, true).ok().ok_or(())?;
        fs::remove_file(&zip).ok().ok_or(())?;
        Ok(())
    }

    fn zip_path(&self) -> PathBuf {
        self.cache.parent().unwrap().join(format!("{}.zip", &self.name))
    }

    fn is_downloaded(&self) -> bool {
        self.cache.exists()
    }

    fn verify_download(&self) {
        assert!(self.is_downloaded(), "Dataset '{}' is not downloaded yet into {}.", self.name, self.cache.to_str().unwrap());
        assert!(self.edge_path().exists(), "Edge file {} is missing.", self.edge_path().to_str().unwrap());
        assert!(self.graph_path().exists(), "Graph indicator file {} is missing.", self.graph_path().to_str().unwrap());
        assert_eq!(self.has_node_labels, self.node_labels_path().exists(), "Node label path {} is missing or unexpected.", self.node_labels_path().to_str().unwrap());
        assert_eq!(self.has_edge_labels, self.edge_labels_path().exists(), "Edge label path {} is missing or unexpected.", self.edge_labels_path().to_str().unwrap());
    }

    fn edge_path(&self) -> PathBuf {
        self.cache.join(format!("{}_A.txt", self.name))
    }

    fn graph_path(&self) -> PathBuf {
        self.cache.join(format!("{}_graph_indicator.txt", self.name))
    }

    fn node_labels_path(&self) -> PathBuf {
        self.cache.join(format!("{}_node_labels.txt", self.name))
    }

    fn edge_labels_path(&self) -> PathBuf {
        self.cache.join(format!("{}_edge_labels.txt", self.name))
    }
}

/// A downloaded dataset.
#[derive(Clone, Debug, PartialEq)]
pub struct Dataset(DatasetInfo);

impl Deref for Dataset {
    type Target = DatasetInfo;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl Dataset {
    pub fn new(info: DatasetInfo) -> Self {
        info.verify_download();
        Self(info)
    }

    pub fn unlabelled_graphs(&self) -> Vec<UnGraph> {
        self.graphs_with_edge_order().into_iter().map(|(g, _)| g).collect_vec()
    }

    pub fn node_labelled_graphs(&self) -> Option<Vec<NodeLabelledGraph<Undirected>>> {
        self.node_labelled_graphs_with_edge_order().map(|x| x.into_iter().map(|(g, _)| g).collect_vec())
    }

    pub fn edge_labelled_graphs(&self) -> Option<Vec<EdgeLabelledGraph<Undirected>>> {
        if !self.has_edge_labels {
            return None;
        }
        let graphs_with_edge_order = if self.has_node_labels {
            self.node_labelled_graphs_with_edge_order().unwrap()
        } else {
            self.graphs_with_edge_order().into_iter().map(|(x, es)| (
                NodeLabelledGraph::new(repeat_n(0, x.len()), x.edges()), es)).collect()
        };

        let all_edge_labels = self.read_edge_labels().collect_vec();
        let mut start = 0;
        let mut out = vec![];
        for (x, edge_indices) in graphs_with_edge_order {
            let num_edges = edge_indices.len();
            let edge_labels = &all_edge_labels[start..start + num_edges];
            start += num_edges;
            let edges = edge_indices.into_iter().zip_eq(edge_labels.into_iter().cloned());
            out.push(EdgeLabelledGraph::new(x.node_labels(), edges).into_undirected());
        }
        assert_eq!(start, all_edge_labels.len());
        Some(out)
    }

    fn read_indices(path: &Path) -> impl Iterator<Item=Vec<usize>> {
        lines(path).map(|l| l.split(",").map(|i| i.trim().parse::<usize>().unwrap()).collect())
    }

    fn read_1indices(path: &Path) -> impl Iterator<Item=Vec<usize>> {
        Self::read_indices(path).map(|x| x.into_iter().map(|x| x - 1).collect())
    }

    fn graphs_with_edge_order(&self) -> Vec<(UnGraph, Vec<EdgeIndex>)> {
        let edges: Vec<EdgeIndex> = Self::read_1indices(&self.edge_path()).into_iter().map(
            |x| x.into_iter().collect_tuple().unwrap()
        ).collect();
        let graphs_by_node: Vec<usize> = Self::read_1indices(&self.graph_path()).map(|x| x[0]).collect();
        let nodes_by_graph: FHashMap<_, _> = FHashMap::from_iter(
            graphs_by_node.iter().enumerate().map(|(i, g)| (*g, i)).into_group_map().into_iter());
        let mut edges_by_graph: FHashMap<usize, Vec<_>> = FHashMap::from_iter(nodes_by_graph.keys().map(|&g| (g, vec![])));
        for edge in edges {
            let g = graphs_by_node[edge.0];
            assert_eq!(g, graphs_by_node[edge.1]);
            edges_by_graph.get_mut(&g).map(|val| val.push(edge));
        }

        let graphs = nodes_by_graph.into_iter().sorted_unstable().map(|(g, nodes)| {
            let min = *nodes.iter().min().unwrap();
            assert_eq!((min..(min + nodes.len())).collect_vec(), nodes);
            let es = edges_by_graph[&g].iter().map(|&(i, j)| (i - min, j - min)).collect_vec();
            // Duplicate edges exist for example in IMDB-BINARY:
            let dedup = es.iter().cloned().collect::<FHashSet<_>>();
            (DiGraph::plain(nodes.len(), dedup).into_undirected(), es)
        }).collect_vec();
        assert_eq!(graphs_by_node.len(), graphs.iter().map(|(g, _)| g.len()).sum::<usize>());
        self.verify_graphs(&graphs.iter().map(|(g, _)| g).collect_vec());
        graphs
    }

    fn verify_graphs<'a>(&self, graphs: &Vec<&'a UnGraph>) {
        let num_graphs = graphs.len();
        assert_eq!(self.num_graphs, num_graphs, "{} graphs expected, found {}.", self.num_graphs, num_graphs);
        let avg_nodes = graphs.iter().map(|g| g.len()).sum::<usize>() as f64 / num_graphs as f64;
        assert!((self.avg_nodes - avg_nodes).abs() <= 5e-3, "{} nodes on average expected, found {}.", self.avg_nodes, avg_nodes);
        let avg_edges = graphs.iter().map(|g| g.num_edges()).sum::<usize>() as f64 / num_graphs as f64;
        assert!((self.avg_edges - avg_edges).abs() <= 5e-3, "{} edges on average expected, found {}.", self.avg_edges, avg_edges);
    }

    fn read_node_labels(&self) -> impl Iterator<Item=usize> + '_ {
        Self::read_indices(&self.node_labels_path()).into_iter().map(|x| {
            if self.has_temporal_node_labels {
                assert!(x.len() >= 2);
                assert_eq!(x[0], 0);
                assert_eq!(x[1], 0);
                return match x.len() {
                    2 => 0,
                    4 => {
                        assert_eq!(x[3], 1);
                        x[2] + 1
                    }
                    _ => panic!("Invalid temporal node label: {:?}", x)
                };
            }
            if self.name == "Cuneiform" {
                assert_eq!(x.len(), 2);
                assert!(x[0] < 4);
                assert!(x[1] < 3);
                return x[0] * 3 + x[1];
            }

            assert_eq!(x.len(), 1);
            x[0]
        })
    }

    fn read_edge_labels(&self) -> impl Iterator<Item=usize> {
        Self::read_indices(&self.edge_labels_path()).into_iter().map(|x| {
            assert_eq!(x.len(), 1);
            x[0]
        })
    }

    fn node_labelled_graphs_with_edge_order(&self) -> Option<Vec<(NodeLabelledGraph<Undirected>, Vec<EdgeIndex>)>> {
        if !self.has_node_labels {
            return None;
        }
        let graphs_with_order = self.graphs_with_edge_order();
        let all_node_labels = self.read_node_labels().collect_vec();
        let mut start = 0;
        let mut out = vec![];
        for (g, edge_indices) in graphs_with_order {
            let node_labels = all_node_labels[start..start + g.len()].to_vec();
            start += g.len();
            let edges = g.edges();
            out.push((NodeLabelledGraph::new(node_labels, edges), edge_indices));
        }
        assert_eq!(all_node_labels.len(), start);
        Some(out)
    }
}

struct Source {
    info: SourceInfo,
    cache: PathBuf,
    index_path: PathBuf,
}

impl Source {
    fn new(info: SourceInfo, parent: &Path) -> Self {
        let cache = parent.join(&info.name);
        fs::create_dir_all(&cache).unwrap();
        let cache = fs::canonicalize(cache).unwrap();
        let index_path = cache.join("datasets.md");
        Self { info, cache, index_path }
    }

    fn index_download(&self) -> Option<Download> {
        if self.index_path.exists() {
            return None;
        }
        Some(Download::new(&self.info.index_url).file_name(&self.index_path))
    }

    fn read_index(&self) -> Vec<DatasetInfo> {
        let mut out = vec![];
        let mut category = None;
        for line in lines(&self.index_path) {
            if line.starts_with("## ") {
                category = Some(line[3..].to_string());
                continue;
            } else if !line.starts_with(&"|**") && !line.starts_with(&"| **") {
                continue;
            }
            let fields = line.split("|").map(|x| x.trim().to_string()).collect_vec();
            assert!(fields.len() >= 2);
            assert_eq!(fields.first().unwrap(), "");
            assert_eq!(fields.last().unwrap(), "");
            let fields = &fields[1..fields.len() - 1];

            assert_eq!(fields.len(), 12);
            let name = &fields[0];
            let name = name[2..name.len() - 2].to_string();
            if name == "Name" {
                assert_eq!(fields[1], "**Source**");
                assert_eq!(fields[2], "**Statistics**");
                assert_eq!(fields[11], "**Download (ZIP)**");
                continue;
            }
            let num_graphs = fields[2].parse::<usize>().unwrap();
            let avg_nodes = fields[4].parse::<f64>().unwrap();
            let avg_edges = fields[5].parse::<f64>().unwrap();
            let node_labels_field = fields[6].as_str();
            let (has_node_labels, has_temporal_node_labels) = match node_labels_field {
                "--" => (false, false),
                "+" => (true, false),
                "temporal" => (true, true),
                _ => panic!("Unknown value '{node_labels_field}' in node labels column.")
            };
            let edge_labels_field = fields[7].as_str();
            let has_edge_labels = match edge_labels_field {
                "--" => false,
                "+" => true,
                _ => panic!("Unknown value '{edge_labels_field}' in edge labels column.")
            };
            let url_link = fields[11].to_string();
            let start = format!("[{name}](");
            assert!(url_link.starts_with(&start));
            assert!(url_link.ends_with(")"));
            let zip_url = url_link[start.len()..url_link.len() - 1].to_string();
            let cache = self.cache.join(&name).to_owned();
            let index = self.info.clone();
            let category = category.clone().unwrap();
            out.push(DatasetInfo { name, zip_url, cache, source: index, category, has_node_labels, has_temporal_node_labels, has_edge_labels, num_graphs, avg_nodes, avg_edges });
        }
        out
    }
}

/// Have a static DOWNLOADER instance to avoid collisions from parallel threads.
/// This is a coarse implementation, but it is sufficient for the current use case.
static DOWNLOADER: LazyLock<Mutex<Downloader>> = LazyLock::new(|| Mutex::new(Downloader::builder().build().unwrap()));

fn download_files(downloader: &mut MutexGuard<Downloader>, downloads: &Vec<Download>) {
    for d in downloads.iter() {
        assert!(d.file_name.is_absolute(), "Download file path must be absolute.");
    }
    let downloads = downloads.as_slice();
    for result in downloader.download(downloads).unwrap() {
        result.unwrap();
    }
}

pub fn download(datasets: &Vec<DatasetInfo>) -> Vec<Dataset> {
    let downloader = &mut DOWNLOADER.lock().unwrap();
    let downloads = datasets.iter().filter_map(|i| i.zip_download()).collect_vec();
    if !downloads.is_empty() {
        print_flush!("Downloading datasets [{}]...", downloads.iter().map(|d| d.file_name.file_stem().unwrap().to_str().unwrap()).join(", "));
        download_files(downloader, &downloads);
        println!(" done.")
    }
    datasets.into_iter().map(|info| Dataset::new(info.clone())).collect()
}

/// Graph datasets from the given index files cached at the given path.
pub fn datasets_from_at(indices: impl IntoIterator<Item=SourceInfo>, cache: &Path) -> Vec<DatasetInfo> {
    let downloader = &mut DOWNLOADER.lock().unwrap();
    fs::create_dir_all(cache).unwrap();
    let indices = indices.into_iter().map(|s| Source::new(s, cache)).collect_vec();
    download_files(downloader, &indices.iter().filter_map(|s| s.index_download()).collect());
    indices.iter().flat_map(|s| s.read_index()).collect()
}

#[allow(unused)]
pub fn datasets_from(indices: impl IntoIterator<Item=SourceInfo>) -> Vec<DatasetInfo> {
    datasets_from_at(indices, Path::new("data"))
}

#[allow(unused)]
pub fn datasets() -> Vec<DatasetInfo> {
    datasets_from(SourceInfo::defaults())
}

fn dataset_from(datasets: &Vec<DatasetInfo>, name: &str) -> Dataset {
    let matched = datasets.iter().filter(|x| x.name == name).cloned().collect_vec();
    assert_eq!(matched.len(), 1);
    download(&matched)[0].clone()
}

#[allow(unused)]
pub fn dataset(name: &str) -> Dataset {
    dataset_from(&datasets(), name)
}

fn lines(path: &Path) -> impl Iterator<Item=String> {
    BufReader::new(File::open(path).unwrap()).lines().map(|x| x.unwrap())
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::bench::datasets::dataset;
    use crate::bench::{test_and_print, DatasetStats, TestConfig};
    use crate::codec::Independent;
    use crate::experimental::complete_joint::aut::AutCanonizable;
    use crate::experimental::complete_joint::graph::tests::{with_edge_labelled_uniform, with_node_labelled_uniform};
    use crate::experimental::graph::tests::{test_graph_shuffle_codecs, test_joint_shuffle_codecs};
    use crate::experimental::graph::{erdos_renyi_indices, polya_urn, EmptyCodec, GraphIID};
    use crate::permutable::{Permutable, Unordered};
    #[test]
    fn unlabelled() {
        let graphs = dataset("MUTAG").unlabelled_graphs();
        let DatasetStats { edge, .. } = DatasetStats::unlabelled(&graphs);
        let codec = Independent::new(graphs.iter().map(|x| {
            GraphIID::new(x.len(), erdos_renyi_indices(x.len(), edge.clone(), false), EmptyCodec::default(), EmptyCodec::default())
        }).collect_vec());

        assert!(test_and_print(&codec, &graphs, 0..1) < 16500.);
    }

    #[test]
    fn test_polyas_urn() {
        let graphs = &dataset("MUTAG").unlabelled_graphs().into_iter().take(10).collect_vec();
        let unordered = &graphs.into_iter().cloned().map(|g| Unordered(g)).collect_vec();
        for loops in [false, true] {
            for redraws in [false, true] {
                let pu = &graphs.iter().map(|x| {
                    let edge_indices = polya_urn(x.len(), x.num_edges(), loops, redraws);
                    GraphIID::new(x.len(), edge_indices, EmptyCodec::default(), EmptyCodec::default())
                }).collect();
                test_joint_shuffle_codecs(&pu, unordered, &TestConfig::test(0));
                println!()
            }
        }
    }

    #[test]
    fn node_labelled() {
        let graphs = dataset("MUTAG").node_labelled_graphs().unwrap();
        test_graph_shuffle_codecs(with_node_labelled_uniform(graphs.into_iter().take(20), 0.1, false), &TestConfig::test(0));
    }

    #[test]
    fn edge_labelled() {
        let graphs = dataset("MUTAG").edge_labelled_graphs().unwrap();
        test_graph_shuffle_codecs(with_edge_labelled_uniform(graphs.into_iter().take(20), 0.1, false), &TestConfig::test(0));
    }

    #[test]
    #[ignore = "long-running"]
    fn edge_labels_without_node_labels() {
        let graphs = dataset("COIL-DEL").edge_labelled_graphs().unwrap();
        test_graph_shuffle_codecs(with_edge_labelled_uniform(graphs.into_iter().take(20), 0.1, false), &TestConfig::test(0));
    }

    #[test]
    #[ignore = "long-running"]
    fn node_labels_special_case() {
        let graphs = dataset("Cuneinform").node_labelled_graphs().unwrap();
        test_graph_shuffle_codecs(with_node_labelled_uniform(graphs.into_iter().take(20), 0.1, false), &TestConfig::test(0));
    }

    #[test]
    #[ignore = "long-running"]
    fn test_slow_reddit_graph() {
        let graphs = dataset("REDDIT-BINARY").unlabelled_graphs();
        for g in graphs {
            g.automorphisms();
        }
        // let slow_stab_chain = 1173; // MULTI-5K: 4935
        // test_shuffle_undirected_codecs(with_unlabelled_codecs([graph.clone()]), 0);
    }
}
