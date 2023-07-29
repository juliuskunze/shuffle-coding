This is accompanying code for our workshop paper "Entropy Coding of Unordered Data Structures", which was accepted for oral presentation at the ICML 2023 Neural Compression Workshop.

### [Workshop paper](https://openreview.net/attachment?id=PggJ9CbEN7&name=pdf) | [Presentation](https://www.youtube.com/watch?v=RJ92tN4P14M) (13 minutes)

# Shuffle Coding

Shuffle coding is a general method for optimal compression of sequences of unordered
objects using bits-back coding. Data structures that can be compressed using shuffle coding include multisets, graphs, hypergraphs, and others. 
This implementation can easily be adapted to different data types and statistical models, 
and achieves state-of-the-art compression rates on a range of graph datasets including molecular data.

## Run

Apart from [Rust](https://www.rust-lang.org/tools/install), you need a C compiler to build this project due to dependency on [nauty and Traces](https://github.com/a-maier/nauty-Traces-sys).

Use the following commands to replicate all experiments from the paper:

```shell
cargo run --release -- --symstats --stats --er --shuffle AIDS..mit_ct2 reddit_threads..TRIANGLES
cargo run --release -- --symstats --stats --er --eru --pu --pur --shuffle MUTAG PTC_MR
cargo run --release -- --symstats --stats --er --eru --pu --pur --shuffle --plain MUTAG PTC_MR ZINC_full..ZINC_val PROTEINS IMDB-BINARY IMDB-MULTI
cargo run --release -- --symstats --stats --er --pu --shuffle SZIP
```

Graph datasets are downloaded automatically as needed (from [TU](https://chrsmrrs.github.io/datasets/) and [SZIP](https://github.com/juliuskunze/szip-graphs)). Full results can be found [here](https://docs.google.com/spreadsheets/d/1YP0om6hNktaUhyFzFVVSh5SCTefZ8kNafYDYYIDolhM/edit#gid=1483743764).

To retrieve standard deviations for the stochastic Polya urn models, you can set a random seed (0, 1 and 2 were used for the paper):
```shell
cargo run --release -- MUTAG PTC_MR ZINC_full..ZINC_val PROTEINS IMDB-BINARY IMDB-MULTI --plain --pu --shuffle --seed=0
cargo run --release -- MUTAG PTC_MR --pu --shuffle --seed=0 
cargo run --release -- SZIP --pu --shuffle --seed=0
```

Use `cargo run --help` to see all available options.