[Preview.]

# Shuffle Coding

Shuffle coding is a general method for optimal compression of unordered objects using bits-back
coding.
Data structures that can be compressed with our method include multisets, graphs, hypergraphs, and others.
Shuffle coding state-of-the-art one-shot compression rates on various large network graphs at
competitive speeds.
This implementation can be easily adapted to different data types and statistical models.

We published [Practical Shuffle Coding](https://neurips.cc/virtual/2024/poster/93780) at NeurIPS,
based on our earlier ICLR publication [Entropy Coding of Unordered Data Structures](https://arxiv.org/abs/2408.08837).
This is the official implementation for both papers.

## Build and run

Apart from [Rust](https://www.rust-lang.org/tools/install), you need a C compiler to build this project due to
dependency on [nauty and Traces](https://github.com/a-maier/nauty-Traces-sys).

To replicate experiments from our "Practical Shuffle Coding" paper, run:

```shell
# Multiset:
cargo test --release benchmark_multiset -- --ignored
# Joint:
for convs in {0..7};
  cargo run --release --feature bench -- --joint --ap SZIP --seeds 10 --convs $convs

cargo run --release --feature bench -- --joint --er TU --seeds 3 --threads 1
cargo run --release --feature bench -- --joint --ap SZIP --seeds 10
cargo run --release --feature bench -- --joint --ap SZIP --seeds 10 --threads 1
cargo run --release --feature bench -- --joint --ap REC --seeds 10
cargo run --release --feature bench -- --joint --ap REC --seeds 10 --threads 1
# Autoregressive:
for chunks in 1 2 4 8 16 32 64 128 256 512 1024;
  cargo run --release --feature bench -- --ar --ap SZIP --seeds 10 --chunks $chunks

cargo run --release --feature bench -- --ar --ap SZIP --seeds 10 --ae
cargo run --release --feature bench -- --ar --ap SZIP --seeds 10 --chunks 200
cargo run --release --feature bench -- --ar --ap REC --seeds 10 --ordered
cargo run --release --feature bench -- --ar --ap REC --seeds 10 --threads=1
```

To replicate experiments from our earlier paper "Entropy Coding of Unordered Data Structures", run:

```shell
cargo run --release --feature bench -- --complete --no-incomplete --joint --ordered --symstats --stats --er AIDS..mit_ct2 reddit_threads..TRIANGLES
cargo run --release --feature bench -- --complete --no-incomplete --joint --seeds 3 --symstats --stats --er --eru --pur --pu MUTAG PTC_MR
cargo run --release --feature bench -- --complete --no-incomplete --joint --seeds 3 --nolabels --er --eru --pur --pu MUTAG PTC_MR ZINC_full..ZINC_val PROTEINS IMDB-BINARY IMDB-MULTI
cargo run --release --feature bench -- --complete --no-incomplete --joint --ordered --symstats --stats --er SZIP # Takes days
cargo run --release --feature bench -- --complete --no-incomplete --joint --seeds 3 --pu --ap SZIP # Takes weeks
```

Graph datasets are downloaded automatically as
needed ([TU](https://chrsmrrs.github.io/datasets/), [SZIP](https://github.com/juliuskunze/szip-graphs)
and [REC](https://github.com/juliuskunze/rec-graphs)).

Use `cargo run -- --help` to see all available options.

## Module structure

- [`codec`](src/codec): Asymmetric numeral systems (ANS) and basic codecs.
- [`permutable`](src/permutable) Multisets and graphs.
- [`joint`](src/joint): Joint shuffle coding.
- [`autoregressive`](src/autoregressive): Autoregressive shuffle coding.
- [`experimental`](src/experimental): Experimental code that was previously used for research, typically too slow to be
  practical. Here for reference.
- [`bench`](src/bench): Benchmarks used for the experiments in our papers.

## Citing

If you find this code useful, please reference in your paper:

```bibtex
@article{kunze2024shuffle,
  title={Practical Shuffle Coding},
  author={Kunze, Julius and Severo, Daniel and van de Meent, Jan-Willem and Townsend, James},
  journal={NeurIPS},
  year={2024}
}

@article{kunze2024entropy,
  title={Entropy Coding of Unordered Data Structures},
  author={Kunze, Julius and Severo, Daniel and Zani, Giulio and van de Meent, Jan-Willem and Townsend, James},
  journal={ICLR},
  year={2024}
}
```
