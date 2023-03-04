use fnntw::{point::Point, query_k::container::Container, Tree};
use rayon::prelude::*;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_format::{Locale, ToFormattedString};

type T = f64;
const D: usize = 3;
const QUERY: usize = 4_000_000;
const BOXSIZE: [T; D] = [1.0; D];
const NDATA: usize = 4_000_000;
const LS: usize = 32;

fn criterion_benchmark(c: &mut Criterion) {
    let data: Vec<[T; D]> = (0..NDATA)
        .map(|_| [(); D].map(|_| rand::random()))
        .collect();
    let query: Vec<[T; D]> = (0..QUERY)
        .map(|_| [(); D].map(|_| rand::random()))
        .collect();

    let mut group = c.benchmark_group(format!(
        "{} queries (ndata = {})",
        QUERY.to_formatted_string(&Locale::en),
        NDATA.to_formatted_string(&Locale::en)
    ));
    group
        .plot_config(
            criterion::PlotConfiguration::default()
                .summary_scale(criterion::AxisScale::Logarithmic),
        )
        .sample_size(10)
        .confidence_level(0.95);

    for k in [10] {
        let tree = Tree::new(black_box(&data), LS).unwrap();

        group.bench_function(format!("k={k} axis nonpbc"), |b| {
            b.iter(|| {
                let v = tree.query_nearest_k_parallel_axis(black_box(&query), black_box(k), 0);
                v
            })
        });
        let tree = tree.with_boxsize(&BOXSIZE).unwrap();
        group.bench_function(format!("k={k} axis pbc"), |b| {
            b.iter(|| {
                let v = tree.query_nearest_k_parallel_axis(black_box(&query), black_box(k), 0);
                v
            })
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
