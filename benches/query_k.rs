use fnntw::Tree;
use rayon::prelude::*;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_format::{Locale, ToFormattedString};

const D: usize = 3;
const QUERY: usize = 1_000_000;
const BOXSIZE: [f64; D] = [1.0; D];
const NDATA: usize = 100_000;

fn criterion_benchmark(c: &mut Criterion) {

    let data: Vec<[f64; D]> = (0..NDATA)
        .map(|_| [(); D].map(|_| rand::random()))
        .collect();
    let query: Vec<[f64; D]> = (0..QUERY)
        .map(|_| [(); D].map(|_| rand::random()))
        .collect();

    let mut group = c.benchmark_group(
        format!(
            "{} queries (ndata = {})",
            QUERY.to_formatted_string(&Locale::en),
            NDATA.to_formatted_string(&Locale::en)
        )
    );
    group.plot_config(
        criterion::PlotConfiguration::default()
        .summary_scale(criterion::AxisScale::Logarithmic)
    );

    let tree = Tree::new(black_box(&data), black_box(32)).unwrap();
    for log2k in [5] {
        group.bench_function(
            format!("log2(k)={log2k} nonpbc"),
            |b| {
                b.iter(|| {
                    let v: Vec<_> = black_box(&query)
                        .par_iter()
                        .map_with(black_box(&tree), |t, q| {
                            let results = black_box(t.query_nearest_k(black_box(q), black_box(2_usize.pow(log2k))));
                            drop(results);
                        })
                        .collect();
                    drop(v)
                })
            },
        );

        group.bench_function(
            format!("log2(k)={log2k} pbc"),
            |b| {
                b.iter(|| {
                    let v: Vec<_> = black_box(&query)
                        .par_iter()
                        .map_with(black_box(&tree), |t, q| {
                            let results = black_box(t.query_nearest_k_periodic(black_box(q), black_box(2_usize.pow(log2k)), &BOXSIZE));
                            drop(results);
                        })
                        .collect();
                    drop(v)
                })
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
