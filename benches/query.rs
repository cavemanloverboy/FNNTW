use fnntw::Tree;
use ordered_float::NotNan;
use rayon::prelude::*;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_format::{Locale, ToFormattedString};

const D: usize = 3;
const QUERY: usize = 1_000_000;
const BOXSIZE: [NotNan<f64>; D] = unsafe { [NotNan::new_unchecked(1.0); D] };

fn criterion_benchmark(c: &mut Criterion) {

    // Bench building tree
    for ndata in [3, 4, 5].map(|p| 10_usize.pow(p)) {

        let data: Vec<[NotNan<f64>; D]> = (0..ndata)
            .map(|_| [(); D].map(|_| unsafe { NotNan::new_unchecked(rand::random()) }))
            .collect();
        let query: Vec<[NotNan<f64>; D]> = (0..QUERY)
            .map(|_| [(); D].map(|_| unsafe { NotNan::new_unchecked(rand::random()) }))
            .collect();

        let mut group = c.benchmark_group(
            format!(
                "{} queries (ndata = {})",
                QUERY.to_formatted_string(&Locale::en),
                ndata
            )
        );

        let tree = Tree::new(black_box(&data), black_box(32)).unwrap();
        group.bench_function(
            "non-periodic",
            |b| {
                b.iter(|| {
                    let v: Vec<_> = black_box(&query)
                        .par_iter()
                        .map_with(black_box(&tree), |t, q| {
                            let (dist, idx, ptr) = t.query_nearest(black_box(q));
                            drop(dist);
                            drop(idx);
                            drop(ptr);
                        })
                        .collect();
                    drop(v)
                })
            },
        );

        group.bench_function(
            "periodic",
            |b| {
                b.iter(|| {
                    let v: Vec<_> = black_box(&query)
                        .par_iter()
                        .map_with(black_box(&tree), |t, q| {
                            let (dist, idx, ptr) = t.query_nearest_periodic(black_box(q), &BOXSIZE);
                            drop(dist);
                            drop(idx);
                            drop(ptr);
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
