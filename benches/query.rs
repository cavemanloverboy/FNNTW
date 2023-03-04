use std::time::Duration;

use fnntw::Tree;
use rayon::prelude::*;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_format::{Locale, ToFormattedString};

type T = f32;
const D: usize = 3;
const QUERY: usize = 4_000_000;
const BOXSIZE: [T; D] = [1.0; D];

fn criterion_benchmark(c: &mut Criterion) {
    for ndata in [100_000, 4_000_000] {
        let data: Vec<[T; D]> = (0..ndata)
            .map(|_| [(); D].map(|_| rand::random()))
            .collect();
        let query: Vec<[T; D]> = (0..QUERY)
            .map(|_| [(); D].map(|_| rand::random()))
            .collect();

        let mut group = c.benchmark_group(format!(
            "{} queries (ndata = {})",
            QUERY.to_formatted_string(&Locale::en),
            ndata
        ));
        group
            .confidence_level(0.99)
            .sample_size(500)
            .warm_up_time(Duration::from_secs(5))
            .measurement_time(Duration::from_secs(20));

        let tree = Tree::new(black_box(&data), black_box(32)).unwrap();
        group.bench_function("non-periodic", |b| {
            b.iter(|| {
                let v: Vec<_> = black_box(&query)
                    .par_iter()
                    .map_with(black_box(&tree), |t, q| {
                        t.query_nearest(black_box(q)).unwrap()
                    })
                    .collect();
                drop(v)
            })
        });

        let tree = tree.with_boxsize(&BOXSIZE).unwrap();
        group.bench_function("periodic", |b| {
            b.iter(|| {
                let v: Vec<_> = black_box(&query)
                    .par_iter()
                    .map_with(black_box(&tree), |t, q| {
                        drop(t.query_nearest(black_box(q)).unwrap())
                    })
                    .collect();
                drop(v)
            })
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
