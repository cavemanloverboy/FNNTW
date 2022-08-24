use fnntw::distance::*;
use ordered_float::NotNan;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean");
    group.sample_size(100);

    const NDATA: usize = 100_000;
    const D: usize = 3;

    let aa: Vec<[NotNan<f64>; D]> = [(); NDATA]
        .map(|_| [(); D].map(|_| NotNan::new(rand::random()).unwrap()))
        .to_vec();
    let bb: Vec<[NotNan<f64>; D]> = [(); NDATA]
        .map(|_| [(); D].map(|_| NotNan::new(rand::random()).unwrap()))
        .to_vec();

    group.bench_function("squared_euclidean 3D", |b| {
        b.iter(|| {
            for i in 0..NDATA {
                unsafe {
                    squared_euclidean(
                        black_box(aa.get_unchecked(i)),
                        black_box(bb.get_unchecked(i)),
                    )
                };
            }
        })
    });

    group.bench_function("squared_euclidean_sep 3D", |b| {
        b.iter(|| {
            for i in 0..NDATA {
                unsafe {
                    squared_euclidean_sep(
                        black_box(aa.get_unchecked(i)),
                        black_box(bb.get_unchecked(i)),
                    )
                };
            }
        })
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
