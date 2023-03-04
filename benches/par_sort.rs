use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use fnntw::{point::Point, NotNan};
use rayon::current_num_threads;

type T = f32;
const D: usize = 3;

fn criterion_benchmark(c: &mut Criterion) {
    for ndata in [3, 4, 5, 6, 7, 8, 9].map(|p| 10_usize.pow(p)) {
        let data_inner: Vec<[NotNan<T>; D]> = (0..ndata)
            .map(|_| [(); D].map(|_| NotNan::new(rand::random()).unwrap()))
            .collect();
        let data: Vec<Point<T, D>> = data_inner.iter().map(Point::new).collect();
        let mut group = c.benchmark_group(format!("median vs moms"));
        group
            // .confidence_level(0.99)
            .sample_size(100)
            .measurement_time(Duration::from_secs(5));

        group.bench_function(format!("median {}", data.len()), |b| {
            let median_index = data.len() / 2 + data.len() % 2;
            b.iter_batched(
                || data.clone(),
                |data| {
                    black_box(data).select_nth_unstable_by(median_index, |a, b| unsafe {
                        a.get_unchecked(D / 2)
                            .partial_cmp(b.get_unchecked(D / 2))
                            .unwrap_unchecked()
                    });
                },
                BatchSize::LargeInput,
            );
        });
        for bsize in [
            ndata / (rayon::current_num_threads() * 8),
            ndata / (rayon::current_num_threads() * 5),
            ndata / (rayon::current_num_threads() * 4),
            ndata / (rayon::current_num_threads() * 2),
            ndata / rayon::current_num_threads(),
        ] {
            // let bsize = ndata / (rayon::current_num_threads() * 4);
            group.bench_function(format!("par approx median {}/{bsize}", data.len()), |b| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        fnntw::moms::moms_seq(&mut data, Some(bsize), 0);
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
