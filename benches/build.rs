use fnntw::Tree;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

type T = f64;
const D: usize = 3;

fn criterion_benchmark(c: &mut Criterion) {
    // Bench building tree
    for ndata in [5, 6, 7, 8].map(|p| 10_usize.pow(p)) {
        let data: Vec<[T; D]> = (0..ndata)
            .map(|_| [(); D].map(|_| rand::random()))
            .collect();
        let mut g = c.benchmark_group(format!("{ndata}"));
        g.sample_size(10).confidence_level(0.9);

        g.bench_function(format!("Build (ndata = {ndata})").as_str(), |b| {
            b.iter(|| {
                let tree = Tree::new(black_box(&data), black_box(32)).unwrap();
                drop(tree)
            })
        });

        for par_split in 1..=4 {
            g.sample_size(10).bench_function(
                format!("Parallel({par_split}) Build (ndata = {ndata})").as_str(),
                |b| {
                    b.iter(|| {
                        let tree = Tree::new_parallel(
                            black_box(&data),
                            black_box(32),
                            black_box(par_split),
                        )
                        .unwrap();
                        drop(tree)
                    })
                },
            );
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
