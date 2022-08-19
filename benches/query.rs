use rayon::prelude::*;
use fnstw::Tree;
use ordered_float::NotNan;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_format::{Locale, ToFormattedString};

const D: usize = 3;


fn criterion_benchmark(c: &mut Criterion) {

    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(1)
    //     .build_global();

    let mut group = c.benchmark_group("sample-size-example");
    group.sample_size(100);

    const QUERY: usize = 1_000_000;
    // Bench building tree
    for ndata in [3, 4, 5].map(|p| 10_usize.pow(p)){

        let data: Vec<[NotNan<f64>; D]> = (0..ndata)
            .map(|_| [(); D].map(|_| unsafe { NotNan::new_unchecked(rand::random()) } ))
            .collect();
        let query: Vec<[NotNan<f64>; D]> = (0..QUERY)
            .map(|_| [(); D].map(|_| unsafe { NotNan::new_unchecked(rand::random()) } ))
            .collect();

        let tree = Tree::new(black_box(&data), black_box(32)).unwrap();
        group.bench_function(
            format!("{} queries (ndata = {})", QUERY.to_formatted_string(&Locale::en), ndata).as_str(),
            |b| b.iter(|| {
            let v: Vec<_> = black_box(&query)
                .par_iter()
                .map_with(black_box(&tree), |t, q| {
                    t.query_nearest(black_box(q))
                })
                .collect();
            drop(v)
        } ));
    }
    group.finish();

}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);