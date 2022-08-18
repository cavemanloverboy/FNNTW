use rayon::prelude::*;
use fnstw::Tree;
use ordered_float::NotNan;

use criterion::{black_box, criterion_group, criterion_main, Criterion};


const D: usize = 3;


fn criterion_benchmark(c: &mut Criterion) {

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

        let tree = Tree::new(&data, 32).unwrap();
        group.bench_function(format!("Query (ndata = {ndata})").as_str(), |b| b.iter(|| {
            let v: Vec<_> = query
                .par_iter()
                .map_with(&tree, |t, q| {
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