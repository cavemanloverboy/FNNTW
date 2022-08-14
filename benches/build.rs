use std::error::Error;
use fnstw::Tree;
use ordered_float::NotNan;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

const D: usize = 3;


fn criterion_benchmark(c: &mut Criterion) {

    let mut group = c.benchmark_group("sample-size-example");
    group.sample_size(10);

    // Bench building tree
    for ndata in [3, 4, 5].map(|p| 10_usize.pow(p)){

        let data: Vec<[NotNan<f64>; D]> = (0..ndata)
            .map(|_| [(); D].map(|_| unsafe { NotNan::new_unchecked(rand::random()) } ))
            .collect();

        group.bench_function(format!("Build (ndata = {ndata})").as_str(), |b| b.iter(|| {
            let tree = Tree::new(&data, 16).unwrap();
            // println!("size = {}", tree.size())
        } ));        
    }
    group.finish();

}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);