use fnntw::{point::Point, query_k::container::Container, Tree};
use rayon::prelude::*;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_format::{Locale, ToFormattedString};

type T = f64;
const D: usize = 3;
const QUERY: usize = 100_000;
const BOXSIZE: [T; D] = [1.0; D];
const NDATA: usize = 100_000;

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
        .sample_size(10);

    for log2k in [5, 6, 7, 8, 9, 10] {
        let tree = Tree::new(black_box(&data), black_box(32)).unwrap();

        group.bench_function(format!("log2(k)={log2k} nonpbc"), |b| {
            b.iter(|| {
                let v: Vec<_> = black_box(&query)
                    .par_iter()
                    .map_with(black_box(&tree), |t, q| {
                        let results = black_box(
                            t.query_nearest_k(black_box(q), black_box(2_usize.pow(log2k))),
                        );
                        drop(results);
                    })
                    .collect();
                drop(v)
            })
        });
        // group.bench_function(format!("log2(k)={log2k} with nonpbc"), |b| {
        //     let (tx, rx) = crossbeam_channel::bounded(rayon::max_num_threads());

        //     b.iter(|| {
        //         let v: Vec<_> = black_box(&query)
        //             .par_iter()
        //             .map_with(black_box(&tree), |t, q| {
        //                 let (mut container, mut point_vec) = rx.try_recv().unwrap_or_else(|_| {
        //                     (
        //                         Container::<T, D>::new(2_usize.pow(log2k)),
        //                         Vec::<(&usize, &Point<T, D>, T)>::with_capacity(tree.height_hint),
        //                     )
        //                 });
        //                 let results = black_box(t.query_nearest_k_with(
        //                     black_box(q),
        //                     black_box(2_usize.pow(log2k)),
        //                     // SAFETY: we are transmuting the reference lifetimes of empty vectors,
        //                     unsafe { std::mem::transmute(&mut container) },
        //                     unsafe { std::mem::transmute(&mut point_vec) },
        //                 ))
        //                 .unwrap();
        //                 drop(results);
        //                 tx.send((container, point_vec)).unwrap();
        //             })
        //             .collect();
        //         drop(v)
        //     })
        // });
        // group.bench_function(format!("log2(k)={log2k} parallel nonpbc"), |b| {
        //     b.iter(|| {
        //         let v = tree
        //             .query_nearest_k_parallel(black_box(&query), black_box(2_usize.pow(log2k)))
        //             .unwrap();
        //         drop(v)
        //     })
        // });

        // group.bench_function(format!("log2(k)={log2k} parallel with nonpbc"), |b| {
        //     b.iter(|| {
        //         let v = tree
        //             .query_nearest_k_parallel_with(black_box(&query), black_box(2_usize.pow(log2k)))
        //             .unwrap();
        //         drop(v)
        //     })
        // });

        let tree = tree.with_boxsize(&BOXSIZE).unwrap();
        group.bench_function(format!("log2(k)={log2k} pbc"), |b| {
            b.iter(|| {
                let v: Vec<_> = black_box(&query)
                    .par_iter()
                    .map_with(black_box(&tree), |t, q| {
                        let results = black_box(
                            t.query_nearest_k(black_box(q), black_box(2_usize.pow(log2k))),
                        );
                        drop(results);
                    })
                    .collect();
                drop(v)
            })
        });

        // group.bench_function(format!("log2(k)={log2k} with pbc"), |b| {
        //     let (tx, rx) = crossbeam_channel::bounded(rayon::max_num_threads());
        //     b.iter(|| {
        //         let v: Vec<_> = black_box(&query)
        //             .par_iter()
        //             .map_with(black_box(&tree), |t, q| {
        //                 let (mut container, mut point_vec) = rx.try_recv().unwrap_or_else(|_| {
        //                     (
        //                         Container::<T, D>::new(2_usize.pow(log2k)),
        //                         Vec::<(&usize, &Point<T, D>, T)>::with_capacity(tree.height_hint),
        //                     )
        //                 });
        //                 let results = black_box(t.query_nearest_k_with(
        //                     black_box(q),
        //                     black_box(2_usize.pow(log2k)),
        //                     // SAFETY: we are transmuting the reference lifetimes of empty vectors,
        //                     unsafe { std::mem::transmute(&mut container) },
        //                     unsafe { std::mem::transmute(&mut point_vec) },
        //                 ));
        //                 drop(results);
        //                 tx.send((container, point_vec)).unwrap();
        //             })
        //             .collect();
        //         drop(v)
        //     })
        // });

        // group.bench_function(format!("log2(k)={log2k} parallel pbc"), |b| {
        //     b.iter(|| {
        //         let v = tree
        //             .query_nearest_k_parallel(black_box(&query), black_box(2_usize.pow(log2k)))
        //             .unwrap();
        //         drop(v)
        //     })
        // });

        // group.bench_function(format!("log2(k)={log2k} parallel with pbc"), |b| {
        //     b.iter(|| {
        //         let v = tree
        //             .query_nearest_k_parallel_with(black_box(&query), black_box(2_usize.pow(log2k)))
        //             .unwrap();
        //         drop(v)
        //     })
        // });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
