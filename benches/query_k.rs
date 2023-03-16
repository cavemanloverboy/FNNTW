use fnntw::Tree;
use rayon::prelude::*;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_format::{Locale, ToFormattedString};

type T = f64;
const D: usize = 3;
const QUERY: usize = 4_000_000;
const BOXSIZE: [T; D] = [1.0; D];
const NDATA: usize = 4_000_000;
const LS: usize = 32;

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
        .sample_size(10)
        .confidence_level(0.95);

    for k in [10] {
        let tree = Tree::new(black_box(&data), LS).unwrap();

        group.bench_function(format!("k={k} nonpbc"), |b| {
            b.iter(|| {
                let v: Vec<_> = black_box(&query)
                    .par_iter()
                    .map_with(black_box(&tree), |t, q| {
                        black_box(t.query_nearest_k(black_box(q), black_box(k))).unwrap();
                    })
                    .collect();
                v
            })
        });
        group.bench_function(format!("k={k} nonpbc noidx"), |b| {
            b.iter(|| {
                let v: Vec<_> = black_box(&query)
                    .par_iter()
                    .map_with(black_box(&tree), |t, q| {
                        black_box(t.query_nearest_k_noidx(black_box(q), black_box(k))).unwrap();
                    })
                    .collect();
                v
            })
        });
        // group.bench_function(format!("k={k} with nonpbc"), |b| {
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
        //                     black_box(k),
        //                     // SAFETY: we are transmuting the reference lifetimes of empty vectors,
        //                     unsafe { std::mem::transmute(&mut container) },
        //                     unsafe { std::mem::transmute(&mut point_vec) },
        //                 ))
        //                 .unwrap();
        //                 drop(results);
        //                 tx.send((container, point_vec)).unwrap();
        //             })
        //             .collect();
        //     })
        // });
        group.bench_function(format!("k={k} parallel nonpbc"), |b| {
            b.iter(|| {
                tree.query_nearest_k_parallel(black_box(&query), black_box(k))
                    .unwrap();
            })
        });

        // group.bench_function(format!("k={k} parallel with nonpbc"), |b| {
        //     b.iter(|| {
        //         tree
        //             .query_nearest_k_parallel_with(black_box(&query), black_box(k))
        //             .unwrap()
        //     })
        // });
        #[cfg(feature = "no-position")]
        group.bench_function(format!("k={k} parallel py nonpbc"), |b| {
            b.iter(|| {
                let mut distances = Vec::<T>::with_capacity(query.len() * k);
                let mut indices = Vec::<u64>::with_capacity(query.len() * k);
                let dist_ptr_usize = distances.as_mut_ptr() as usize;
                let idx_ptr_usize = indices.as_mut_ptr() as usize;
                query.par_iter().enumerate().for_each_with(
                    (dist_ptr_usize, idx_ptr_usize),
                    |(d, i), (j, q)| {
                        let result = tree
                            .query_nearest_k(black_box(&q), black_box(k))
                            .expect("error occurred during query");
                        unsafe {
                            let d = *d as *mut T;
                            let i = *i as *mut u64;
                            for kk in 0..k {
                                *d.add(k * j + kk) = *result.0.get_unchecked(kk);
                                *i.add(k * j + kk) = *result.1.get_unchecked(kk);
                            }
                        }
                    },
                );
                unsafe {
                    distances.set_len(query.len() * k);
                    indices.set_len(query.len() * k);
                }
            })
        });
        #[cfg(feature = "no-position")]
        group.bench_function(format!("k={k} parallel py buffered nonpbc"), |b| {
            let mut distances = Vec::<T>::with_capacity(query.len() * k);
            let mut indices = Vec::<u64>::with_capacity(query.len() * k);
            let dist_ptr_usize = distances.as_mut_ptr() as usize;
            let idx_ptr_usize = indices.as_mut_ptr() as usize;
            b.iter(|| {
                query.par_iter().enumerate().for_each_with(
                    (dist_ptr_usize, idx_ptr_usize),
                    |(d, i), (j, q)| {
                        let (dists, idxs) = tree
                            .query_nearest_k(black_box(&q), black_box(k))
                            .expect("error occurred during query");
                        unsafe {
                            let d = *d as *mut T;
                            let i = *i as *mut u64;
                            for kk in 0..k {
                                *d.add(k * j + kk) = *dists.get_unchecked(kk);
                            }
                            for kk in 0..k {
                                *i.add(k * j + kk) = *idxs.get_unchecked(kk);
                            }
                        }
                    },
                );
                unsafe {
                    distances.set_len(query.len() * k);
                    indices.set_len(query.len() * k);
                }
            })
        });

        let tree = tree.with_boxsize(&BOXSIZE).unwrap();
        // group.bench_function(format!("k={k} pbc"), |b| {
        //     b.iter(|| {
        //         let v: Vec<_> = black_box(&query)
        //             .par_iter()
        //             .map_with(black_box(&tree), |t, q| {
        //                 let results = black_box(
        //                     t.query_nearest_k(black_box(q), black_box(k)),
        //                 );
        //                 drop(results);
        //             })
        //             .collect();
        //         drop(v)
        //     })
        // });

        // group.bench_function(format!("k={k} with pbc"), |b| {
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
        //                     black_box(k),
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

        group.bench_function(format!("k={k} parallel pbc"), |b| {
            b.iter(|| {
                tree.query_nearest_k_parallel(black_box(&query), black_box(k))
                    .unwrap()
            })
        });

        // group.bench_function(format!("k={k} parallel with pbc"), |b| {
        //     b.iter(|| {
        //         let v = tree
        //             .query_nearest_k_parallel_with(black_box(&query), black_box(k))
        //             .unwrap();
        //         drop(v)
        //     })
        // });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
