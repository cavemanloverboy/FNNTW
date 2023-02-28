use fnntw::Tree;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

type T = f64;
const D: usize = 3;
const QUERY: usize = 1_000;
const BOXSIZE: [T; D] = [1.0; D];
const NDATA: usize = 1_000;
// const K: usize = 32;
const K: usize = 80;

#[test]
fn test_query_nearest_k_parallel() {
    let data: Vec<[T; D]> = (0..NDATA)
        .map(|_| [(); D].map(|_| rand::random()))
        .collect();
    let query: Vec<[T; D]> = (0..QUERY)
        .map(|_| [(); D].map(|_| rand::random()))
        .collect();

    let mut tree = Tree::new(&data, 32).unwrap();
    println!("constructed tree");

    // non pbc check
    let non_par_result: Vec<(Vec<f64>, Vec<u64>)> = query
        .iter()
        .map(|q| tree.query_nearest_k(q, K).unwrap())
        .collect();
    println!("finished par iter query");
    let par_result: (Vec<f64>, Vec<u64>) = tree.query_nearest_k_parallel(&query, K).unwrap();
    println!("finished native par query");
    for i in 0..QUERY {
        assert_eq!(
            &non_par_result[i].0.len(),
            &par_result.0[i * K..(i + 1) * K].len(),
            "{i}"
        );
        assert_eq!(
            &non_par_result[i].0,
            &par_result.0[i * K..(i + 1) * K],
            "{i}"
        );
        assert_eq!(
            &non_par_result[i].1,
            &par_result.1[i * K..(i + 1) * K],
            "{i}"
        );
    }
    println!("finished non pbc check");

    // pbc check
    let tree = tree.with_boxsize(&BOXSIZE).unwrap();
    let non_par_result: Vec<(Vec<f64>, Vec<u64>)> = query
        .par_iter()
        .map(|q| tree.query_nearest_k(q, K).unwrap())
        .collect();
    let par_result: (Vec<f64>, Vec<u64>) = tree.query_nearest_k_parallel(&query, K).unwrap();
    for i in 0..QUERY {
        assert_eq!(
            &non_par_result[i].0,
            &par_result.0[i * K..(i + 1) * K],
            "{i}"
        );
        assert_eq!(
            &non_par_result[i].1,
            &par_result.1[i * K..(i + 1) * K],
            "{i}"
        );
    }
    println!("finished pbc check");
}
