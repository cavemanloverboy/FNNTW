use fnntw::{Tree, distance::squared_euclidean};
use ordered_float::NotNan;
use rand::{rngs::ThreadRng, Rng};
use std::error::Error;

const NDATA: usize = 100;
const NQUERY: usize = 10_000;
const D: usize = 3;
const K: usize = 80;

#[test]
fn test_brute_force_k() -> Result<(), Box<dyn Error>>{

    // Random number generator
    let mut rng = rand::thread_rng();

    // Generate random data, query
    let mut data = Vec::with_capacity(NDATA);
    let mut query = Vec::with_capacity(NQUERY);
    for _ in 0..NDATA {
        data.push(random_point(&mut rng));
    }
    for _ in 0..NQUERY {
        query.push(random_point(&mut rng));
    }

    // Construct tree
    let tree = Tree::<'_, D>::new_parallel(&data, 1, 1).unwrap();

    // Query tree
    let mut results = Vec::with_capacity(NQUERY);
    for q in &query {
        results.push(tree.query_nearest_k(q, K)?);
    }

    // Brute force check results
    for (i, q) in query.iter().enumerate() {
        let result = &results[i];
        let expected = brute_force_k(q, &data, K);
        assert_eq!(result.len(), K);
        assert_eq!(expected.len(), K);
        assert_eq!(*result, expected);
        assert_eq!(*result, expected);
        assert_eq!(*result, expected);
    }

    Ok(())
}


fn random_point<const D: usize>(rng: &mut ThreadRng) -> [f64; D] {
    [(); D].map(|_|rng.gen() )
}

#[cfg(feature = "do-not-return-position")]
fn brute_force_k<'d, const D: usize>(
    q: &[f64; D],
    data: &'d [[f64; D]],
    k: usize,
) -> Vec<(f64, u64)> {

    // No need for nan checks here
    let q: &[NotNan<f64>; D]= unsafe { std::mem::transmute(q) };
    let data: &'d [[NotNan<f64>; D]] = unsafe { std::mem::transmute(data) };

    let mut all = Vec::with_capacity(data.len());

    for (d, i) in data.iter().zip(0_u64..) {
        let dist = squared_euclidean::<D>(q, d);
        all.push((dist, i))
    }

    // this is safe so long as [0, 1] randoms are used
    all.sort_by(|p1, p2| p1.0.partial_cmp(&p2.0).unwrap());
    all.truncate(k);
    all
}

#[cfg(not(feature = "do-not-return-position"))]
fn brute_force_k<'d, const D: usize>(
    q: &[f64; D],
    data: &'d [[f64; D]],
    k: usize,
) -> Vec<(f64, u64, &'d[NotNan<f64>; D])> {

    // No need for nan checks here
    let q: &[NotNan<f64>; D]= unsafe { std::mem::transmute(q) };
    let data: &'d [[NotNan<f64>; D]] = unsafe { std::mem::transmute(data) };

    let mut all = Vec::with_capacity(data.len());

    for (d, i) in data.iter().zip(0_u64..) {
        let dist = squared_euclidean::<D>(q, d);
        all.push((dist, i, d))
    }

    // this is safe so long as [0, 1] randoms are used
    all.sort_by(|p1, p2| p1.0.partial_cmp(&p2.0).unwrap());
    all.truncate(k);
    all
}