use fnntw::{Tree, distance::squared_euclidean};
use ordered_float::NotNan;
use rand::{rngs::ThreadRng, Rng};
use std::error::Error;


const NDATA: usize = 100;
const NQUERY: usize = 10_000;
const D: usize = 3;

#[test]
fn test_brute_force() -> Result<(), Box<dyn Error>> {

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
    let tree = Tree::<'_, D>::new(&data, 1).unwrap();

    // Query tree
    let mut results = Vec::with_capacity(NQUERY);
    for q in &query {
        results.push(tree.query_nearest(q)?);
    }

    // Brute force check results
    for (i, q) in query.iter().enumerate() {
        assert_eq!(results[i], brute_force(q, &data))
    }

    Ok(())
}


fn random_point<const D: usize>(rng: &mut ThreadRng) -> [f64; D] {
    [(); D].map(|_| rng.gen())
}

#[cfg(not(feature = "do-not-return-position"))]
fn brute_force<'d, const D: usize>(
    q: &[f64; D],
    data: &'d [[f64; D]],
) -> (f64, u64, &'d[NotNan<f64>; D]) {

    // No need for nan checks here
    let q: &[NotNan<f64>; D]= unsafe { std::mem::transmute(q) };
    let data: &'d [[NotNan<f64>; D]] = unsafe { std::mem::transmute(data) };

    let mut best_dist = std::f64::MAX;
    let mut best = (std::f64::MAX, std::u64::MAX, data.get(0).unwrap());
    for (d, i) in data.iter().zip(0..) {
        
        let dist = squared_euclidean(q, d);

        if dist < best_dist {
            best_dist = dist;
            best = (best_dist, i, d)
        }
    }

    best
}

#[cfg(feature = "do-not-return-position")]
fn brute_force<'d, const D: usize>(
    q: &[f64; D],
    data: &'d [[f64; D]],
) -> (f64, u64) {

    // No need for nan checks here
    let q: &[NotNan<f64>; D]= unsafe { std::mem::transmute(q) };
    let data: &'d [[NotNan<f64>; D]] = unsafe { std::mem::transmute(data) };

    let mut best_dist = std::f64::MAX;
    let mut best = (std::f64::MAX, std::u64::MAX);
    for (d, i) in data.iter().zip(0..) {
        
        let dist = squared_euclidean(q, d);

        if dist < best_dist {
            best_dist = dist;
            best = (best_dist, i)
        }
    }

    best
}