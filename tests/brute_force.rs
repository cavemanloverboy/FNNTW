use fnntw::{distance::squared_euclidean, utils::QueryResult, Tree};
use ordered_float::NotNan;
use rand::{rngs::ThreadRng, Rng};
use std::error::Error;

const NDATA: usize = 100;
const NQUERY: usize = 1_000;
const D: usize = 3;

type T = f64;

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
    let tree = Tree::<'_, T, D>::new(&data, 1).unwrap();

    // Query tree
    let mut results = Vec::with_capacity(NQUERY);
    for q in &query {
        results.push(tree.query_nearest(q)?);
    }

    // Brute force check results
    for (i, q) in query.iter().enumerate() {
        assert_eq!(results[i], brute_force(q, &data), "u64 max is {}", u64::MAX)
    }

    Ok(())
}

fn random_point<const D: usize>(rng: &mut ThreadRng) -> [T; D] {
    [(); D].map(|_| rng.gen())
}

fn brute_force<'d, const D: usize>(q: &[T; D], data: &'d [[T; D]]) -> QueryResult<'d, T, D> {
    // No need for nan checks here
    let q: &[NotNan<T>; D] = unsafe { std::mem::transmute(q) };
    let data: &'d [[NotNan<T>; D]] = unsafe { std::mem::transmute(data) };

    let mut best_dist = T::MAX;
    let mut best = (
        T::MAX,
        std::u64::MAX,
        #[cfg(not(feature = "no-position"))]
        data.get(0).unwrap(),
    );
    for (d, i) in data.iter().zip(0..) {
        let dist = squared_euclidean(q, d);
        #[cfg(feature = "sqrt-dist2")]
        let dist = dist.sqrt();

        if dist < best_dist {
            best_dist = dist;
            best = (
                best_dist,
                i,
                #[cfg(not(feature = "no-position"))]
                d,
            );
        }
    }

    best
}
