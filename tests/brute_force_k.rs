use fnntw::{distance::squared_euclidean, point::Float, utils::QueryKResult, Tree};
use ordered_float::NotNan;
use rand::{rngs::ThreadRng, Rng};
use std::error::Error;

const NDATA: usize = 100;
const NQUERY: usize = 1_000;
const D: usize = 3;
const K: usize = 80;

#[test]
fn test_brute_force_k() -> Result<(), Box<dyn Error>> {
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
    let tree = Tree::<'_, _, D>::new_parallel(&data, 1, 1).unwrap();

    // Query tree
    let mut results = Vec::with_capacity(NQUERY);
    for q in &query {
        results.push(tree.query_nearest_k(q, K)?);
    }

    // Brute force check results
    for (i, q) in query.iter().enumerate() {
        let result = &results[i];
        let expected = brute_force_k(q, &data, K);
        assert_eq!(result.0.len(), K);
        assert_eq!(expected.0.len(), K);
        assert_eq!(*result, expected);
        assert_eq!(*result, expected);
        assert_eq!(*result, expected);
    }

    Ok(())
}

fn random_point<const D: usize>(rng: &mut ThreadRng) -> [f64; D] {
    [(); D].map(|_| rng.gen())
}

fn brute_force_k<'d, T: Float, const D: usize>(
    q: &[T; D],
    data: &'d [[T; D]],
    k: usize,
) -> QueryKResult<'d, T, D> {
    // No need for nan checks here
    let q: &[NotNan<T>; D] = unsafe { std::mem::transmute(q) };
    let data: &'d [[NotNan<T>; D]] = unsafe { std::mem::transmute(data) };

    let mut all = Vec::with_capacity(data.len());

    for (d, i) in data.iter().zip(0..) {
        let dist = squared_euclidean::<T, D>(q, d);
        all.push((
            dist,
            i,
            #[cfg(not(feature = "no-position"))]
            d,
        ))
    }

    // this is safe so long as [0, 1] randoms are used
    all.sort_by(|p1, p2| p1.0.partial_cmp(&p2.0).unwrap());
    #[cfg(feature = "sqrt-dist2")]
    all.iter_mut().for_each(|p| {
        p.0 = p.0.sqrt();
    });
    all.truncate(k);

    #[cfg(feature = "no-position")]
    return all.into_iter().unzip();

    #[cfg(not(feature = "no-position"))]
    {
        let mut result = (vec![], vec![], vec![]);
        for a in all {
            result.0.push(a[0]);
            result.1.push(a[1]);
            result.2.push(a[2]);
        }
    }
}
