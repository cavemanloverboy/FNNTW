use fnntw::{distance::squared_euclidean, point::Float, utils::QueryKResult, Tree};
use ordered_float::NotNan;
use rand::{rngs::ThreadRng, Rng};
use std::error::Error;

const NDATA: usize = 100;
const NQUERY: usize = 1_000;
const BOXSIZE: [f64; 3] = [1.0; 3];
const D: usize = 3;
const K: usize = 80;

#[test]
fn test_brute_force_periodic_k() -> Result<(), Box<dyn Error>> {
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
    let tree = Tree::<'_, _, D>::new_parallel(&data, 1, 1)?.with_boxsize(&BOXSIZE)?;

    // Query tree
    let mut results = Vec::with_capacity(NQUERY);
    for q in &query {
        results.push(tree.query_nearest_k_noidx(q, K)?);
    }

    // Brute force check results
    for (i, q) in query.iter().enumerate() {
        let result = &results[i];
        let expected = brute_force_periodic_k(q, &data, K);
        assert_eq!(result.len(), K);
        assert_eq!(expected.0.len(), K);
        assert_eq!(results[i], expected.0);
    }

    Ok(())
}

fn random_point<const D: usize>(rng: &mut ThreadRng) -> [f64; D] {
    [(); D].map(|_| rng.gen())
}

fn brute_force_periodic_k<'d, T: Float, const D: usize>(
    q: &[T; D],
    data: &'d [[T; D]],
    k: usize,
) -> QueryKResult<'d, T, D> {
    // No need for nan checks here
    let q: &[NotNan<T>; D] = unsafe { std::mem::transmute(q) };
    let data: &'d [[NotNan<T>; D]] = unsafe { std::mem::transmute(data) };

    // Costly...
    let mut all = Vec::with_capacity(data.len() * 2_usize.pow(D as u32));

    for (d, i) in data.iter().zip(0..) {
        // Note this is 0..2^D here so that we can check all images incl real without
        // checking for which we don't need to do (as in the library)
        for image in 0..2_usize.pow(D as u32) {
            // Closest image in the form of bool array
            let closest_image = (0..D as u32).map(|idx| ((image / 2_usize.pow(idx)) % 2) == 1);

            let mut image_to_check = q.clone();

            for (idx, flag) in closest_image.enumerate() {
                // If moving image along this dimension
                if flag {
                    // Do a single index here. This is equal to distance to lower side
                    let query_component: &NotNan<T> = unsafe { q.get_unchecked(idx) };

                    // Single index here as well
                    let boxsize_component =
                        T::from(unsafe { *BOXSIZE.get_unchecked(idx) }).unwrap();

                    unsafe {
                        if **query_component < boxsize_component / T::from(2.0).unwrap() {
                            // Add if in lower half of box
                            *image_to_check.get_unchecked_mut(idx) =
                                query_component + boxsize_component
                        } else {
                            // Subtract if in upper half of box
                            *image_to_check.get_unchecked_mut(idx) =
                                query_component - boxsize_component
                        }
                    }
                }
            }

            let dist = squared_euclidean(&image_to_check, d);

            all.push((
                dist,
                i,
                #[cfg(not(feature = "no-position"))]
                d,
            ))
        }
    }

    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
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
