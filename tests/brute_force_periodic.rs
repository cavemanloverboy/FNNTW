use fnntw::{distance::squared_euclidean, Tree};
use ordered_float::NotNan;
use rand::{rngs::ThreadRng, Rng};

use std::error::Error;

const NDATA: usize = 100;
const NQUERY: usize = 10_000;
const D: usize = 3;
const BOXSIZE: [f64; D] = [1.0; D];

#[test]
fn test_brute_force_periodic() -> Result<(), Box<dyn Error>> {
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
    let tree = Tree::<'_, D>::new(&data, 1)?.with_boxsize(&BOXSIZE)?;

    // Query tree
    let mut results = Vec::with_capacity(NQUERY);
    for q in &query {
        results.push(tree.query_nearest(q)?);
    }

    // Brute force check results
    for (i, q) in query.iter().enumerate() {
        assert_eq!(results[i], brute_force_periodic(q, &data))
    }

    Ok(())
}

fn random_point<const D: usize>(rng: &mut ThreadRng) -> [f64; D] {
    [(); D].map(|_| rng.gen())
}

#[cfg(not(feature = "do-not-return-position"))]
fn brute_force_periodic<'d, const D: usize>(
    q: &[f64; D],
    data: &'d [[f64; D]],
) -> (f64, u64, &'d [NotNan<f64>; D]) {
    // No need for nan checks here
    let q: &[NotNan<f64>; D] = unsafe { std::mem::transmute(q) };
    let data: &'d [[NotNan<f64>; D]] = unsafe { std::mem::transmute(data) };

    let mut best_dist = std::f64::MAX;
    let mut best = (std::f64::MAX, std::u64::MAX, data.get(0).unwrap());

    for (d, i) in data.iter().zip(0..) {
        // Note this is 0..2^D here so that we can check all images incl real without
        // checking for which we don't need to do (as in the library)
        for image in 0..2_usize.pow(D as u32) {
            // Closest image in the form of bool array
            let closest_image = (0..D).map(|idx| ((image / 2_usize.pow(idx as u32)) % 2) == 1);

            let mut image_to_check = q.clone();

            for (idx, flag) in closest_image.enumerate() {
                // If moving image along this dimension
                if flag {
                    // Do a single index here. This is equal to distance to lower side
                    let query_component: &NotNan<f64> = unsafe { q.get_unchecked(idx) };

                    // Single index here as well
                    let boxsize_component = unsafe { BOXSIZE.get_unchecked(idx) };

                    unsafe {
                        if **query_component < boxsize_component / 2.0 {
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

            if dist < best_dist {
                best_dist = dist;
                best = (best_dist, i, d)
            }
        }
    }

    best
}

#[cfg(feature = "do-not-return-position")]
fn brute_force_periodic<'d, const D: usize>(q: &[f64; D], data: &'d [[f64; D]]) -> (f64, u64) {
    // No need for nan checks here
    let q: &[NotNan<f64>; D] = unsafe { std::mem::transmute(q) };
    let data: &'d [[NotNan<f64>; D]] = unsafe { std::mem::transmute(data) };

    let mut best_dist = std::f64::MAX;
    let mut best = (std::f64::MAX, std::u64::MAX);

    for (d, i) in data.iter().zip(0..) {
        // Note this is 0..2^D here so that we can check all images incl real without
        // checking for which we don't need to do (as in the library)
        for image in 0..2_usize.pow(D as u32) {
            // Closest image in the form of bool array
            let closest_image = (0..D).map(|idx| ((image / 2_usize.pow(idx as u32)) % 2) == 1);

            let mut image_to_check = q.clone();

            for (idx, flag) in closest_image.enumerate() {
                // If moving image along this dimension
                if flag {
                    // Do a single index here. This is equal to distance to lower side
                    let query_component: &NotNan<f64> = unsafe { q.get_unchecked(idx) };

                    // Single index here as well
                    let boxsize_component = unsafe { BOXSIZE.get_unchecked(idx) };

                    unsafe {
                        if **query_component < boxsize_component / 2.0 {
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

            if dist < best_dist {
                best_dist = dist;
                best = (best_dist, i)
            }
        }
    }

    best
}
