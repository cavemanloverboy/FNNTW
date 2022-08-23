use ordered_float::NotNan;

use crate::query_k::container::Container;

#[inline(always)]
pub fn squared_euclidean<const D: usize>(a: &[NotNan<f64>; D], b: &[NotNan<f64>; D]) -> f64 {
    // Initialize accumulator var
    let mut dist_sq: f64 = 0.0;

    for idx in 0..D {
        unsafe {
            dist_sq += (a.get_unchecked(idx) - b.get_unchecked(idx)).powi(2);
        }
    }

    dist_sq
}

#[inline(always)]
pub fn squared_euclidean_sep<const D: usize>(a: &[NotNan<f64>; D], b: &[NotNan<f64>; D]) -> f64 {
    // Initialize diff array
    let mut diff = [0.0_f64; D];

    for idx in 0..D {
        unsafe {
            *diff.get_unchecked_mut(idx) = *(a.get_unchecked(idx) - b.get_unchecked(idx));
        }
    }

    diff.iter().map(|x| x * x).sum()
}

#[inline(always)]
pub fn new_best<'i, 'o, const D: usize>(
    query: &[NotNan<f64>; D],
    candidate: &'i [NotNan<f64>; D],
    current_best_dist_sq: &'o mut f64,
    current_best_candidate: &'o mut &'i [NotNan<f64>; D],
) -> bool
where
    'i: 'o,
{
    debug_assert!(*current_best_dist_sq >= 0.0, "distance must be nonnegative");

    // Run usual squared_euclidean fn
    let dist_sq: f64 = squared_euclidean(query, candidate);

    // Compare squared dist
    if dist_sq < *current_best_dist_sq {
        // New best branch
        *current_best_dist_sq = dist_sq;
        *current_best_candidate = candidate;
        true
    } else {
        false
    }
}

#[inline(always)]
pub(crate) fn new_best_kth<'t, 'i, 'o, const D: usize>(
    query: &[NotNan<f64>; D],
    candidate: &'i [NotNan<f64>; D],
    container: &'o mut Container<'i, D>,
) -> bool
where
    'i: 'o,
    't: 'i
{

    // Run usual squared_euclidean fn
    let dist_sq: f64 = squared_euclidean(query, candidate);

    // Compare squared dist
    let new_best = dist_sq < *container.best_dist2();
    if new_best {
        container.push((dist_sq, candidate));
        true
    } else {
        false
    }
}



/// Calculate the distance from `query` to some space defined by `lower` and `upper`.
///
/// This function constructs a point (neither in the tree nor is it `query`) that
/// represents the point in the space defined by `lower` and `upper` that is closest
/// to `query`. Then, it computes the squared euclidean distance between the two.
pub fn calc_dist_sq_to_space<const D: usize>(
    query: &[NotNan<f64>; D],
    lower: &[NotNan<f64>; D],
    upper: &[NotNan<f64>; D],
) -> f64 {
    // Initilalize a point
    let mut closest_point = [unsafe { NotNan::new_unchecked(0.0_f64) }; D];

    for i in 0..D {
        unsafe {
            *closest_point.get_unchecked_mut(i) = *query
                .get_unchecked(i)
                .min(upper.get_unchecked(i))
                .max(lower.get_unchecked(i));
        }
    }

    // Calculate squared euclidean distance between this point and the query
    squared_euclidean(query, &closest_point)
}

/// This uses a short circuiting squared euclidean comparison.
///
/// For example, in 3D if `(dx*dx + dy*dy) > current_best_squared`
/// is true, then `(dx*dx + dy*dy + dz*dz) > current_best_squared`
/// is true, so we don't need to calculate `dz*dz` and add it to
/// the accumulator variable before doing a comparison. This does
/// more comparisons but less aritmetic.
///
/// This was innovative, but unfortunately not better...
#[inline(always)]
pub fn new_best_short<'a, 'b, 'c, const D: usize>(
    query: &'c [NotNan<f64>; D],
    candidate: &'b [NotNan<f64>; D],
    current_best_dist_sq: &'a mut f64,
    current_best_candidate: &'a mut &'b [NotNan<f64>; D],
) -> bool {
    debug_assert!(*current_best_dist_sq >= 0.0, "distance must be nonnegative");

    // Initialize accumulator var
    let mut dist_sq = 0.0;

    for idx in 0..D {
        unsafe {
            // Add to accumulator variable
            dist_sq += (query.get_unchecked(idx) - candidate.get_unchecked(idx)).powi(2);
        }

        // short circuit
        if dist_sq >= *current_best_dist_sq {
            return false;
        }
    }

    // The short circuit includes the full dist_sq calculation,
    // and so if you reach this point then the squared euclidean dist
    // is indeed lower than the current best.
    //
    // I.e. new best was found
    *current_best_dist_sq = dist_sq;
    *current_best_candidate = candidate;
    true
}

#[cfg(test)]
mod tests {

    use crate::distance::squared_euclidean;
    use approx_eq::assert_approx_eq;
    use ordered_float::NotNan;

    use super::calc_dist_sq_to_space;

    #[test]
    fn test_squared_euclidean() {
        use approx_eq::assert_approx_eq;

        unsafe {
            let a = [NotNan::new_unchecked(1.0), NotNan::new_unchecked(1.0)];
            let b = [NotNan::new_unchecked(0.0), NotNan::new_unchecked(0.0)];

            assert_approx_eq!(squared_euclidean(&a, &b), 2.0);
        }
    }

    #[test]
    fn test_calc_dist_to_space_above() {
        unsafe {
            let query = &[NotNan::new_unchecked(2.0); 3];
            let lower = &[NotNan::new_unchecked(0.0); 3];
            let upper = &[NotNan::new_unchecked(1.0); 3];

            assert_approx_eq!(calc_dist_sq_to_space(query, lower, upper), 3.0);
        }
    }

    #[test]
    fn test_calc_dist_to_space_below() {
        unsafe {
            let query = &[NotNan::new_unchecked(-1.0); 3];
            let lower = &[NotNan::new_unchecked(0.0); 3];
            let upper = &[NotNan::new_unchecked(1.0); 3];

            assert_approx_eq!(calc_dist_sq_to_space(query, lower, upper), 3.0);
        }
    }

    #[test]
    fn test_calc_dist_to_space_within() {
        unsafe {
            let query = &[NotNan::new_unchecked(0.5); 3];
            let lower = &[NotNan::new_unchecked(0.0); 3];
            let upper = &[NotNan::new_unchecked(1.0); 3];

            assert_approx_eq!(calc_dist_sq_to_space(query, lower, upper), 0.0);
        }
    }
}
