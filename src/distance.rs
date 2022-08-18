use ordered_float::NotNan;

#[inline(always)]
pub fn squared_euclidean<const D: usize>(
    a: &[NotNan<f64>; D],
    b: &[NotNan<f64>; D],
) -> f64 {
    
    // Initialize accumulator var
    let mut dist_sq: f64 = 0.0;

    for idx in 0..D {
        unsafe {
            dist_sq += (a.get_unchecked(idx) - b.get_unchecked(idx)).powi(2);
        }
    }

    // unsafe { NotNan::new_unchecked(dist_sq) }
    dist_sq
}

#[inline(always)]
pub fn squared_euclidean_sep<const D: usize>(
    a: &[NotNan<f64>; D],
    b: &[NotNan<f64>; D],
) -> NotNan<f64> {

    // Initialize diff array
    let mut diff = [0.0_f64; D];

    for idx in 0..D {
        unsafe {
            *diff.get_unchecked_mut(idx) = *(a.get_unchecked(idx) - b.get_unchecked(idx));
        }
    }

    unsafe { NotNan::new_unchecked(diff.iter().map(|x| x*x).sum()) }
}


#[inline(always)]
pub fn new_best<'a, 'b, 'c, const D: usize>(
    query: &'c [NotNan<f64>; D],
    candidate: &'b [NotNan<f64>; D],
    current_best_dist_sq: &'a mut f64,
    current_best_candidate: &'a mut &'b [NotNan<f64>; D]
) -> bool {

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
    current_best_candidate: &'a mut &'b [NotNan<f64>; D]
) -> bool {

    debug_assert!(*current_best_dist_sq >= 0.0, "distance must be nonnegative");

    // Initialize accumulator var
    let mut dist_sq = 0.0;

    for idx in 0..D {
        unsafe {

            // Add to accumulator variable
            dist_sq += (query.get_unchecked(idx)-candidate.get_unchecked(idx)).powi(2);
        }

        // short circuit
        if dist_sq >= *current_best_dist_sq { return false }
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