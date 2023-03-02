use std::ops::AddAssign;

use ordered_float::NotNan;

use crate::{
    point::{Float, Point},
    query_k::{container::Container, container_axis::ContainerAxis},
};

pub fn squared_euclidean<T: Float, const D: usize>(a: &[NotNan<T>; D], b: &[NotNan<T>; D]) -> T
where
    T: AddAssign,
{
    // Initialize accumulator var
    let mut dist_sq: T = T::zero();

    for idx in 0..D {
        // safety: made safe by const generic
        unsafe {
            dist_sq += (a.get_unchecked(idx) - b.get_unchecked(idx)).powi(2);
        }
    }

    dist_sq
}

pub fn squared_euclidean_axis<T: Float, const D: usize>(
    a: &[NotNan<T>; D],
    b: &[NotNan<T>; D],
    axis: usize,
) -> (T, T, T)
where
    T: AddAssign + PartialEq,
{
    // Initialize accumulator vars
    let mut ax: T = T::zero();
    let mut nonax: T = T::zero();

    for idx in 0..D {
        // safety: made safe by const generic
        if idx == axis {
            unsafe {
                ax += (a.get_unchecked(idx) - b.get_unchecked(idx)).powi(2);
            }
        } else {
            unsafe {
                nonax += (a.get_unchecked(idx) - b.get_unchecked(idx)).powi(2);
            }
        }
    }

    (ax + nonax, ax, nonax)
}

pub fn new_best<'t, 'i, 'o, T: Float, const D: usize>(
    query: &[NotNan<T>; D],
    candidate: &'i Point<T, D>,
    current_best_dist_sq: &'o mut T,
    current_best_candidate: &'o mut &'i Point<T, D>,
) -> bool
where
    'i: 'o,
    't: 'i,
{
    debug_assert!(
        *current_best_dist_sq >= T::zero(),
        "distance must be nonnegative"
    );

    // Run usual squared_euclidean fn
    let dist_sq: T = squared_euclidean(query, unsafe { candidate.position() });

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

pub(crate) fn new_best_kth<'t, 'i, 'o, T: Float, const D: usize>(
    query: &[NotNan<T>; D],
    candidate: &'i Point<T, D>,
    container: &'o mut Container<'i, T, D>,
) where
    'i: 'o,
    't: 'i,
{
    // Run usual squared_euclidean fn
    let dist_sq: T = squared_euclidean(query, unsafe { candidate.position() });

    // Compare squared dist
    let new_best = dist_sq <= *container.best_dist2();
    if new_best {
        container.push((dist_sq, candidate));
    }
}

pub(crate) fn new_best_kth_axis<'t, 'i, 'o, T: Float, const D: usize>(
    query: &[NotNan<T>; D],
    candidate: &'i Point<T, D>,
    container: &'o mut ContainerAxis<'i, T, D>,
    axis: usize,
) where
    'i: 'o,
    't: 'i,
{
    // Run usual squared_euclidean fn
    let (dist2, ax, nonax) = squared_euclidean_axis(query, unsafe { candidate.position() }, axis);

    // Compare squared dist
    let new_best = dist2 <= *container.best_dist2();
    if new_best {
        container.push(((dist2, ax, nonax), candidate));
    }
}

/// Calculate the distance from `query` to some space defined by `lower` and `upper`.
///
/// This function constructs a point (neither in the tree nor is it `query`) that
/// represents the point in the space defined by `lower` and `upper` that is closest
/// to `query`. Then, it computes the squared euclidean distance between the two.

pub fn calc_dist_sq_to_space<T: Float, const D: usize>(
    query: &[NotNan<T>; D],
    lower: &[NotNan<T>; D],
    upper: &[NotNan<T>; D],
) -> T {
    // Initialize a point
    // safety: 0.0 is always not nan
    let mut closest_point = [unsafe { NotNan::new_unchecked(T::zero()) }; D];

    for i in 0..D {
        // safety: made safe by const generic
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

pub fn new_best_short<'a, 'b, 'c, T: Float, const D: usize>(
    query: &'c [NotNan<T>; D],
    candidate: &'b [NotNan<T>; D],
    current_best_dist_sq: &'a mut T,
    current_best_candidate: &'a mut &'b [NotNan<T>; D],
) -> bool {
    debug_assert!(
        *current_best_dist_sq >= T::zero(),
        "distance must be nonnegative"
    );

    // Initialize accumulator var
    let mut dist_sq = T::zero();

    for idx in 0..D {
        unsafe {
            // Add to accumulator variable
            // safety: made safe by const generic
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

        let a = [NotNan::new(1.0).unwrap(), NotNan::new(1.0).unwrap()];
        let b = [NotNan::new(0.0).unwrap(), NotNan::new(0.0).unwrap()];

        assert_approx_eq!(squared_euclidean(&a, &b), 2.0);
    }

    #[test]
    fn test_calc_dist_to_space_above() {
        let query = &[NotNan::new(2.0).unwrap(); 3];
        let lower = &[NotNan::new(0.0).unwrap(); 3];
        let upper = &[NotNan::new(1.0).unwrap(); 3];

        assert_approx_eq!(calc_dist_sq_to_space(query, lower, upper), 3.0);
    }

    #[test]
    fn test_calc_dist_to_space_below() {
        let query = &[NotNan::new(-1.0).unwrap(); 3];
        let lower = &[NotNan::new(0.0).unwrap(); 3];
        let upper = &[NotNan::new(1.0).unwrap(); 3];

        assert_approx_eq!(calc_dist_sq_to_space(query, lower, upper), 3.0);
    }

    #[test]
    fn test_calc_dist_to_space_within() {
        let query = &[NotNan::new(0.5).unwrap(); 3];
        let lower = &[NotNan::new(0.0).unwrap(); 3];
        let upper = &[NotNan::new(1.0).unwrap(); 3];

        assert_approx_eq!(calc_dist_sq_to_space(query, lower, upper), 0.0);
    }
}
