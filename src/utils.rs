use std::fmt::Debug;

use crate::point::Float;
use ordered_float::NotNan;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use thiserror::Error;

pub type FnntwResult<R, T> = Result<R, FnntwError<T>>;
pub type QueryResult<'t, T, const D: usize> = (T, u64);
#[cfg(not(feature = "no-position"))]
pub type QueryResult<'t, T, const D: usize> = (T, u64, &'t [NotNan<T>; D]);

#[cfg(not(feature = "no-index"))]
pub type QueryKResult<'t, T, const D: usize> = (Vec<T>, Vec<u64>);
#[cfg(not(feature = "no-position"))]
pub type QueryKResult<'t, T, const D: usize> = (Vec<T>, Vec<u64>, Vec<[NotNan<T>; D]>);

pub type QueryKAxisResult<'t, T, const D: usize> = (Vec<T>, Vec<T>);

pub(super) fn check_data<'d, T: Float + Debug, const D: usize>(
    data: &'d [[T; D]],
) -> FnntwResult<(), T> {
    data.into_par_iter().try_for_each(check_point)
}

/// Note: by explicity annotating lifetimes, we are ensuring that transmute does not modify lifetimes
/// and simply bitwise moves from T to NotNan<T> after checks
pub(crate) fn check_point_return<'p, T: Float + Debug, const D: usize>(
    point: &'p [T; D],
) -> FnntwResult<&'p [NotNan<T>; D], T> {
    for component in point {
        // Check if invalid
        if component.is_nan() || component.is_infinite() {
            return Err(FnntwError::InvalidInputData {
                data_point: Box::from(*point),
            });
        }
    }

    // After all checks are formed, bitwise move the Ts into the same-size wrapper type
    // safety: just checked all the things that NotNan needs, and lifetime is not being transmuted
    Ok(unsafe { std::mem::transmute(point) })
}

/// Note: by explicity annotating lifetimes, we are ensuring that transmute does not modify lifetimes
/// and simply bitwise moves from T to NotNan<T> after checks
pub(crate) fn check_point<'p, T: Float + Debug, const D: usize>(
    point: &'p [T; D],
) -> FnntwResult<(), T> {
    for component in point {
        // Check if invalid
        if component.is_nan() || component.is_infinite() {
            return Err(FnntwError::InvalidInputData {
                data_point: Box::from(*point),
            });
        }
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum FnntwError<T: Float + Debug> {
    #[error("Invalid input data was detected: {data_point:?}")]
    InvalidInputData { data_point: Box<[T]> },

    #[error("Input data has zero length")]
    ZeroLengthInputData,

    #[error("Invalid boxsize: data does not fit in the specified box")]
    SmallBoxsize,

    #[error("Invalid boxsize: contains nan, inf, or subnormal float")]
    InvalidBoxsize,

    #[error("Requested an axis that does not exist (incorrect dimensionality)")]
    InvalidAxis,

    #[error(
        "At least one of your data points has a negative component. \
             To use periodic queries, shift your data bounding box to start \
             at the origin"
    )]
    NegativeDataPeriodicQuery,
}

#[cfg(feature = "sqrt-dist2")]

pub(crate) fn process_result<'t, T: Float, const D: usize>(
    mut result: QueryResult<'t, T, D>,
) -> QueryResult<'t, T, D> {
    result.0 = result.0.sqrt();
    result
}

#[cfg(feature = "sqrt-dist2")]

pub(crate) fn process_dist2<T: Float>(dist2: T) -> T {
    dist2.sqrt()
}

#[cfg(not(feature = "sqrt-dist2"))]

pub(crate) fn process_dist2<T: Float>(dist2: T) -> T {
    dist2
}
