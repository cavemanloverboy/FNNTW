use std::fmt::Debug;

use crate::point::{Float, Point};
use ordered_float::NotNan;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use thiserror::Error;

pub type FnntwResult<R, T> = Result<R, FnntwError<T>>;
#[cfg(feature = "no-position")]
pub type QueryResult<'t, T, const D: usize> = (T, u64);
#[cfg(not(feature = "no-position"))]
pub type QueryResult<'t, T, const D: usize> = (T, u64, &'t [NotNan<T>; D]);

#[cfg(feature = "no-position")]
pub type QueryKResult<'t, T, const D: usize> = (Vec<T>, Vec<u64>);
#[cfg(not(feature = "no-position"))]
pub type QueryKResult<'t, T, const D: usize> = (Vec<T>, Vec<u64>, Vec<[NotNan<T>; D]>);

pub(super) fn check_data<'d, T: Float + Debug, const D: usize>(
    data: &'d [[T; D]],
) -> FnntwResult<Vec<Point<'d, T, D>>, T> {
    // Cheap checks first
    // Nonzero Length
    if data.len() == 0 {
        return Err(FnntwError::ZeroLengthInputData);
    }

    if data.len() < 1_000_000 {
        return Ok(data
            .into_iter()
            .zip(0..)
            .map(|(data_point, index)| -> FnntwResult<Point<'d, T, D>, T> {
                // Do check on point
                let _ = check_point(data_point)?;

                // After all checks are performed, bitwise move the Ts into the same-size wrapper type
                // safety: just checked all the things that NotNan needs, and lifetime is not being transmuted
                let position = unsafe { std::mem::transmute(data_point) };

                // Index point
                Ok(Point { position, index })
            })
            .collect::<FnntwResult<Vec<Point<'d, T, D>>, T>>()?);
    } else {
        return Ok(data
            .into_par_iter()
            .enumerate()
            .map(|(index, data_point)| -> FnntwResult<Point<'d, T, D>, T> {
                // Do check on point
                let _ = check_point(data_point)?;

                // After all checks are performed, bitwise move the Ts into the same-size wrapper type
                // safety: just checked all the things that NotNan needs, and lifetime is not being transmuted
                let position = unsafe { std::mem::transmute(data_point) };

                // Index point
                Ok(Point {
                    position,
                    index: index as u64,
                })
            })
            .collect::<FnntwResult<Vec<Point<'d, T, D>>, T>>()?);
    }
}

/// Note: by explicity annotating lifetimes, we are ensuring that transmute does not modify lifetimes
/// and simply bitwise moves from T to NotNan<T> after checks
pub(crate) fn check_point<'p, T: Float + Debug, const D: usize>(
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

    #[error(
        "At least one of your data points has a negative component. \
             To use periodic queries, shift your data bounding box to start \
             at the origin"
    )]
    NegativeDataPeriodicQuery,
}

#[cfg(feature = "sqrt-dist2")]
#[inline(always)]
pub(crate) fn process_result<'t, T: Float, const D: usize>(
    mut result: QueryResult<'t, T, D>,
) -> QueryResult<'t, T, D> {
    result.0 = result.0.sqrt();
    result
}

#[cfg(not(feature = "sqrt-dist2"))]
#[inline(always)]
pub(crate) fn process_result<'t, T: Float, const D: usize>(
    result: QueryResult<'t, T, D>,
) -> QueryResult<'t, T, D> {
    result
}

#[cfg(feature = "sqrt-dist2")]
#[inline(always)]
pub(crate) fn process_dist2<T: Float>(dist2: &mut T) {
    *dist2 = dist2.sqrt();
}
