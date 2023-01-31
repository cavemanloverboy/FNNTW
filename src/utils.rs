use std::fmt::Debug;

use ordered_float::NotNan;
use thiserror::Error;

use crate::point::{Float, Point};

pub type FnntwResult<R, T> = Result<R, FnntwError<T>>;

pub(super) fn check_data<'d, T: Float + Debug, const D: usize>(
    data: &'d [[T; D]],
) -> FnntwResult<Vec<Point<'d, T, D>>, T> {
    // Cheap checks first
    // Nonzero Length
    if data.len() == 0 {
        return Err(FnntwError::ZeroLengthInputData);
    }

    #[cfg(not(feature = "parallel"))]
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

    #[cfg(feature = "parallel")]
    return Ok(data
        .into_par_iter()
        .enumerate()
        .map(|(index, data_point)| -> Result<Point<'d, T, D>> {
            // Do check on point
            let _ = check_point(data_point)?;

            // After all checks are performed, bitwise move the Ts into the same-size wrapper type
            // safety: just checked all the things that NotNan needs, and lifetime is not being transmuted
            let position = unsafe { std::mem::transmute(data_point) };

            // Index point
            Ok(Point { position, index })
        })
        .collect::<Result<Vec<Point<'d, T, D>>>>()?);
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
