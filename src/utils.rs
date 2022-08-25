use ordered_float::NotNan;
use thiserror::Error;


pub type FnntwResult<T> = Result<T, FnntwError>;

pub(super) fn check_data<'d, const D: usize>(
    data: &'d [[f64; D]],
) -> FnntwResult<&'d [[NotNan<f64>; D]]> {

    // Cheap checks first
    // Nonzero Length
    if data.len() == 0 {
        return Err(FnntwError::ZeroLengthInputData);
    }

    for data_point in data {
        let _ = check_point(data_point)?;
    }

    // After all checks are formed, bitwise move the f64s into the same-size wrapper type
    // safety: just checked all the things that NotNan needs, and lifetime is not being transmuted
    Ok( unsafe { std::mem::transmute(data) })
}

/// Note: by explicity annotating lifetimes, we are ensuring that transmute does not modify lifetimes
/// and simply bitwise moves from f64 to NotNan<f64> after checks
pub(crate) fn check_point<'p, const D: usize>(
    point: &'p [f64; D]
) -> FnntwResult<&'p [NotNan<f64>; D]> {

    for component in point {
        // Check if invalid
        if component.is_nan() || component.is_infinite() || component.is_subnormal() {
            return Err(FnntwError::InvalidInputData { data_point: Box::from(*point) })
        }
    }

    // After all checks are formed, bitwise move the f64s into the same-size wrapper type
    // safety: just checked all the things that NotNan needs, and lifetime is not being transmuted
    Ok( unsafe { std::mem::transmute(point) })
}




#[derive(Debug, Error)]
pub enum FnntwError {

    #[error("Invalid input data was detected: {data_point:?}")]
    InvalidInputData {
        data_point: Box<[f64]>
    },

    #[error("Input data has zero length")]
    ZeroLengthInputData,

    #[error("Invalid boxsize: data does not fit in the specified box")]
    SmallBoxsize,

    #[error("Invalid boxsize: contains nan, inf, or subnormal float")]
    InvalidBoxsize,

    #[error("At least one of your data points has a negative component. \
             To use periodic queries, shift your data bounding box to start \
             at the origin")]
    NegativeDataPeriodicQuery,

}

