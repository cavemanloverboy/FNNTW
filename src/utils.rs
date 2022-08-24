use ordered_float::NotNan;
use thiserror::Error;


pub type FnntwResult<'d, T> = Result<T, FnntwError<'d>>;

pub(super) fn check_data<'d, const D: usize>(data: &'d [[f64; D]]) -> FnntwResult<&'d [[NotNan<f64>; D]]> {

    // Cheap checks first
    // Nonzero Length
    if data.len() == 0 {
        return Err(FnntwError::ZeroLengthInputData);
    }

    for data_point in data {
        for component in data_point {

            // Check if invalid
            if component.is_nan() || component.is_infinite() || component.is_subnormal() {
                return Err(FnntwError::InvalidInputData { data_point })
            }
        }
    }

    // After all checks are formed, bitwise move the f64s into the same-size wrapper type
    Ok( unsafe { std::mem::transmute(data) })
}




#[derive(Debug, Error)]
pub enum FnntwError<'d> {

    #[error("Invalid input data was detected: {data_point:?}")]
    InvalidInputData {
        data_point: &'d [f64]
    },

    #[error("Input data has zero length")]
    ZeroLengthInputData,

}

