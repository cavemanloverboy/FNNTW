use std::{fmt::Debug, ops::AddAssign};

use ordered_float::{Float as ExternalFloat, NotNan};

#[derive(PartialEq, Debug, Clone, Copy)]
// TODO: should this really be copy?
pub struct Point<'t, T: Float, const D: usize> {
    pub index: u64,
    pub position: &'t [NotNan<T>; D],
}

impl<'t, T: Float, const D: usize> Point<'t, T, D> {
    /// SAFETY: it is the callers responsibility to ensure i < D
    pub unsafe fn get_unchecked(&self, i: usize) -> &'t NotNan<T> {
        self.position.get_unchecked(i)
    }
}

pub trait Float: ExternalFloat + Debug + Send + Sync + AddAssign {}

impl<T: ExternalFloat + Debug + Send + Sync + AddAssign> Float for T {}
