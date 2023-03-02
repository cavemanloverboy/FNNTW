use std::{fmt::Debug, ops::AddAssign};

use ordered_float::{Float as ExternalFloat, NotNan};

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Point<T: Float, const D: usize> {
    pub(crate) ptr: *const [NotNan<T>; D],
}

impl<T: Float, const D: usize> Point<T, D> {
    /// SAFETY: This is only used for data which has been checked for correct dimensionality
    /// and which lives for 't

    pub(crate) unsafe fn get_unchecked<'t>(&self, i: usize) -> &'t NotNan<T> {
        (*self.ptr).get_unchecked(i)
    }

    /// SAFETY: This is only used for data which has been checked for correct dimensionality
    /// and which lives for 't

    pub(crate) unsafe fn position<'t>(&self) -> &[NotNan<T>; D] {
        &*self.ptr
    }

    /// SAFETY: This is only used for data which has been checked for correct dimensionality
    /// and which lives for 't. `start` must be the start of the data vector

    pub(crate) unsafe fn index<'t>(&self, start: *const [NotNan<T>; D]) -> u64 {
        // core::intrinsics::ptr_offset_from_unsigned(self.ptr, std::mem::transmute(start)) as u64
        self.ptr.offset_from(start) as u64
    }
}

pub trait Float: ExternalFloat + Debug + Send + Sync + AddAssign {}

impl<T: ExternalFloat + Debug + Send + Sync + AddAssign> Float for T {}

unsafe impl<T: Float, const D: usize> Send for Point<T, D> {}
unsafe impl<T: Float, const D: usize> Sync for Point<T, D> {}
