use std::collections::BinaryHeap;

#[cfg(feature = "sqrt-dist2")]
use crate::utils::process_dist2;
use crate::{
    point::{Float, Point},
    utils::QueryKResult,
    NotNan,
};

#[derive(Clone)]
pub struct ContainerNoIndex<T: Float + Sized, const D: usize> {
    items: BinaryHeap<NotNan<T>>,
    k_or_datalen: usize,
}

impl<T: Float + Sized, const D: usize> ContainerNoIndex<T, D> {
    pub fn new(k: usize) -> Self {
        ContainerNoIndex {
            items: BinaryHeap::with_capacity(k),
            k_or_datalen: k,
        }
    }

    #[allow(unused)]
    pub(super) fn check(&self, k_or_datalen: usize) -> bool {
        self.k_or_datalen == k_or_datalen
    }

    // Euclidean needs access to this one
    // The caller of this function has already done a dist2 <= max_dist2 check
    // This should never be NaN...
    pub(crate) fn push(&mut self, neighbor: T) {
        if self.items.len() >= self.k_or_datalen {
            unsafe {
                *self.items.peek_mut().unwrap_unchecked() = NotNan::new_unchecked(neighbor);
            }
        } else {
            // If less than k elements, add element.
            unsafe {
                self.items.push(NotNan::new_unchecked(neighbor));
            }
        }
    }

    // Euclidean needs access to this one
    pub(crate) fn best_dist2(&self) -> &T {
        self.items.peek().unwrap()
    }

    pub(super) fn finish<'i>(self) -> Vec<T> {
        // Preallocate
        let mut result: Vec<T> = Vec::with_capacity(self.k_or_datalen);
        let ptr = result.as_mut_ptr();

        // let mut idx = self.items.len();
        // while let Some(Candidate((dist2, neighbor))) = self.items.pop() {
        let mut idx = 0;
        for dist2 in self.items.into_sorted_vec() {
            unsafe {
                *ptr.add(idx) = process_dist2(*dist2);
            }
            idx += 1;
        }
        unsafe {
            result.set_len(self.k_or_datalen);
        }
        result
    }
}
