use std::collections::BinaryHeap;

#[cfg(feature = "sqrt-dist2")]
use crate::utils::process_dist2;
use crate::{
    point::{Float, Point},
    utils::QueryKResult,
    NotNan,
};

/// Using this struct to impl PartialOrd for T.
#[repr(transparent)]
#[derive(Clone)]
pub(crate) struct Candidate<'t, T: Float, const D: usize>((T, &'t Point<T, D>));

#[derive(Clone)]
pub struct Container<'t, T: Float, const D: usize> {
    items: BinaryHeap<Candidate<'t, T, D>>,
    k_or_datalen: usize,
}

impl<'t, T: Float, const D: usize> Container<'t, T, D> {
    pub fn new(k: usize) -> Self {
        Container {
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
    pub(crate) fn push(&mut self, neighbor: (T, &'t Point<T, D>)) {
        if self.items.len() >= self.k_or_datalen {
            // If >=k elements, eject largest
            let neighbor: Candidate<T, D> = Candidate(neighbor);
            // SAFETY: the length was just checked
            unsafe {
                *self.items.peek_mut().unwrap_unchecked() = neighbor;
            }
        } else {
            // If less than k elements, add element.
            self.items.push(Candidate(neighbor));
        }
    }

    // Euclidean needs access to this one
    pub(crate) fn best_dist2(&self) -> &T {
        &self.items.peek().unwrap().0 .0
    }

    #[allow(unused_mut)] // if sqrt-dist2 is on, mut is not used

    pub(super) fn index<'i>(&mut self, start: *const [NotNan<T>; D]) -> QueryKResult<'t, T, D>
    where
        't: 'i,
    {
        // Preallocate
        let mut result: QueryKResult<'t, T, D> = (
            Vec::with_capacity(self.k_or_datalen),
            Vec::with_capacity(self.k_or_datalen),
            #[cfg(not(feature = "no-position"))]
            Vec::with_capacity(self.k_or_datalen),
        );
        let ptrs = (
            result.0.as_mut_ptr(),
            result.1.as_mut_ptr(),
            #[cfg(not(feature = "no-position"))]
            {
                result.2.as_mut_ptr()
            },
        );

        // let mut idx = self.items.len();
        // while let Some(Candidate((dist2, neighbor))) = self.items.pop() {
        let mut idx = 0;
        for Candidate((mut dist2, neighbor)) in std::mem::take(&mut self.items).into_sorted_vec() {
            unsafe {
                *ptrs.0.add(idx) = process_dist2(dist2);
                *ptrs.1.add(idx) = neighbor.index(start);
                #[cfg(not(feature = "no-position"))]
                {
                    *ptrs.2.add(idx) = neighbor.position;
                }
            }
            idx += 1;
        }
        unsafe {
            result.0.set_len(self.k_or_datalen);
            result.1.set_len(self.k_or_datalen);
            #[cfg(not(feature = "no-position"))]
            result.2.set_len(self.k_or_datalen);
        }
        result
    }

    #[allow(unused_mut)] // if sqrt-dist2 is on, mut is not used
    #[allow(unused)]
    pub(super) fn index_with<'i>(
        mut self,
        start: *const [NotNan<T>; D],
    ) -> (QueryKResult<'t, T, D>, Self)
    where
        't: 'i,
    {
        // Preallocate
        let mut result: QueryKResult<'t, T, D> = (
            Vec::with_capacity(self.k_or_datalen),
            Vec::with_capacity(self.k_or_datalen),
            #[cfg(not(feature = "no-position"))]
            Vec::with_capacity(self.k_or_datalen),
        );
        let ptrs = (
            result.0.as_mut_ptr(),
            result.1.as_mut_ptr(),
            #[cfg(not(feature = "no-position"))]
            {
                result.2.as_mut_ptr()
            },
        );

        // let mut idx = self.items.len();
        // while let Some(Candidate((dist2, neighbor))) = self.items.pop() {
        let mut idx = 0;
        for Candidate((mut dist2, neighbor)) in std::mem::take(&mut self.items).into_sorted_vec() {
            unsafe {
                *ptrs.0.add(idx) = process_dist2(dist2);
                *ptrs.1.add(idx) = neighbor.index(start);
                #[cfg(not(feature = "no-position"))]
                {
                    *ptrs.2.add(idx) = neighbor.position;
                }
            }
            idx += 1;
        }
        unsafe {
            result.0.set_len(self.k_or_datalen);
            result.1.set_len(self.k_or_datalen);
            #[cfg(not(feature = "no-position"))]
            result.2.set_len(self.k_or_datalen);
        }
        (result, self)
    }

    pub(super) fn index_into<'i>(
        &mut self,
        distances_ptr: usize,
        indices_ptr: usize,
        query_index: usize,
        start: *const [NotNan<T>; D],
    ) where
        't: 'i,
    {
        let dptr = distances_ptr as *mut T;
        let iptr = indices_ptr as *mut u64;

        {
            let neighbors: Vec<_> = std::mem::take(&mut self.items).into_sorted_vec();
            let mut idx = 0;
            for Candidate((dist2, _)) in &neighbors {
                unsafe {
                    *dptr.add(query_index * self.k_or_datalen + idx) = process_dist2(*dist2);
                }
                idx += 1;
            }
            let mut idx = 0;
            for Candidate((_, neighbor)) in &neighbors {
                unsafe {
                    *iptr.add(query_index * self.k_or_datalen + idx) = neighbor.index(start);
                }
                idx += 1;
            }
            #[cfg(not(feature = "no-position"))]
            {
                let mut idx = 0;
                for Candidate((_, neighbor)) in &neighbors {
                    unsafe {
                        *ptrs.2.add(idx) = neighbor.position;
                    }
                    idx += 1;
                }
            }
        }
    }
}

impl<'t, T: Float, const D: usize> PartialEq<Self> for Candidate<'t, T, D> {
    fn eq(&self, other: &Self) -> bool {
        self.0 .0.eq(&other.0 .0)
    }
}

impl<'t, T: Float, const D: usize> PartialEq<(T, &'t [NotNan<T>; D])> for Candidate<'t, T, D> {
    fn eq(&self, other: &(T, &'t [NotNan<T>; D])) -> bool {
        self.0 .0.eq(&other.0)
    }
}

impl<'t, T: Float, const D: usize> Eq for Candidate<'t, T, D> {}

impl<'t, T: Float, const D: usize> PartialOrd for Candidate<'t, T, D> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0 .0.partial_cmp(&other.0 .0)
    }
}

impl<'t, T: Float, const D: usize> PartialOrd<(T, &'t [NotNan<T>; D])> for Candidate<'t, T, D> {
    fn partial_cmp(&self, other: &(T, &'t [NotNan<T>; D])) -> Option<std::cmp::Ordering> {
        self.0 .0.partial_cmp(&other.0)
    }
}

impl<'t, T: Float, const D: usize> Ord for Candidate<'t, T, D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
             .0
            .partial_cmp(&other.0 .0)
            .expect("Some NaN or Inf value was encountered")
    }
}
