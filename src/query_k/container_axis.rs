use std::collections::BinaryHeap;

#[cfg(feature = "sqrt-dist2")]
use crate::utils::process_dist2;
use crate::{
    point::{Float, Point},
    NotNan,
};

/// Using this struct to impl PartialOrd for T.
#[repr(transparent)]
#[derive(Clone)]
pub(crate) struct CandidateAxis<'t, T: Float, const D: usize>(((T, T, T), &'t Point<T, D>));

#[derive(Clone)]
pub struct ContainerAxis<'t, T: Float, const D: usize> {
    items: BinaryHeap<CandidateAxis<'t, T, D>>,
    k_or_datalen: usize,
}

impl<'t, T: Float, const D: usize> ContainerAxis<'t, T, D> {
    pub fn new(k: usize) -> Self {
        ContainerAxis {
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
    pub(crate) fn push(&mut self, neighbor: ((T, T, T), &'t Point<T, D>)) {
        if self.items.len() >= self.k_or_datalen {
            // If >=k elements, eject largest
            let neighbor: CandidateAxis<T, D> = CandidateAxis(neighbor);
            // SAFETY: the length was just checked
            unsafe {
                *self.items.peek_mut().unwrap_unchecked() = neighbor;
            }
        } else {
            // If less than k elements, add element.
            self.items.push(CandidateAxis(neighbor));
        }
    }

    // Euclidean needs access to this one
    // Note this is best kth, not best 1NN
    pub(crate) fn best_dist2(&self) -> &T {
        &self.items.peek().unwrap().0 .0 .0
    }

    // #[allow(unused_mut)] // if sqrt-dist2 is on, mut is not used
    // pub(super) fn index<'i>(&mut self, start: *const [NotNan<T>; D]) -> QueryKAxisResult<'t, T, D>
    // where
    //     't: 'i,
    // {
    //     // Preallocate
    //     let mut result: QueryKAxisResult<'t, T, D> = (
    //         Vec::with_capacity(self.k_or_datalen),
    //         Vec::with_capacity(self.k_or_datalen),
    //         #[cfg(not(feature = "no-index"))]
    //         Vec::with_capacity(self.k_or_datalen),
    //         #[cfg(not(feature = "no-position"))]
    //         Vec::with_capacity(self.k_or_datalen),
    //     );
    //     let ptrs = (
    //         result.0.as_mut_ptr(),
    //         result.1.as_mut_ptr(),
    //         #[cfg(not(feature = "no-position"))]
    //         {
    //             result.2.as_mut_ptr()
    //         },
    //     );

    //     // let mut idx = self.items.len();
    //     // while let Some(Candidate((dist2, neighbor))) = self.items.pop() {
    //     let mut idx = 0;
    //     for CandidateAxis(((_, ax, nonax), _neighbor)) in
    //         std::mem::take(&mut self.items).into_sorted_vec()
    //     {
    //         unsafe {
    //             *ptrs.0.add(idx) = process_dist2(ax);
    //             *ptrs.1.add(idx) = process_dist2(nonax);
    //             #[cfg(not(feature = "no-index"))]
    //             {
    //                 *ptrs.2.add(idx) = _neighbor.index(start);
    //             }
    //             #[cfg(not(feature = "no-position"))]
    //             {
    //                 *ptrs.3.add(idx) = _neighbor.position;
    //             }
    //         }
    //         idx += 1;
    //     }
    //     unsafe {
    //         result.0.set_len(self.k_or_datalen);
    //         result.1.set_len(self.k_or_datalen);
    //         #[cfg(not(feature = "no-position"))]
    //         result.2.set_len(self.k_or_datalen);
    //     }
    //     result
    // }

    #[allow(unused_mut)] // if sqrt-dist2 is on, mut is not used

    // pub(super) fn index_with<'i>(
    //     mut self,
    //     start: *const [NotNan<T>; D],
    // ) -> (QueryKAxisResult<'t, T, D>, Self)
    // where
    //     't: 'i,
    // {
    //     // Preallocate
    //     let mut result: QueryKAxisResult<'t, T, D> = (
    //         Vec::with_capacity(self.k_or_datalen),
    //         Vec::with_capacity(self.k_or_datalen),
    //         #[cfg(not(feature = "no-index"))]
    //         Vec::with_capacity(self.k_or_datalen),
    //         #[cfg(not(feature = "no-position"))]
    //         Vec::with_capacity(self.k_or_datalen),
    //     );
    //     let ptrs = (
    //         result.0.as_mut_ptr(),
    //         result.1.as_mut_ptr(),
    //         #[cfg(not(feature = "no-position"))]
    //         {
    //             result.2.as_mut_ptr()
    //         },
    //     );

    //     // let mut idx = self.items.len();
    //     // while let Some(Candidate((dist2, neighbor))) = self.items.pop() {
    //     let mut idx = 0;
    //     for CandidateAxis(((_, ax, nonax), neighbor)) in
    //         std::mem::take(&mut self.items).into_sorted_vec()
    //     {
    //         unsafe {
    //             *ptrs.0.add(idx) = process_dist2(ax);
    //             *ptrs.1.add(idx) = process_dist2(nonax);
    //             #[cfg(not(feature = "no-index"))]
    //             {
    //                 *ptrs.2.add(idx) = neighbor.index(start);
    //             }
    //             #[cfg(not(feature = "no-position"))]
    //             {
    //                 *ptrs.3.add(idx) = neighbor.position;
    //             }
    //         }
    //         idx += 1;
    //     }
    //     unsafe {
    //         result.0.set_len(self.k_or_datalen);
    //         result.1.set_len(self.k_or_datalen);
    //         #[cfg(not(feature = "no-index"))]
    //         result.2.set_len(self.k_or_datalen);
    //         #[cfg(not(feature = "no-position"))]
    //         result.3.set_len(self.k_or_datalen);
    //     }
    //     (result, self)
    // }

    // TODO: position
    pub(super) fn index_into<'i>(
        &mut self,
        ax_ptr: usize,
        nonax_ptr: usize,
        #[cfg(not(feature = "no-index"))] indices_ptr: usize,
        query_index: usize,
        #[cfg(not(feature = "no-index"))] start: *const [NotNan<T>; D],
    ) where
        't: 'i,
    {
        let axptr = ax_ptr as *mut T;
        let nonaxptr = nonax_ptr as *mut T;
        #[cfg(not(feature = "no-index"))]
        let iptr = indices_ptr as *mut u64;

        {
            let neighbors: Vec<_> = std::mem::take(&mut self.items).into_sorted_vec();
            let mut idx = 0;
            for CandidateAxis(((_d2, ax, _nonax), _p)) in &neighbors {
                unsafe {
                    *axptr.add(query_index * self.k_or_datalen + idx) = process_dist2(*ax);
                }
                idx += 1;
            }
            let mut idx = 0;
            for CandidateAxis(((_d2, _ax, nonax), _p)) in &neighbors {
                unsafe {
                    *nonaxptr.add(query_index * self.k_or_datalen + idx) = process_dist2(*nonax);
                }
                idx += 1;
            }
            #[cfg(not(feature = "no-index"))]
            {
                let mut idx = 0;
                for CandidateAxis((_, neighbor)) in &neighbors {
                    unsafe {
                        *iptr.add(query_index * self.k_or_datalen + idx) = neighbor.index(start);
                    }
                    idx += 1;
                }
            }
            #[cfg(not(feature = "no-position"))]
            {
                let mut idx = 0;
                for CandidateAxis((_, neighbor)) in &neighbors {
                    unsafe {
                        *ptrs.2.add(idx) = neighbor.position;
                    }
                    idx += 1;
                }
            }
        }
    }
}

impl<'t, T: Float, const D: usize> PartialEq<Self> for CandidateAxis<'t, T, D> {
    fn eq(&self, other: &Self) -> bool {
        self.0 .0 .0.eq(&other.0 .0 .0)
    }
}

impl<'t, T: Float, const D: usize> PartialEq<((T, T, T), &'t [NotNan<T>; D])>
    for CandidateAxis<'t, T, D>
{
    fn eq(&self, other: &((T, T, T), &'t [NotNan<T>; D])) -> bool {
        self.0 .0 .0.eq(&other.0 .0)
    }
}

impl<'t, T: Float, const D: usize> Eq for CandidateAxis<'t, T, D> {}

impl<'t, T: Float, const D: usize> PartialOrd for CandidateAxis<'t, T, D> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0 .0 .0.partial_cmp(&other.0 .0 .0)
    }
}

impl<'t, T: Float, const D: usize> PartialOrd<((T, T, T), &'t [NotNan<T>; D])>
    for CandidateAxis<'t, T, D>
{
    fn partial_cmp(&self, other: &((T, T, T), &'t [NotNan<T>; D])) -> Option<std::cmp::Ordering> {
        self.0 .0 .0.partial_cmp(&other.0 .0)
    }
}

impl<'t, T: Float, const D: usize> Ord for CandidateAxis<'t, T, D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
             .0
             .0
            .partial_cmp(&other.0 .0 .0)
            .expect("Some NaN or Inf value was encountered")
    }
}
