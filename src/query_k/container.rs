use std::collections::BinaryHeap;

use ordered_float::NotNan;

use crate::point::{Float, Point};

#[repr(transparent)]
/// Using this struct to impl PartialOrd for T.
pub(crate) struct Candidate<'t, T: Float, const D: usize>((T, &'t Point<'t, T, D>));

pub(crate) struct Container<'t, T: Float, const D: usize> {
    items: BinaryHeap<Candidate<'t, T, D>>,
    k: usize,
}

impl<'t, T: Float, const D: usize> Container<'t, T, D> {
    pub(super) fn new(k: usize) -> Self {
        Container {
            items: BinaryHeap::with_capacity(k),
            k,
        }
    }

    // Euclidean needs access to this one
    pub(crate) fn push(&mut self, neighbor: (T, &'t Point<'t, T, D>)) {
        if self.items.len() >= self.k {
            // If >=k elements, eject largest
            let neighbor: Candidate<T, D> = Candidate(neighbor);
            *self.items.peek_mut().unwrap() = neighbor;
        } else {
            // If less than k elements, add element.
            self.items.push(Candidate(neighbor));
        }
    }

    // Euclidean needs access to this one
    pub(crate) fn best_dist2(&self) -> &T {
        &self.items.peek().unwrap().0 .0
    }

    pub(super) fn index<'i>(self) -> Vec<(T, u64, &'i [NotNan<T>; D])>
    where
        't: 'i,
    {
        let mut indexed: Vec<_> = self
            .items
            .into_iter()
            .map(|neighbor| (neighbor.0 .0, neighbor.0 .1.index, neighbor.0 .1.position))
            .collect();
        indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("encountered nan or inf"));
        indexed
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
