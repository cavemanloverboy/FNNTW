use std::collections::{BinaryHeap, HashMap};

use ordered_float::NotNan;

#[repr(transparent)]
#[derive(Debug)]
pub(crate) struct Candidate<'t, const D: usize>((f64, &'t[NotNan<f64>; D]));

#[cfg(feature = "binary-heap-container")]
pub(crate) struct Container<'t, const D: usize> {
    items: BinaryHeap<Candidate<'t, D>>,
    k: usize,
}

#[cfg(feature = "vec-container")]
pub(crate) struct Container<'t, const D: usize> {
    items: Vec<Candidate<'t, D>>,
    k: usize,
}

#[cfg(feature = "binary-heap-container")]
impl<'t, const D: usize> Container<'t,D> {
    pub(super) fn new(k: usize) -> Self {
        Container {
            items: BinaryHeap::new(),
            k,
        }
    }

    // Euclidean needs access to this one
    pub(crate) fn push(&mut self, neighbor: (f64, &'t [NotNan<f64>; D])) {
        if self.items.len() >= self.k {

            // If >=k elements, eject largest
            let neighbor: Candidate<D> = unsafe { std::mem::transmute(neighbor) };
            *self.items.peek_mut().unwrap() = neighbor;
        } else {

            // If less than k elements, add element.
            self.items.push( unsafe { std::mem::transmute(neighbor) });
        }
    }

    // Euclidean needs access to this one
    pub(crate) fn best_dist2(&self) -> &f64 {
        &self.items.peek().unwrap().0.0
    }

    pub(super) fn index<'i>(
        self,
        data_index: &HashMap<&'t [NotNan<f64>; D], u64>
    ) -> Vec<(f64, u64, &'i [NotNan<f64>; D])>
    where
        't: 'i,
    {
        let mut indexed: Vec<_> = self.items
            .into_iter()
            .map(|neighbor| {
                (neighbor.0.0, data_index[neighbor.0.1], neighbor.0.1)
            }).collect();
        indexed.sort_by(|a, b| a.0.partial_cmp(&b.0)
            .expect("encountered nan or inf"));
        indexed
    }
}

#[cfg(feature = "vec-container")]
impl<'t, const D: usize> Container<'t,D> {
    pub(super) fn new(k: usize) -> Self {
        Container {
            items: BinaryHeap::new(),
            k,
        }
    }

    // Euclidean needs access to this one
    pub(crate) fn push(&mut self, neighbor: (f64, &'t [NotNan<f64>; D])) {
        if self.items.len() >= self.k {

            // If >=k elements, eject largest
            let neighbor: Candidate<D> = unsafe { std::mem::transmute(neighbor) };
            *self.items.peek_mut().unwrap() = neighbor;
        } else {

            // If less than k elements, add element.
            self.items.push( unsafe { std::mem::transmute(neighbor) });
        }
    }

    // Euclidean needs access to this one
    pub(crate) fn best_dist2(&self) -> &f64 {
        &self.items.peek().unwrap().0.0
    }

    pub(super) fn index<'i>(
        self,
        data_index: &HashMap<&'t [NotNan<f64>; D], u64>
    ) -> Vec<(f64, u64, &'i [NotNan<f64>; D])>
    where
        't: 'i,
    {
        let mut indexed: Vec<_> = self.items
            .into_iter()
            .map(|neighbor| {
                (neighbor.0.0, data_index[neighbor.0.1], neighbor.0.1)
            }).collect();
        indexed.sort_by(|a, b| a.0.partial_cmp(&b.0)
            .expect("encountered nan or inf"));
        indexed
    }
}



impl<'t, const D: usize> PartialEq<Self> for Candidate<'t, D> {
    fn eq(&self, other: &Self) -> bool {
        self.0.0.eq(&other.0.0)
    }
}

impl<'t, const D: usize> PartialEq<(f64, &'t [NotNan<f64>; D])> for Candidate<'t, D> {
    fn eq(&self, other: &(f64, &'t[NotNan<f64>; D])) -> bool {
        self.0.0.eq(&other.0)
    }
}

impl<'t, const D: usize> Eq for Candidate<'t, D> { }

impl<'t, const D: usize> PartialOrd for Candidate<'t, D> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.0.partial_cmp(&other.0.0)
    }
}

impl<'t, const D: usize> PartialOrd<(f64, &'t [NotNan<f64>; D])> for Candidate<'t, D> {
    fn partial_cmp(&self, other: &(f64, &'t[NotNan<f64>; D])) -> Option<std::cmp::Ordering> {
        self.0.0.partial_cmp(&other.0)
    }
}

impl<'t, const D: usize> Ord for Candidate<'t, D> { 
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.0.partial_cmp(&other.0.0)
            .expect("Some NaN or Inf value was encountered")
    }
}


#[test]
fn test_transmute_neighbor() {

    // Define some point
    let point = unsafe { &[NotNan::new_unchecked(0.6); 3]};

    // Define some neighbor that refers to this point
    let neighbor = Candidate((0.3, &point));

    // Transmute neighbor using the same value
    let tn: (f64, &[NotNan<f64>; 3]) = unsafe { 
        std::mem::transmute(Candidate((0.3, &point))) 
    };

    // Check they are equal
    assert_eq!(
        neighbor,
        tn
    );

    // Check the pointers point to the same neighbor
    assert_eq!(
        *neighbor.0.1,
        *tn.1
    )
}


