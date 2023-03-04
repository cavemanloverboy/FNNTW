use std::{
    sync::atomic::{AtomicPtr, AtomicUsize},
    time::Instant,
};

use rayon::{
    prelude::{IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use crate::point::{Float, Point};

pub fn moms<T: Float, const D: usize>(
    slice: &mut [[T; D]],
    chunk_size: usize,
    axis: usize,
) -> (&mut [[T; D]], &mut [T; D], &mut [[T; D]]) {
    let medians_timer = Instant::now();
    let mut medians: Vec<&mut [T; D]> = slice
        .par_chunks_mut(chunk_size)
        .flat_map(|chunk| {
            if chunk.len() == 0 {
                return None;
            }
            println!("chunk = {:?}", chunk);
            let chunk_median_index = chunk.len() / 2;
            let (_, chunk_median, _) =
                chunk.select_nth_unstable_by(chunk_median_index, |a, b| unsafe {
                    a.get_unchecked(axis)
                        .partial_cmp(b.get_unchecked(axis))
                        .unwrap_unchecked()
                });
            Some(chunk_median)
        })
        .collect();
    println!(
        "medians took {} micros",
        medians_timer.elapsed().as_micros()
    );

    let moms_timer = Instant::now();
    let approx_index = medians.len() / 2;
    // Note this pattern matching makes a copy!
    // This is important because otherwise this element will shift!
    let &mut approx_median = if medians.len() == 1 {
        medians.pop().unwrap()
    } else {
        medians
            .select_nth_unstable_by(approx_index, |a, b| unsafe {
                a.get_unchecked(axis)
                    .partial_cmp(b.get_unchecked(axis))
                    .unwrap_unchecked()
            })
            .1
    };
    println!("moms took {} micros", moms_timer.elapsed().as_micros());

    let counter_timer = Instant::now();
    let left = AtomicUsize::new(0);
    let equal = AtomicUsize::new(0);
    use likely_stable::unlikely;
    use std::sync::atomic::Ordering;
    slice.into_par_iter().for_each(|s| unsafe {
        if unlikely(s.get_unchecked(axis) == approx_median.get_unchecked(axis)) {
            equal.fetch_add(1, Ordering::Relaxed);
        } else if s.get_unchecked(axis) < approx_median.get_unchecked(axis) {
            left.fetch_add(1, Ordering::Relaxed);
        }
    });
    println!(
        "counter took {} micros",
        counter_timer.elapsed().as_micros()
    );

    // let (tx, rx) = crossbeam_channel::bounded(100);
    // let rearrange = std::thread::

    let num_left = left.load(Ordering::Acquire);
    let num_eq = equal.load(Ordering::Acquire);
    let num_right = AtomicUsize::new(slice.len() - num_left - num_eq);
    // let left_ptr = slice.as_mut_ptr();
    // let eq_ptr = unsafe { slice.as_mut_ptr().add(num_left) };
    // let right_ptr = unsafe { slice.as_mut_ptr().add(num_left + num_eq) };

    let left_atomptr = AtomicPtr::new(slice.as_mut_ptr());
    let eq_atomptr =
        AtomicPtr::new(unsafe { slice.as_mut_ptr().add(left.load(Ordering::Acquire)) });
    let right_atomptr = AtomicPtr::new(unsafe {
        slice
            .as_mut_ptr()
            .add(left.load(Ordering::Acquire) + equal.load(Ordering::Acquire))
    });
    // for i in 0..num_left {
    //     unsafe {
    //         if unlikely(slice.get_unchecked(i).get_unchecked(axis) == approx_median.get_unchecked(axis)) {
    //             left_ptr.swap(with)
    //         } else if {
    //             s.
    //         }
    //     }
    // }
    let left_timer = Instant::now();
    (0..num_left).into_par_iter().for_each(|_| unsafe {
        let left_check_idx = left_atomptr
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |ptr| Some(ptr.add(1)))
            .unwrap();
        loop {
            if unlikely((*left_check_idx).get_unchecked(axis) == approx_median.get_unchecked(axis))
            {
                let eq_idx = eq_atomptr
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |ptr| Some(ptr.add(1)))
                    .unwrap();
                left_check_idx.swap(eq_idx);
            } else if (*left_check_idx).get_unchecked(axis) > approx_median.get_unchecked(axis) {
                let right_idx = right_atomptr
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |ptr| Some(ptr.add(1)))
                    .unwrap();
                num_right.fetch_sub(1, Ordering::Relaxed);
                left_check_idx.swap(right_idx);
            } else {
                break;
            }
        }
    });
    println!(
        "left arrange took {} micros",
        left_timer.elapsed().as_micros()
    );

    #[cfg(feature = "timing")]
    let right_timer = Instant::now();
    (0..num_right.load(Ordering::Relaxed))
        .into_par_iter()
        .for_each(|_| unsafe {
            let right_check_idx = right_atomptr
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |ptr| Some(ptr.add(1)))
                .unwrap();
            loop {
                if unlikely(
                    (*right_check_idx).get_unchecked(axis) == approx_median.get_unchecked(axis),
                ) {
                    let eq_idx = eq_atomptr
                        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |ptr| Some(ptr.add(1)))
                        .unwrap();
                    right_check_idx.swap(eq_idx);
                } else {
                    break;
                }
            }
        });
    println!(
        "right arrange took {} micros",
        left_timer.elapsed().as_micros()
    );

    let (left, remainder) = slice.split_at_mut(num_left);
    let (approx_median, right) = remainder.split_first_mut().unwrap();

    (left, approx_median, right)
}

pub fn moms_seq<T: Float, const D: usize>(
    // slice: &mut [[T; D]],
    slice: &mut [Point<T, D>],
    chunk_size: Option<usize>,
    axis: usize,
) -> (&mut [Point<T, D>], &mut Point<T, D>, &mut [Point<T, D>]) {
    if slice.len() < 100_000 {
        return slice.select_nth_unstable_by(slice.len() / 2, |a, b| unsafe {
            a.get_unchecked(axis)
                .partial_cmp(b.get_unchecked(axis))
                .unwrap_unchecked()
        });
    }
    #[cfg(feature = "timing")]
    let medians_timer = Instant::now();
    let chunk_size = chunk_size.unwrap_or(
        (slice.len() / (rayon::current_num_threads() * 4))
            .max(5)
            .min(250_000),
    );
    let mut medians: Vec<&mut Point<T, D>> = slice
        .par_chunks_mut(chunk_size)
        .map(|chunk| {
            let chunk_median_index = chunk.len() / 2;
            let (_, chunk_median, _) =
                chunk.select_nth_unstable_by(chunk_median_index, |a, b| unsafe {
                    a.get_unchecked(axis)
                        .partial_cmp(b.get_unchecked(axis))
                        .unwrap_unchecked()
                });
            chunk_median
        })
        .collect();
    #[cfg(feature = "timing")]
    println!(
        "medians took {} micros",
        medians_timer.elapsed().as_micros()
    );

    #[cfg(feature = "timing")]
    let moms_timer = Instant::now();
    let approx_index = medians.len() / 2;
    // Note this pattern matching makes a copy!
    // This is important because otherwise this element will shift!
    let approx_median = if medians.len() == 1 {
        *medians.pop().unwrap()
    } else {
        **medians
            .select_nth_unstable_by(approx_index, |a, b| unsafe {
                a.get_unchecked(axis)
                    .partial_cmp(b.get_unchecked(axis))
                    .unwrap_unchecked()
            })
            .1
    };
    #[cfg(feature = "timing")]
    println!(
        "moms took {} micros: {approx_median:?}",
        moms_timer.elapsed().as_micros()
    );

    #[cfg(feature = "timing")]
    let counter_timer = Instant::now();
    let mut left = 0;
    let mut equal = 0;
    use likely_stable::unlikely;
    for s in &*slice {
        unsafe {
            if unlikely(s.get_unchecked(axis) == approx_median.get_unchecked(axis)) {
                equal += 1;
            } else if s.get_unchecked(axis) < approx_median.get_unchecked(axis) {
                left += 1;
            }
        }
    }
    #[cfg(feature = "timing")]
    println!(
        "counter took {} micros",
        counter_timer.elapsed().as_micros()
    );

    let mut right = slice.len() - left - equal;
    let mut left_ptr = slice.as_mut_ptr();
    let mut eq_ptr = unsafe { slice.as_mut_ptr().add(left) };
    let mut right_ptr = unsafe { slice.as_mut_ptr().add(left + equal) };

    // for i in 0..num_left {
    //     unsafe {
    //         if unlikely(slice.get_unchecked(i).get_unchecked(axis) == approx_median.get_unchecked(axis)) {
    //             left_ptr.swap(with)
    //         } else if {
    //             s.
    //         }
    //     }
    // }
    #[cfg(feature = "timing")]
    let left_timer = Instant::now();
    for _ in 0..left {
        unsafe {
            loop {
                if unlikely((*left_ptr).get_unchecked(axis) == approx_median.get_unchecked(axis)) {
                    left_ptr.swap(eq_ptr);
                    eq_ptr = eq_ptr.add(1);
                } else if (*left_ptr).get_unchecked(axis) > approx_median.get_unchecked(axis) {
                    left_ptr.swap(right_ptr);
                    right_ptr = right_ptr.add(1);
                    right -= 1;
                } else {
                    break;
                }
            }
            left_ptr = left_ptr.add(1);
        }
    }
    #[cfg(feature = "timing")]
    println!(
        "left arrange took {} micros",
        left_timer.elapsed().as_micros()
    );

    #[cfg(feature = "timing")]
    let right_timer = Instant::now();
    for _ in 0..right {
        unsafe {
            loop {
                if unlikely((*right_ptr).get_unchecked(axis) == approx_median.get_unchecked(axis)) {
                    right_ptr.swap(eq_ptr);
                    eq_ptr = eq_ptr.add(1);
                } else {
                    break;
                }
            }
            right_ptr = right_ptr.add(1);
        }
    }
    #[cfg(feature = "timing")]
    println!(
        "right arrange took {} micros",
        right_timer.elapsed().as_micros()
    );

    let (left, remainder) = slice.split_at_mut(left);
    let (approx_median, right) = remainder.split_first_mut().unwrap();

    (left, approx_median, right)
}

// #[test]
// fn test_moms() {
//     let mut values: Vec<[f32; 1]> = (0..75).map(|_| [rand::random()]).collect();
//     let index = values.len() / 2;
//     let (_, &mut actual, _) = values.select_nth_unstable_by(index, |a, b| unsafe {
//         a.get_unchecked(0).partial_cmp(b.get_unchecked(0)).unwrap()
//     });
//     let (left, approx_median, right) = moms(values.as_mut_slice(), 5, 0);
//     for l in left {
//         assert!(l < approx_median);
//     }
//     for r in right {
//         assert!(r >= approx_median);
//     }
//     // This might fail on a rare occasion...
//     assert!(*approx_median > [0.2f32] && *approx_median < [0.8f32]);

//     println!("{approx_median:?} vs {actual:?}");
//     drop(approx_median);
//     println!("{values:?}");
// }

// #[test]
// fn test_moms_seq() {
//     let mut values: Vec<[f32; 1]> = (0..25).map(|_| [rand::random()]).collect();
//     let index = values.len() / 2;
//     let (_, &mut actual, _) = values.select_nth_unstable_by(index, |a, b| unsafe {
//         a.get_unchecked(0).partial_cmp(b.get_unchecked(0)).unwrap()
//     });
//     let (left, approx_median, right) = moms_seq(values.as_mut_slice(), 0);
//     for l in &*left {
//         assert!(l < approx_median);
//     }
//     for r in right {
//         assert!(r >= approx_median);
//     }
//     // This might fail on a rare occasion...
//     println!("{approx_median:?} vs {actual:?}: {}", left.len());
//     assert!(*approx_median > [0.2f32] && *approx_median < [0.8f32]);
//     drop(approx_median);
//     println!("{values:?}");
// }
