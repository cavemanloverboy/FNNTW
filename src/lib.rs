#![doc = include_str!("../README.md")]

use ordered_float::NotNan;

#[cfg(feature = "timing")]
use std::sync::atomic::Ordering;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

#[cfg(feature = "timing")]
use num_format::{Locale, ToFormattedString};

pub mod distance;
pub mod query;
pub mod query_k;
mod utils;

use utils::*;

// mod medians;
#[cfg(feature = "timing")]
static INITIAL_VEC_REF: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "timing")]
static LEAF_VEC_ALLOC: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "timing")]
static LEAF_WRITE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "timing")]
static STEM_MEDIAN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "timing")]
static STEM_WRITE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "timing")]
static mut TOTAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

const FIRST: bool = true;
const NOT_FIRST: bool = false;

const IS_LEFT: bool = true;
const IS_RIGHT: bool = false;

/// This [`Tree`] struct is the core struct that holds all nodes in the kdtree.
/// It also contains the hashmap that is used to index the data.
pub struct Tree<'t, const D: usize> {

    /// Data in the tree. Here for user reference mainly. For example,
    /// to inspect the data that was used to build the tree.
    #[allow(unused)]
    input: &'t [[f64; D]],

    // O(1) lookup table for index of point
    pub data_index: HashMap<&'t [NotNan<f64>; D], u64>,

    // Unused, but here for future/user reference
    #[allow(unused)]
    pub leafsize: usize,

    // Container of all nodes (stems, leaves), in the tree, except the root node.
    pub nodes: Vec<Node<'t, D>>,

    // Approximate height (used for determining some allocation sizes)
    pub height_hint: usize,

    // Root node
    root_node: Node<'t, D>,
}

#[derive(Debug)]
pub enum Node<'t, const D: usize> {
    Stem {
        split_dim: usize,
        point: &'t [NotNan<f64>; D],
        left: usize,
        right: usize,
        lower: [NotNan<f64>; D],
        upper: [NotNan<f64>; D],
    },
    Leaf {
        points: Leaf<'t, D>,
        lower: [NotNan<f64>; D],
        upper: [NotNan<f64>; D],
    },
}

impl<'t, const D: usize> Node<'t, D> {
    fn is_stem(&self) -> bool {
        match self {
            Node::Stem { .. } => true,
            Node::Leaf { .. } => false,
        }
    }

    #[cfg(not(feature = "single_ref"))]
    /// This was here just to test performance vs single ref.
    /// double ref seems to win (this one is double ref)
    fn iter<'q>(&'q self) -> impl Iterator<Item = &'q &'t [NotNan<f64>; D]> {
        match self {
            Node::Leaf { points, .. } => points.iter(),
            _ => unreachable!("this function should only be used on leaves"),
        }
    }

    #[cfg(feature = "single_ref")]
    /// This was here just to test performance vs double ref.
    /// double ref seems to win
    fn iter(&'t self) -> impl Iterator<Item = &'t [NotNan<f64>; D]> {
        match self {
            Node::Leaf { points, .. } => points.clone().into_iter(),
            _ => unreachable!("this function should only be used on leaves"),
        }
    }

    fn stem_position(&self) -> &'t [NotNan<f64>; D] {
        match self {
            Node::Stem {
                point: position, ..
            } => position,
            _ => unreachable!("only to be used on stems"),
        }
    }

    // fn stem_position(&self) -> &'t [NotNan<f64>; D] {
    //     match self {
    //         Node::Stem { split_dim: .. } => position,
    //         _ => unreachable!("only to be used on stems"),
    //     }
    // }

    fn get_bounds(&'t self) -> (&'t [NotNan<f64>; D], &'t [NotNan<f64>; D]) {
        match self {
            Node::Leaf { lower, upper, .. } => (lower, upper),
            Node::Stem { lower, upper, .. } => (lower, upper),
        }
    }
}

type Leaf<'t, const D: usize> = Vec<&'t [NotNan<f64>; D]>;

impl<'t, const D: usize> Tree<'t, D> {
    /// Create a new FNSTW kdTree [Tree] using a parallel build. The parameter
    /// `par_split_level` specified the depth (during the build) at which the
    /// parallelism begins.
    pub fn new_parallel(
        input: &'t [[f64; D]],
        leafsize: usize,
        par_split_level: usize,
    ) -> FnntwResult<Tree<'t, D>> {

        // Perform checks for valid data
        let data: &'t [[NotNan<f64>; D]] = check_data(input)?;

        // This is used to determine the size several allocations
        //
        // TODO: There exists a way to get the exact height during build,
        // but that will require a refactor that I don't want to do just yet
        // and i'm not convinced it will be that big of a performance boost
        let data_len = input.len();
        let height_hint = ilog2(data_len);

        // Unsafe operations require some min leafsize
        // Also probably a good idea to keep above 4 anyway.
        // if leafsize < 4 {
        //     return Err("Choose a leafsize >= 4");
        // }

        // Initialize variables for recursive function
        let split_level: usize = 0;
        #[cfg(feature = "timing")]
        let timer = std::time::Instant::now();
        let vec_ref: &mut [&'t [NotNan<f64>; D]] = &mut data.iter().collect::<Vec<_>>();
        // let data_index_map = data.iter().zip(0_u64..).collect::<HashMap<_,_>>();
        #[cfg(feature = "timing")]
        let initial_vec_ref = timer.elapsed().as_nanos();
        let nodes = Arc::new(RwLock::new(Vec::with_capacity(size_of_tree(data_len, leafsize))));

        // Build index in the background
        std::thread::scope(|s| {
            let data_idx_handle =
                s.spawn(move || data.iter().zip(0_u64..).collect::<HashMap<_, _>>());

            // Run recursive build
            Tree::<'t, D>::build_nodes_parallel::<FIRST, true>(
                vec_ref,
                split_level,
                leafsize,
                Arc::clone(&nodes),
                None,
                None,
                par_split_level,
            );

            #[cfg(feature = "timing")]
            {
                // safe because no other thread can hold this mutable reference
                unsafe {
                    *TOTAL.get_mut() = initial_vec_ref as usize
                        + LEAF_VEC_ALLOC.load(Ordering::SeqCst)
                        + LEAF_WRITE.load(Ordering::SeqCst)
                        + STEM_MEDIAN.load(Ordering::SeqCst)
                        + STEM_WRITE.load(Ordering::SeqCst);
                }

                // Load atomics
                let total = TOTAL.load(Ordering::SeqCst);
                let leaf_write = LEAF_WRITE.load(Ordering::SeqCst);
                let leaf_vec_alloc = LEAF_VEC_ALLOC.load(Ordering::SeqCst);
                let stem_median = STEM_MEDIAN.load(Ordering::SeqCst);
                let stem_write = STEM_WRITE.load(Ordering::SeqCst);

                // Time elapsed strs
                let total_str = total.to_formatted_string(&Locale::en);
                let ivr_str = initial_vec_ref.to_formatted_string(&Locale::en);
                let leaf_write_str = leaf_write.to_formatted_string(&Locale::en);
                let leaf_vec_alloc_str = leaf_vec_alloc.to_formatted_string(&Locale::en);
                let stem_median_str = stem_median.to_formatted_string(&Locale::en);
                let stem_write_str = stem_write.to_formatted_string(&Locale::en);

                // Frac strs
                let ivr_frac_str = format!("{:.2}", 100.0 * initial_vec_ref as f64 / total as f64);
                let leaf_write_frac_str =
                    format!("{:.2}", 100.0 * leaf_write as f64 / total as f64);
                let leaf_vec_alloc_frac_str =
                    format!("{:.2}", 100.0 * leaf_vec_alloc as f64 / total as f64);
                let stem_median_frac_str =
                    format!("{:.2}", 100.0 * stem_median as f64 / total as f64);
                let stem_write_frac_str =
                    format!("{:.2}", 100.0 * stem_write as f64 / total as f64);

                println!("\nINITIAL_VEC_REF = {} nanos, {}%", ivr_str, ivr_frac_str);
                println!(
                    "LEAF_VEC_ALLOC = {} nanos, {}%",
                    leaf_vec_alloc_str, leaf_vec_alloc_frac_str
                );
                println!(
                    "LEAF_WRITE = {} nanos {}%",
                    leaf_write_str, leaf_write_frac_str
                );
                println!(
                    "STEM_MEDIAN = {} nanos, {}%",
                    stem_median_str, stem_median_frac_str
                );
                println!(
                    "STEM_WRITE = {} nanos, {}%",
                    stem_write_str, stem_write_frac_str
                );
                println!("TOTAL = {}\n", total_str);
            }

            // Unwrap the Arc
            let mut nodes = Arc::try_unwrap(nodes).unwrap().into_inner().unwrap();
            let root_node = nodes.pop().expect("root node should exist");

            Ok(Tree {
                input,
                data_index: data_idx_handle.join().unwrap(),
                leafsize,
                nodes,
                height_hint,
                root_node,
            })
        })
    }

    // A recursive private function.
    fn build_nodes_parallel<'a, const F: bool, const L: bool>(
        subset: &'a mut [&'t [NotNan<f64>; D]],
        mut split_level: usize,
        leafsize: usize,
        nodes: Arc<RwLock<Vec<Node<'t, D>>>>,
        level_up_bounds: Option<([NotNan<f64>; D], [NotNan<f64>; D])>,
        level_up_split_val: Option<&'t NotNan<f64>>,
        par_split_level: usize,
    ) -> usize {
        // Increment split level if not first
        if !F {
            split_level += 1
        };

        // Get split dimension
        let split_dim = split_level % D;

        // Determine leaf-ness
        let is_leaf = subset.len() <= leafsize;

        // Get space bounds
        let (lower, upper) = {
            if F {
                // If this is the first iteration, we must find the bounds of the data before median
                subset.iter().fold(
                    unsafe {
                        (
                            [NotNan::new_unchecked(std::f64::MAX); D],
                            [NotNan::new_unchecked(std::f64::MIN); D],
                        )
                    },
                    |(mut lo, mut hi), point| {
                        for idx in 0..D {
                            unsafe {
                                let lo_idx = lo.get_unchecked_mut(idx);
                                *lo_idx = (*lo_idx).min(*point.get_unchecked(idx));

                                let hi_idx = hi.get_unchecked_mut(idx);
                                *hi_idx = (*hi_idx).max(*point.get_unchecked(idx));
                            }
                        }
                        (lo, hi)
                    },
                )
            } else {
                // If not the first iteration, get bounds from parent and modify
                // the split_dim component
                let (mut lo, mut hi) = level_up_bounds
                    .map(|(lo, hi)| (lo.clone(), hi.clone()))
                    .unwrap();

                // Modify parent split_dim component
                let parent_split_dim = (split_level - 1) % D;
                unsafe {
                    if L {
                        // If we are left, then our upper bound got cut off
                        *hi.get_unchecked_mut(parent_split_dim) = *level_up_split_val.unwrap();
                    } else {
                        // If we are right, then our lower bound got cut off
                        *lo.get_unchecked_mut(parent_split_dim) = *level_up_split_val.unwrap();
                    }
                }
                (lo, hi)
            }
        };

        match is_leaf {
            true => {

                #[cfg(feature = "timing")]
                let timer = std::time::Instant::now();
                let leaf = Node::Leaf {
                    points: subset.to_vec(),
                    lower,
                    upper,
                };
                #[cfg(feature = "timing")]
                let vec_alloc = timer.elapsed().as_nanos();
                #[cfg(feature = "timing")]
                LEAF_VEC_ALLOC.fetch_add(vec_alloc as usize, Ordering::SeqCst);

                #[cfg(feature = "timing")]
                let timer = std::time::Instant::now();
                let mut node_lock = nodes.write().unwrap();
                let leaf_index = node_lock.len();
                node_lock.push(leaf);
                #[cfg(feature = "timing")]
                let write_time = timer.elapsed().as_nanos();
                #[cfg(feature = "timing")]
                LEAF_WRITE.fetch_add(write_time as usize, Ordering::SeqCst);

                leaf_index
            }
            false => {
                // Calculate index of median
                let sub_len = subset.len();
                let median_index = sub_len / 2 - ((sub_len + 1) % 2);

                #[cfg(feature = "timing")]
                let timer = std::time::Instant::now();
                // Select median in this subset based on split_dim component
                let (left, median, right) = subset
                    .select_nth_unstable_by(median_index, |a, b| unsafe {
                        a.get_unchecked(split_dim).cmp(&b.get_unchecked(split_dim))
                    });
                let split_val = unsafe { median.get_unchecked(split_dim) };
                #[cfg(feature = "timing")]
                let stem_median = timer.elapsed().as_nanos();
                #[cfg(feature = "timing")]
                STEM_MEDIAN.fetch_add(stem_median as usize, Ordering::SeqCst);

                let mut left_idx = 0;
                let mut right_idx = 0;
                if split_level == par_split_level && par_split_level > 0 {
                    std::thread::scope(|s| {

                        // We are at the level over which user has specified
                        // we should parallelize the build. Handle in scoped threads
                        let left_arc = Arc::clone(&nodes);
                        let right_arc = Arc::clone(&nodes);
                        let left_handle = s.spawn(|| {
                            Tree::build_nodes_parallel::<NOT_FIRST, IS_LEFT>(
                                left,
                                split_level,
                                leafsize,
                                left_arc,
                                Some((lower, upper)),
                                Some(split_val),
                                par_split_level,
                            )
                        });

                        let right_handle = s.spawn(|| {
                            Tree::build_nodes_parallel::<NOT_FIRST, IS_RIGHT>(
                                right,
                                split_level,
                                leafsize,
                                right_arc,
                                Some((lower, upper)),
                                Some(split_val),
                                par_split_level,
                            )
                        });

                        left_idx = left_handle.join().unwrap();
                        right_idx = right_handle.join().unwrap();
                    });
                } else {
                    left_idx = Tree::build_nodes_parallel::<NOT_FIRST, IS_LEFT>(
                        left,
                        split_level,
                        leafsize,
                        Arc::clone(&nodes),
                        Some((lower, upper)),
                        Some(split_val),
                        par_split_level,
                    );

                    right_idx = Tree::build_nodes_parallel::<NOT_FIRST, IS_RIGHT>(
                        right,
                        split_level,
                        leafsize,
                        Arc::clone(&nodes),
                        Some((lower, upper)),
                        Some(split_val),
                        par_split_level,
                    );
                }

                let stem = Node::Stem {
                    split_dim,
                    point: median,
                    left: left_idx,
                    right: right_idx,
                    lower,
                    upper,
                };

                #[cfg(feature = "timing")]
                let timer = std::time::Instant::now();
                let mut node_lock = nodes.write().unwrap();
                let stem_index = node_lock.len();
                node_lock.push(stem);
                drop(node_lock);
                #[cfg(feature = "timing")]
                let stem_write = timer.elapsed().as_nanos();
                #[cfg(feature = "timing")]
                STEM_WRITE.fetch_add(stem_write as usize, Ordering::SeqCst);

                stem_index
            }
        }
    }

    /// Create a new FNSTW kdTree [Tree] using a nonparallel build.
    pub fn new(
        input: &'t [[f64; D]],
        leafsize: usize,
    ) -> FnntwResult<Tree<'t, D>> {
        
        // Perform checks for valid data
        let data: &'t [[NotNan<f64>; D]] = check_data(input)?;

        // This is used to determine the size several allocations
        //
        // TODO: There exists a way to get the exact height during build,
        // but that will require a refactor that I don't want to do just yet
        // and i'm not convinced it will be that big of a performance boost
        let data_len = input.len();
        let height_hint = ilog2(data_len);

        // Initialize variables for recursive function
        let split_level: usize = 0;
        #[cfg(feature = "timing")]
        let timer = std::time::Instant::now();
        let vec_ref: &mut [&'t [NotNan<f64>; D]] = &mut data.iter().collect::<Vec<_>>();
        // let data_index_map = data.iter().zip(0_u64..).collect::<HashMap<_,_>>();
        #[cfg(feature = "timing")]
        let initial_vec_ref = timer.elapsed().as_nanos();
        let mut nodes = vec![];

        // Build index in the background
        std::thread::scope(|s| {
            let data_idx_handle =
                s.spawn(move || data.iter().zip(0_u64..).collect::<HashMap<_, _>>());

            // Run recursive build
            Tree::<'t, D>::build_nodes::<FIRST, true>(
                vec_ref,
                split_level,
                leafsize,
                &mut nodes,
                None,
                None,
            );

            #[cfg(feature = "timing")]
            {
                // safe because no other thread can hold this mutable reference
                unsafe {
                    *TOTAL.get_mut() = initial_vec_ref as usize
                        + LEAF_VEC_ALLOC.load(Ordering::SeqCst)
                        + LEAF_WRITE.load(Ordering::SeqCst)
                        + STEM_MEDIAN.load(Ordering::SeqCst)
                        + STEM_WRITE.load(Ordering::SeqCst);
                }

                // Load atomics
                let total = TOTAL.load(Ordering::SeqCst);
                let leaf_write = LEAF_WRITE.load(Ordering::SeqCst);
                let leaf_vec_alloc = LEAF_VEC_ALLOC.load(Ordering::SeqCst);
                let stem_median = STEM_MEDIAN.load(Ordering::SeqCst);
                let stem_write = STEM_WRITE.load(Ordering::SeqCst);

                // Time elapsed strs
                let total_str = total.to_formatted_string(&Locale::en);
                let ivr_str = initial_vec_ref.to_formatted_string(&Locale::en);
                let leaf_write_str = leaf_write.to_formatted_string(&Locale::en);
                let leaf_vec_alloc_str = leaf_vec_alloc.to_formatted_string(&Locale::en);
                let stem_median_str = stem_median.to_formatted_string(&Locale::en);
                let stem_write_str = stem_write.to_formatted_string(&Locale::en);

                // Frac strs
                let ivr_frac_str = format!("{:.2}", 100.0 * initial_vec_ref as f64 / total as f64);
                let leaf_write_frac_str =
                    format!("{:.2}", 100.0 * leaf_write as f64 / total as f64);
                let leaf_vec_alloc_frac_str =
                    format!("{:.2}", 100.0 * leaf_vec_alloc as f64 / total as f64);
                let stem_median_frac_str =
                    format!("{:.2}", 100.0 * stem_median as f64 / total as f64);
                let stem_write_frac_str =
                    format!("{:.2}", 100.0 * stem_write as f64 / total as f64);

                println!("\nINITIAL_VEC_REF = {} nanos, {}%", ivr_str, ivr_frac_str);
                println!(
                    "LEAF_VEC_ALLOC = {} nanos, {}%",
                    leaf_vec_alloc_str, leaf_vec_alloc_frac_str
                );
                println!(
                    "LEAF_WRITE = {} nanos {}%",
                    leaf_write_str, leaf_write_frac_str
                );
                println!(
                    "STEM_MEDIAN = {} nanos, {}%",
                    stem_median_str, stem_median_frac_str
                );
                println!(
                    "STEM_WRITE = {} nanos, {}%",
                    stem_write_str, stem_write_frac_str
                );
                println!("TOTAL = {}\n", total_str);
            }

            let root_node: Node<D> = nodes.pop().expect("root node should exist");

            Ok(Tree {
                input,
                data_index: data_idx_handle.join().unwrap(),
                leafsize,
                nodes,
                height_hint,
                root_node
            })
        })
    }

    // A recursive private function.
    fn build_nodes<'a, const F: bool, const L: bool>(
        subset: &'a mut [&'t [NotNan<f64>; D]],
        mut split_level: usize,
        leafsize: usize,
        nodes: &mut Vec<Node<'t, D>>,
        level_up_bounds: Option<([NotNan<f64>; D], [NotNan<f64>; D])>,
        level_up_split_val: Option<&'t NotNan<f64>>,
    ) -> usize {
        // Increment split level if not first
        if !F {
            split_level += 1
        };

        // Get split dimension
        let split_dim = split_level % D;

        // Determine leaf-ness
        let is_leaf = subset.len() <= leafsize;

        // Get space bounds
        let (lower, upper) = {
            if F {
                // If this is the first iteration, we must find the bounds of the data before median
                subset.iter().fold(
                    unsafe {
                        (
                            [NotNan::new_unchecked(std::f64::MAX); D],
                            [NotNan::new_unchecked(std::f64::MIN); D],
                        )
                    },
                    |(mut lo, mut hi), point| {
                        for idx in 0..D {
                            unsafe {
                                let lo_idx = lo.get_unchecked_mut(idx);
                                *lo_idx = (*lo_idx).min(*point.get_unchecked(idx));

                                let hi_idx = hi.get_unchecked_mut(idx);
                                *hi_idx = (*hi_idx).max(*point.get_unchecked(idx));
                            }
                        }
                        (lo, hi)
                    },
                )
            } else {
                // If not the first iteration, get bounds from parent and modify
                // the split_dim component
                let (mut lo, mut hi) = level_up_bounds
                    .map(|(lo, hi)| (lo.clone(), hi.clone()))
                    .unwrap();

                // Modify parent split_dim component
                let parent_split_dim = (split_level - 1) % D;
                unsafe {
                    if L {
                        // If we are left, then our upper bound got cut off
                        *hi.get_unchecked_mut(parent_split_dim) = *level_up_split_val.unwrap();
                    } else {
                        // If we are right, then our lower bound got cut off
                        *lo.get_unchecked_mut(parent_split_dim) = *level_up_split_val.unwrap();
                    }
                }
                (lo, hi)
            }
        };

        match is_leaf {
            true => {
                #[cfg(feature = "timing")]
                let timer = std::time::Instant::now();
                // let mut lower = [(); D].map(|_| unsafe { NotNan::new_unchecked(std::f64::MAX) });
                // let mut upper = [(); D].map(|_| unsafe { NotNan::new_unchecked(std::f64::MIN) });
                // for i in 0..D {
                //     for p in 0..subset.len() {
                //         unsafe {

                //             // get mut refs
                //             let lower_i = lower.get_unchecked_mut(i);
                //             let upper_i = upper.get_unchecked_mut(i);
                //             *lower_i = *(&*lower_i).min(subset.get_unchecked(p).get_unchecked(i));
                //             *upper_i = *(&*upper_i).max(subset.get_unchecked(p).get_unchecked(i));
                //         }
                //     }
                // }
                let leaf = Node::Leaf {
                    points: subset.to_vec(),
                    lower,
                    upper,
                };
                #[cfg(feature = "timing")]
                let vec_alloc = timer.elapsed().as_nanos();
                #[cfg(feature = "timing")]
                LEAF_VEC_ALLOC.fetch_add(vec_alloc as usize, Ordering::SeqCst);

                #[cfg(feature = "timing")]
                let timer = std::time::Instant::now();
                let leaf_index = nodes.len();
                nodes.push(leaf);
                #[cfg(feature = "timing")]
                let write_time = timer.elapsed().as_nanos();
                #[cfg(feature = "timing")]
                LEAF_WRITE.fetch_add(write_time as usize, Ordering::SeqCst);

                leaf_index
            }
            false => {
                // Calculate index of median
                let sub_len = subset.len();
                let median_index = sub_len / 2 - ((sub_len + 1) % 2);

                #[cfg(feature = "timing")]
                let timer = std::time::Instant::now();
                // Select median in this subset based on split_dim component
                let (left, median, right) = subset
                    .select_nth_unstable_by(median_index, |a, b| unsafe {
                        a.get_unchecked(split_dim).cmp(&b.get_unchecked(split_dim))
                    });
                let split_val = unsafe { median.get_unchecked(split_dim) };
                #[cfg(feature = "timing")]
                let stem_median = timer.elapsed().as_nanos();
                #[cfg(feature = "timing")]
                STEM_MEDIAN.fetch_add(stem_median as usize, Ordering::SeqCst);

                let left_handle = Tree::build_nodes::<NOT_FIRST, IS_LEFT>(
                    left,
                    split_level,
                    leafsize,
                    nodes,
                    Some((lower, upper)),
                    Some(split_val),
                );
                let right_handle = Tree::build_nodes::<NOT_FIRST, IS_RIGHT>(
                    right,
                    split_level,
                    leafsize,
                    nodes,
                    Some((lower, upper)),
                    Some(split_val),
                );

                let stem = Node::Stem {
                    split_dim,
                    point: median,
                    left: left_handle,
                    right: right_handle,
                    lower,
                    upper,
                };

                #[cfg(feature = "timing")]
                let timer = std::time::Instant::now();
                let stem_index = nodes.len();
                nodes.push(stem);
                #[cfg(feature = "timing")]
                let stem_write = timer.elapsed().as_nanos();
                #[cfg(feature = "timing")]
                STEM_WRITE.fetch_add(stem_write as usize, Ordering::SeqCst);

                stem_index
            }
        }
    }

    // Returns the number of nodes in the tree
    // The root node is contained in the struct, so must add one.
    pub fn size(&self) -> usize {
        self.nodes.len() + 1
    }
}

#[cfg(test)]
mod tests {

    use crate::Tree;
    use concat_idents::concat_idents;
    use ordered_float::NotNan;
    use seq_macro::seq;

    // Generate 1..16 dimensional size=1 kd tree unit tests
    macro_rules! size_one_kdtree {
        ($d:ident) => {
            concat_idents!(test_name = test_make_, $d, _, dtree, {
                #[test]
                fn test_name() {
                    let leafsize = 16;

                    let data: Vec<_> = (0..leafsize)
                        .map(|x| [x as f64; $d])
                        .collect();

                    let tree = Tree::new(&data, leafsize).unwrap();
                    assert_eq!(tree.size(), 1);
                }
            });
        };
    }
    seq!(D in 0..=8 {
        #[allow(non_upper_case_globals)]
        const two_pow~D: usize = 2_usize.pow(D);
        size_one_kdtree!(two_pow~D);
    });

    #[test]
    fn test_make_1dtree_with_size_three() {
        let data: Vec<[f64; 1]> = [
            [(); 32]
                .map(|_| [0.1_f64])
                .as_ref(),
            [(); 1]
                .map(|_| [0.5_f64])
                .as_ref(),
            [(); 32]
                .map(|_| [0.9_f64])
                .as_ref(),
        ]
        .concat();

        let leafsize = 32;

        let tree = Tree::new(&data, leafsize).unwrap();
        assert_eq!(tree.size(), 3);
    }
}

pub fn ilog2(x: usize) -> usize {
    (x as f64).log2() as usize
}


fn size_of_tree(datalen: usize, leafsize: usize) -> usize {
    if datalen == 0 {
        0
    } else if datalen <= leafsize {
        1
    } else {
        let left = datalen / 2 - 1;
        let right = datalen / 2 - datalen % 2;
        1 + size_of_tree(left, leafsize) + size_of_tree(right, leafsize) 
    }
}

#[test]
fn test_size_of_tree() {

    //   1
    // 2   2
    let datalen = 5;
    let leafsize = 4;
    assert_eq!(size_of_tree(datalen, leafsize), 3);

    //   1
    // 4   4
    let datalen = 9;
    let leafsize = 4;
    assert_eq!(size_of_tree(datalen, leafsize), 3);

    //   1
    // 4   1
    //    2 2
    let datalen = 10;
    let leafsize = 4;
    assert_eq!(size_of_tree(datalen, leafsize), 5);

    //     1
    //   1   1
    //  5 6 5 6
    let datalen = 23;
    let leafsize = 7;
    assert_eq!(size_of_tree(datalen, leafsize), 7);

    //      1
    //   1     1
    // 24 24 24 25
    let datalen = 100;
    let leafsize = 32;
    assert_eq!(size_of_tree(datalen, leafsize), 7);

    // Too large
    let datalen = 100_000;
    let leafsize = 32;
    assert_eq!(size_of_tree(datalen, leafsize), 8191);
}