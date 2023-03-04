use std::fmt::Debug;

use crate::{
    point::{Float, Point},
    utils::{check_point_return, FnntwResult, QueryKResult},
    Node, Tree,
};
use ordered_float::NotNan;

use super::container::Container;

impl<'t, T: Float + Debug, const D: usize> Tree<'t, T, D> {
    #[cfg(all(feature = "parallel", feature = "no-position"))]
    pub fn query_nearest_k_parallel<'q>(
        &'q self,
        queries: &'q [[T; D]],
        k: usize,
    ) -> FnntwResult<QueryKResult<'t, T, D>, T> {
        use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

        #[cfg(feature = "timing")]
        let alloc_timer = std::time::Instant::now();

        let mut distances = Vec::with_capacity(queries.len() * k);
        let mut indices = Vec::with_capacity(queries.len() * k);
        let dist_ptr_usize = distances.as_mut_ptr() as usize;
        let idx_ptr_usize = indices.as_mut_ptr() as usize;

        #[cfg(feature = "timing")]
        println!(
            "time to alloc was {} micros",
            alloc_timer.elapsed().as_micros()
        );

        if let Some(ref boxsize) = self.boxsize {
            queries.into_par_iter().enumerate().try_for_each(
                |(query_index, query)| -> FnntwResult<_, T> {
                    // Check for valid query point
                    let query: &[NotNan<T>; D] = check_point_return(query)?;

                    let (mut container, mut point_vec) = (
                        Container::new(k.min(self.data.len())),
                        Vec::with_capacity(2 * self.height_hint),
                    );

                    // Periodic query
                    self.query_nearest_k_periodic_into(
                        query,
                        k,
                        boxsize,
                        &mut container,
                        &mut point_vec,
                        dist_ptr_usize,
                        idx_ptr_usize,
                        query_index,
                    );

                    Ok(())
                },
            )?;
        } else {
            queries.into_par_iter().enumerate().try_for_each(
                |(query_index, query)| -> FnntwResult<_, T> {
                    // Check for valid query point
                    let query: &[NotNan<T>; D] = check_point_return(query)?;

                    let (mut container, mut point_vec) = (
                        Container::new(k.min(self.data.len())),
                        Vec::with_capacity(2 * self.height_hint),
                    );
                    // Nonperiodic query
                    self.query_nearest_k_nonperiodic_into(
                        query,
                        k,
                        &mut container,
                        &mut point_vec,
                        dist_ptr_usize,
                        idx_ptr_usize,
                        query_index,
                    );

                    Ok(())
                },
            )?;
        }

        unsafe {
            distances.set_len(queries.len() * k);
            indices.set_len(queries.len() * k);
        }

        Ok((distances, indices))
    }

    fn query_nearest_k_nonperiodic_into<'q>(
        &'q self,
        query: &'q [NotNan<T>; D],
        _k: usize,
        container: &mut Container<'q, T, D>,
        points_to_check: &mut Vec<(&'q usize, &'q Point<T, D>, T)>,
        distances_ptr: usize,
        indices_ptr: usize,
        query_index: usize,
    ) where
        't: 'q,
    {
        // Get reference to the root node
        let current_node: &'q Node<T, D> = &self.root_node;
        container.push((T::max_value(), current_node.stem()));

        // Recurse down (and then up and down) the stem
        self.check_stem_k(query, current_node, container, points_to_check);

        // Write to given vector
        container.index_into(distances_ptr, indices_ptr, query_index, self.start());
        // index into clears!!
    }

    fn query_nearest_k_periodic_into<'q, 'i>(
        &'q self,
        query: &'q [NotNan<T>; D],
        _k: usize,
        boxsize: &[NotNan<T>; D],
        container: &mut Container<'q, T, D>,
        points_to_check: &mut Vec<(&'q usize, &'q Point<T, D>, T)>,
        distances_ptr: usize,
        indices_ptr: usize,
        query_index: usize,
    ) where
        't: 'q,
    {
        // First get real image result
        let mut real_image_container: &mut Container<T, D> = {
            // Get reference to the root node
            let current_node: &Node<T, D> = &self.root_node;
            container.push((T::max_value(), current_node.stem()));

            // Recurse down (and then up and down) the stem
            self.check_stem_k(query, current_node, container, points_to_check);

            container
        };

        // Find closest dist2 to every side
        let mut closest_side_dist2 = [T::zero(); D];
        for side in 0..D {
            // Do a single index here. This is equal to distance to lower side
            // safety: made safe by const generic
            let query_component = unsafe { query.get_unchecked(side) };

            // Get distance to upper half
            // safety: made safe by const generic
            let upper = unsafe { boxsize.get_unchecked(side) } - query_component;

            // !negative includes zero
            debug_assert!(!upper.is_sign_negative());
            debug_assert!(!query_component.is_sign_negative());

            // Choose lesser of two and then square
            closest_side_dist2[side] = upper.min(*query_component).powi(2);
        }

        // Find which images we need to check.
        // Initialize vector with real image (which we will remove later)
        let best_real_dist2 = real_image_container.best_dist2();
        let mut images_to_check = Vec::with_capacity(2_usize.pow(D as u32) - 1);
        for image in 1..2_usize.pow(D as u32) {
            // Closest image in the form of bool array
            let closest_image = (0..D as u32).map(|idx| ((image / 2_usize.pow(idx)) % 2) == 1);

            // Find distance to corresponding side, edge, vertex or other higher dimensional equivalent
            let dist_to_side_edge_or_other: T = closest_image
                .clone()
                .enumerate()
                .flat_map(|(side, flag)| {
                    if flag {
                        // Get minimum of dist2 to lower and upper side
                        // safety: made safe by const generic
                        Some(unsafe { closest_side_dist2.get_unchecked(side) })
                    } else {
                        None
                    }
                })
                .fold(T::zero(), |acc, x| acc + *x);

            if dist_to_side_edge_or_other < *best_real_dist2 {
                let mut image_to_check = query.clone();

                for (idx, flag) in closest_image.enumerate() {
                    // If moving image along this dimension
                    if flag {
                        // Do a single index here. This is equal to distance to lower side
                        // safety: made safe by const generic
                        let query_component: &NotNan<T> = unsafe { query.get_unchecked(idx) };

                        // Single index here as well
                        // safety: made safe by const generic
                        let boxsize_component = unsafe { boxsize.get_unchecked(idx) };

                        // safety: made safe by const generic
                        unsafe {
                            if *query_component < boxsize_component / T::from(2.0).unwrap() {
                                // Add if in lower half of box
                                *image_to_check.get_unchecked_mut(idx) =
                                    query_component + boxsize_component
                            } else {
                                // Subtract if in upper half of box
                                *image_to_check.get_unchecked_mut(idx) =
                                    query_component - boxsize_component
                            }
                        }
                    }
                }

                images_to_check.push(image_to_check);
            }
        }

        // Then check all images we need to check
        for image in images_to_check {
            // Get image result
            self.check_stem_k(
                &image,
                &self.root_node,
                &mut real_image_container,
                points_to_check,
            );
        }

        real_image_container.index_into(distances_ptr, indices_ptr, query_index, self.start());
    }
}
