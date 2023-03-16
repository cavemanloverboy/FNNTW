use std::fmt::Debug;

use crate::{
    point::{Float, Point},
    utils::{check_point_return, process_dist2, FnntwResult},
    Node, Tree,
};
use likely_stable::unlikely;
use ordered_float::NotNan;

impl<'t, T: Float + Debug, const D: usize> Tree<'t, T, D> {
    pub fn query_nearest_noidx<'q>(&'q self, query: &[T; D]) -> FnntwResult<T, T> {
        // Check for valid query point
        let query: &[NotNan<T>; D] = check_point_return(query)?;

        if let Some(ref boxsize) = self.boxsize {
            // Periodic query
            Ok(process_dist2(
                self.query_nearest_periodic_noidx(query, boxsize),
            ))
        } else {
            // Non periodic query
            Ok(process_dist2(self.query_nearest_nonperiodic_noidx(query)))
        }
    }

    /// Given a query point `query`, query the tree and return point's nearest neighbor.
    /// The value returned is (`distance_to_neighbor: T`, `neighbor_index: u64`, `neighbor_position: &'t [NotNan<T>; D]`).
    fn query_nearest_nonperiodic_noidx<'q>(&'q self, query: &[NotNan<T>; D]) -> T {
        // Get reference to the root node
        let current_node: &Node<T, D> = &self.root_node;

        // Ledger with info about nodes we've touched, namely the parent and sibling nodes
        // and distance to their associated space in form of (&usize, T), where usize is
        // the index inside of self.nodes. The root node is checked at the end.
        let mut points_to_check: Vec<(&usize, &Point<T, D>, T)> =
            Vec::with_capacity(self.height_hint);

        let mut current_best_dist_sq = T::max_value();
        let mut current_best_neighbor: &'q Point<T, D> = current_node.stem();

        // Recurse down (and then up and down) the stem
        self.check_stem(
            query,
            current_node,
            &mut current_best_dist_sq,
            &mut current_best_neighbor,
            &mut points_to_check,
        );

        return current_best_dist_sq;
    }

    fn query_nearest_periodic_noidx<'q>(
        &'q self,
        query: &[NotNan<T>; D],
        boxsize: &[NotNan<T>; D],
    ) -> T {
        // First get real image result
        let mut best_dist2 = self.query_nearest_nonperiodic_noidx(query);

        // Find closest dist2 to every side
        let mut closest_side_dist2 = [T::zero(); D];
        for side in 0..D {
            // Do a single index here. This is equal to distance to lower side
            // safety: made safe with const generic
            let query_component = unsafe { query.get_unchecked(side) };

            // Get distance to upper half
            // safety: made safe with const gneric
            let upper = unsafe { boxsize.get_unchecked(side) } - query_component;

            // !negative includes zero
            debug_assert!(!upper.is_sign_negative());
            debug_assert!(!query_component.is_sign_negative());

            // Choose lesser of two and then square
            closest_side_dist2[side] = upper.min(*query_component).powi(2);
        }

        // Find which images we need to check.
        // Initialize vector with real image (which we will remove later)
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
                        // safety: made safe with const generic
                        Some(unsafe { closest_side_dist2.get_unchecked(side) })
                    } else {
                        None
                    }
                })
                .fold(T::zero(), |acc, x| acc + *x);

            // INTRINSICS: in any reasonably sized kdtree, most points will not be near the edge
            if unlikely(dist_to_side_edge_or_other < best_dist2) {
                let mut image_to_check = query.clone();

                for (idx, flag) in closest_image.enumerate() {
                    // If moving image along this dimension
                    if flag {
                        // Do a single index here. This is equal to distance to lower side
                        // safety: made safe with const generic
                        let query_component: &NotNan<T> = unsafe { query.get_unchecked(idx) };

                        // Single index here as well
                        // safety: made safe with const generic
                        let boxsize_component = unsafe { boxsize.get_unchecked(idx) };

                        // safety: made safe with const generic
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
        for image in &images_to_check {
            // Get image result
            let query_result = self.query_nearest_nonperiodic_noidx(
                // safety: NotNan --> T, the image will be checked by query_nearest
                unsafe { std::mem::transmute(image) },
            );

            let image_best_dist2 = query_result;

            // INTRINSICS: most images will be further than best_dist2
            if unlikely(image_best_dist2 < best_dist2) {
                best_dist2 = image_best_dist2;
            }
        }

        return best_dist2;
    }
}
