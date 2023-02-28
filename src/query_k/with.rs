use std::fmt::Debug;

use crate::{
    point::{Float, Point},
    utils::{check_point_return, FnntwResult, QueryKResult},
    Node, Tree,
};
use ordered_float::NotNan;

use super::container::Container;

impl<'t, T: Float + Debug, const D: usize> Tree<'t, T, D> {
    pub fn query_nearest_k_with<'i>(
        &'i self,
        query: &'i [T; D],
        k: usize,
        container: &'i mut Container<'t, T, D>,
        points_to_check: &'i mut Vec<(&'i usize, &'i Point<T, D>, T)>,
    ) -> FnntwResult<QueryKResult<'t, T, D>, T>
    where
        't: 'i,
        'i: 't,
    {
        // Check for valid query point
        let query: &[NotNan<T>; D] = check_point_return(query)?;

        // Check container is correct
        container.check(k.min(self.data.len()));

        if let Some(ref boxsize) = self.boxsize {
            // Periodic query
            Ok(self.query_nearest_k_periodic_with(query, boxsize, container, points_to_check))
        } else {
            // Nonperiodic query
            Ok(self.query_nearest_k_nonperiodic_with(query, container, points_to_check))
        }
    }
    // <'i, 'o>(
    //     &'i self,
    //     query: &'o [NotNan<T>; D],
    //     stem: &'i Node<T, D>,
    //     container: &'o mut Container<'i, T, D>,
    //     points_to_check: &'o mut Vec<(&'i usize, &'i Point<T, D>, T)>,

    fn query_nearest_k_nonperiodic_with<'i>(
        &'i self,
        query: &'i [NotNan<T>; D],
        container: &'i mut Container<'t, T, D>,
        points_to_check: &'i mut Vec<(&'i usize, &'i Point<T, D>, T)>,
    ) -> QueryKResult<'t, T, D>
    where
        't: 'i,
        'i: 't,
    {
        // Get reference to the root node
        let current_node: &Node<T, D> = &self.root_node;

        // Ledger with info about nodes we've touched, namely the parent and sibling nodes
        // and distance to their associated space in form of (&usize, T), where usize is
        // the index inside of self.nodes. The root node is checked at the end.
        // let mut points_to_check: Vec<(&usize, &Point<T, D>, T)> =
        // Vec::with_capacity(self.height_hint);

        // Initialize candidate container with dummy point
        // let mut container = Container::new(k.min(self.data.len()));
        container.push((T::max_value(), current_node.stem()));

        // Recurse down (and then up and down) the stem
        self.check_stem_k(query, current_node, container, points_to_check);

        container.index()
    }

    // <'i, 'o>(
    //     &'i self,
    //     query: &'o [NotNan<T>; D],
    //     stem: &'i Node<T, D>,
    //     container: &'o mut Container<'i, T, D>,
    //     points_to_check: &'o mut Vec<(&'i usize, &'i Point<T, D>, T)>,

    fn query_nearest_k_periodic_with<'i>(
        &'i self,
        query: &'i [NotNan<T>; D],
        boxsize: &'i [NotNan<T>; D],
        container: &'i mut Container<'t, T, D>,
        points_to_check: &'i mut Vec<(&'i usize, &'i Point<T, D>, T)>,
    ) -> QueryKResult<'t, T, D>
    where
        't: 'i,
        'i: 't,
    {
        // First get real image result
        let real_image_container: &mut Container<T, D> = {
            // Get reference to the root node
            let current_node: &Node<T, D> = &self.root_node;

            // Ledger with info about nodes we've touched, namely the parent and sibling nodes
            // and distance to their associated space in form of (&usize, T), where usize is
            // the index inside of self.nodes. The root node is checked at the end.
            // let mut points_to_check: Vec<(&usize, &Point<T, D>, T)> =
            // Vec::with_capacity(self.height_hint);

            // Initialize candidate container with dummy point
            // let mut container = Container::new(k.min(self.data.len()));
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
            // Ledger with info about nodes we've touched, namely the parent and sibling nodes
            // and distance to their associated space in form of (&usize, T), where usize is
            // the index inside of self.nodes. The root node is checked at the end.
            // let mut points_to_check: Vec<(&usize, &Point<T, D>, T)> =
            //     Vec::with_capacity(self.height_hint);

            // Get image result
            self.check_stem_k(
                &image,
                &self.root_node,
                real_image_container,
                points_to_check,
            );
        }

        real_image_container.index()
    }
}
