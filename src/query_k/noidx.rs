use std::fmt::Debug;

use ordered_float::NotNan;

use crate::{
    distance::{calc_dist_sq_to_space, new_best_kth, new_best_kth_noidx},
    point::{Float, Point},
    utils::{check_point_return, FnntwResult},
    Node, Tree,
};

use super::container_noidx::ContainerNoIndex;

impl<'t, T: Float + Debug, const D: usize> Tree<'t, T, D> {
    pub fn query_nearest_k_noidx<'q>(
        &'q self,
        query: &'q [T; D],
        k: usize,
    ) -> FnntwResult<Vec<T>, T> {
        // Check for valid query point
        let query: &[NotNan<T>; D] = check_point_return(query)?;

        if let Some(ref boxsize) = self.boxsize {
            // Periodic query
            Ok(self.query_nearest_k_periodic_noidx(query, k, boxsize))
        } else {
            // Nonperiodic query
            Ok(self.query_nearest_k_nonperiodic_noidx(query, k))
        }
    }

    fn query_nearest_k_nonperiodic_noidx<'q>(
        &'q self,
        query: &'q [NotNan<T>; D],
        k: usize,
    ) -> Vec<T>
    where
        't: 'q,
    {
        // Get reference to the root node

        let current_node: &Node<T, D> = &self.root_node;

        // Ledger with info about nodes we've touched, namely the parent and sibling nodes
        // and distance to their associated space in form of (&usize, T), where usize is
        // the index inside of self.nodes. The root node is checked at the end.
        let mut points_to_check: Vec<(&usize, &Point<T, D>, T)> =
            Vec::with_capacity(self.height_hint);

        // Initialize candidate container with dummy point
        let mut container = ContainerNoIndex::new(k.min(self.input.len()));
        container.push(T::max_value());

        // Recurse down (and then up and down) the stem
        self.check_stem_k_noidx(query, current_node, &mut container, &mut points_to_check);

        container.finish()
    }

    fn query_nearest_k_periodic_noidx<'q, 'i>(
        &'q self,
        query: &'q [NotNan<T>; D],
        k: usize,
        boxsize: &[NotNan<T>; D],
    ) -> Vec<T>
    where
        't: 'q,
    {
        // Ledger with info about nodes we've touched, namely the parent and sibling nodes
        // and distance to their associated space in form of (&usize, T), where usize is
        // the index inside of self.nodes. The root node is checked at the end.
        let mut points_to_check: Vec<(&usize, &Point<T, D>, T)> =
            Vec::with_capacity(self.height_hint);

        // First get real image result
        let mut real_image_container: ContainerNoIndex<T, D> = {
            // Get reference to the root node
            let current_node: &Node<T, D> = &self.root_node;

            // Initialize candidate container with dummy point
            let mut container = ContainerNoIndex::new(k.min(self.input.len()));
            container.push(T::max_value());

            // Recurse down (and then up and down) the stem
            self.check_stem_k_noidx(query, current_node, &mut container, &mut points_to_check);

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
            self.check_stem_k_noidx(
                &image,
                &self.root_node,
                &mut real_image_container,
                &mut points_to_check,
            );
        }

        real_image_container.finish()
    }

    /// Upon checking that we are close to some other space during upward traversal of the tree,
    /// this function is called to check candidates in the child space, appending any new candidate spaces
    /// as we go along
    fn check_child_k_noidx<'i, 'o>(
        &'i self,
        query: &'o [NotNan<T>; D],
        sibling: &usize,
        // check's the parent of the sibling (also our parent)
        stem: &'i Point<T, D>,
        container: &'o mut ContainerNoIndex<T, D>,
        points_to_check: &'o mut Vec<(&'i usize, &'i Point<T, D>, T)>,
    ) where
        'i: 'o,
        't: 'i,
    {
        // safety: indices are valid by construction, with the atomic lock on Vec<Node>
        let sibling = unsafe { self.nodes.get_unchecked(*sibling) };
        match sibling {
            // Sibling is a leaf
            Node::Leaf { points, .. } => {
                // the stem here is the parent
                self.check_parent_k_noidx(query, stem, container);
                self.check_leaf_k_noidx(query, points.into_iter(), container)
            }

            // Sibling is a parent (e.g. for unbalanced tree)
            Node::Stem { .. } => {
                self.check_parent_k_noidx(query, stem, container);
                self.check_stem_k_noidx(query, sibling, container, points_to_check)
            }
        }
    }

    fn check_leaf_k_noidx<'i, 'o>(
        &self,
        query: &'o [NotNan<T>; D],
        leaf_points: impl Iterator<Item = &'i Point<T, D>>,
        container: &'o mut ContainerNoIndex<T, D>,
    ) where
        'i: 'o,
        't: 'i,
    {
        // Check all points in leaf
        for candidate in leaf_points {
            new_best_kth_noidx(query, candidate, container);
        }
    }

    /// If sibling is a stem, then we need to recurse back down

    fn check_stem_k_noidx<'i, 'o>(
        &'i self,
        query: &'o [NotNan<T>; D],
        stem: &'i Node<T, D>,
        container: &'o mut ContainerNoIndex<T, D>,
        points_to_check: &'o mut Vec<(&'i usize, &'i Point<T, D>, T)>,
    ) where
        'i: 'o,
        't: 'i,
    {
        // Navigate down the stems until we reach a leaf
        let mut current_node = stem;

        while current_node.is_stem() {
            let next_leafnode = match current_node {
                Node::Stem {
                    ref split_dim,
                    point,
                    left,
                    right,
                    ..
                } => {
                    // Determine left/right split
                    // safety: made safe by const generic
                    if unsafe { query.get_unchecked(*split_dim) > point.get_unchecked(*split_dim) }
                    {
                        // Record sibling node and the dist_sq to sibling's associated space
                        // safety: indices are valid by construction, with the atomic lock on Vec<Node>
                        let (sibling_lower, sibling_upper) =
                            unsafe { self.nodes.get_unchecked(*left) }.get_bounds();
                        let dist_sq_to_space =
                            calc_dist_sq_to_space(query, sibling_lower, sibling_upper);
                        points_to_check.push((left, point, dist_sq_to_space));

                        // Right Branch
                        right
                    } else {
                        // Record sibling node and the dist_sq to its associated space
                        // safety: indices are valid by construction, with the atomic lock on Vec<Node>
                        let (sibling_lower, sibling_upper) =
                            unsafe { self.nodes.get_unchecked(*right) }.get_bounds();
                        let dist_sq_to_space =
                            calc_dist_sq_to_space(query, sibling_lower, sibling_upper);
                        points_to_check.push((right, point, dist_sq_to_space));

                        // Left Branch
                        left
                    }
                }
                _ => unreachable!("we are traversing though stems"),
            };

            // Set leafnode
            // safety: indices are valid by construction, with the atomic lock on Vec<Node>
            current_node = unsafe { self.nodes.get_unchecked(*next_leafnode) };
        }

        // We are now at a leaf; check it
        self.check_leaf_k_noidx(query, current_node.iter(), container);

        // Now we empty out the queue
        while let Some((sibling, parent, dist_sq_to_space)) = points_to_check.pop() {
            if dist_sq_to_space < *container.best_dist2() {
                self.check_child_k_noidx(query, sibling, parent, container, points_to_check);
            }
        }
    }

    fn check_parent_k_noidx<'i, 'o>(
        &self,
        query: &[NotNan<T>; D],
        stem: &'i Point<T, D>,
        container: &'o mut ContainerNoIndex<T, D>,
    ) where
        'i: 'o,
        't: 'i,
    {
        new_best_kth_noidx(query, stem, container);
    }
}
