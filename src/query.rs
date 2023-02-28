use std::fmt::Debug;

use crate::{
    distance::*,
    point::{Float, Point},
    utils::{check_point, process_result, FnntwResult, QueryResult},
    Node, Tree,
};
use likely_stable::unlikely;
use ordered_float::NotNan;

impl<'t, T: Float + Debug, const D: usize> Tree<'t, T, D> {
    pub fn query_nearest<'q>(&'q self, query: &[T; D]) -> FnntwResult<QueryResult<'t, T, D>, T> {
        // Check for valid query point
        let query: &[NotNan<T>; D] = check_point(query)?;

        if let Some(ref boxsize) = self.boxsize {
            // Periodic query
            Ok(process_result::<'t, T, D>(
                self.query_nearest_periodic(query, boxsize),
            ))
        } else {
            // Non periodic query
            Ok(process_result::<'t, T, D>(
                self.query_nearest_nonperiodic(query),
            ))
        }
    }

    /// Given a query point `query`, query the tree and return point's nearest neighbor.
    /// The value returned is (`distance_to_neighbor: T`, `neighbor_index: u64`, `neighbor_position: &'t [NotNan<T>; D]`).
    #[inline(always)]
    fn query_nearest_nonperiodic<'q>(&'q self, query: &[NotNan<T>; D]) -> QueryResult<'t, T, D> {
        // Get reference to the root node
        let current_node: &Node<'t, T, D> = &self.root_node;

        // Ledger with info about nodes we've touched, namely the parent and sibling nodes
        // and distance to their associated space in form of (&usize, T), where usize is
        // the index inside of self.nodes. The root node is checked at the end.
        let mut points_to_check: Vec<(&usize, &Point<'t, T, D>, T)> =
            Vec::with_capacity(self.height_hint);

        let mut current_best_dist_sq = T::max_value();
        let mut current_best_neighbor: &'q Point<'t, T, D> = current_node.stem();

        // Recurse down (and then up and down) the stem
        self.check_stem(
            query,
            current_node,
            &mut current_best_dist_sq,
            &mut current_best_neighbor,
            &mut points_to_check,
        );

        (
            current_best_dist_sq,
            current_best_neighbor.index,
            #[cfg(not(feature = "no-position"))]
            current_best_neighbor.position,
        )
    }

    #[inline(always)]
    fn query_nearest_periodic<'q>(
        &'q self,
        query: &[NotNan<T>; D],
        boxsize: &[NotNan<T>; D],
    ) -> QueryResult<'t, T, D> {
        // First get real image result
        #[cfg(not(feature = "no-position"))]
        let (mut best_dist2, mut best_idx, mut best_nn) = self.query_nearest_nonperiodic(query);
        #[cfg(feature = "no-position")]
        let (mut best_dist2, mut best_idx) = self.query_nearest_nonperiodic(query);

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
            let query_result = self.query_nearest_nonperiodic(
                // safety: NotNan --> T, the image will be checked by query_nearest
                unsafe { std::mem::transmute(image) },
            );

            #[cfg(not(feature = "no-position"))]
            let (image_best_dist2, image_best_idx, image_nn) = query_result;
            #[cfg(feature = "no-position")]
            let (image_best_dist2, image_best_idx) = query_result;

            // INTRINSICS: most images will be further than best_dist2
            if unlikely(image_best_dist2 < best_dist2) {
                best_dist2 = image_best_dist2;
                best_idx = image_best_idx;
                #[cfg(not(feature = "no-position"))]
                {
                    best_nn = image_nn;
                }
            }
        }

        (
            best_dist2,
            best_idx,
            #[cfg(not(feature = "no-position"))]
            best_nn,
        )
    }

    #[inline(always)]
    /// Upon checking that we are close to some other space during upward traversal of the tree,
    /// this function is called to check candidates in the child space, appending any new candidate spaces
    /// as we go along
    fn check_child<'i, 'o>(
        &'i self,
        query: &[NotNan<T>; D],
        sibling: &usize,
        // check's the parent of the sibling (also our parent)
        stem: &'i Point<'t, T, D>,
        current_best_dist_sq: &'o mut T,
        current_best_neighbor: &'o mut &'i Point<'t, T, D>,
        points_to_check: &'o mut Vec<(&'i usize, &'i Point<'t, T, D>, T)>,
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
                self.check_parent(query, stem, current_best_dist_sq, current_best_neighbor);
                self.check_leaf(query, points, current_best_dist_sq, current_best_neighbor)
            }

            // Sibling is a parent (e.g. for unbalanced tree)
            Node::Stem { .. } => {
                self.check_parent(query, stem, current_best_dist_sq, current_best_neighbor);
                self.check_stem(
                    query,
                    sibling,
                    current_best_dist_sq,
                    current_best_neighbor,
                    points_to_check,
                )
            }
        }
    }

    #[inline(always)]
    fn check_leaf<'a, 'b>(
        &self,
        query: &'b [NotNan<T>; D],
        // leaf_points: &'t Vec<&'t [NotNan<T>; D]>,
        leaf_points: impl IntoIterator<Item = &'a Point<'t, T, D>>,
        current_best_dist_sq: &'b mut T,
        current_best_neighbor: &'b mut &'a Point<'t, T, D>,
    ) where
        'a: 'b,
        't: 'a,
    {
        // Check all points in leaf
        for candidate in leaf_points.into_iter() {
            new_best(
                query,
                candidate,
                current_best_dist_sq,
                current_best_neighbor,
            );
        }
    }

    /// If sibling is a stem, then we need to recurse back down
    #[inline(always)]
    fn check_stem<'i, 'o>(
        &'i self,
        query: &[NotNan<T>; D],
        stem: &'i Node<'t, T, D>,
        current_best_dist_sq: &'o mut T,
        current_best_neighbor: &'o mut &'i Point<'t, T, D>,
        points_to_check: &'o mut Vec<(&'i usize, &'i Point<'t, T, D>, T)>,
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
                    ref right,
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
        self.check_leaf(
            query,
            current_node.iter(),
            current_best_dist_sq,
            current_best_neighbor,
        );

        // Now we empty out the queue
        while let Some((sibling, parent, dist_sq_to_space)) = points_to_check.pop() {
            if dist_sq_to_space < *current_best_dist_sq {
                self.check_child(
                    query,
                    sibling,
                    parent,
                    current_best_dist_sq,
                    current_best_neighbor,
                    points_to_check,
                );
            }
        }
    }

    #[inline(always)]
    fn check_parent<'i, 'o>(
        &self,
        query: &[NotNan<T>; D],
        stem: &'i Point<'t, T, D>,
        current_best_dist_sq: &'o mut T,
        current_best_neighbor: &'o mut &'i Point<'t, T, D>,
    ) where
        'i: 'o,
        't: 'i,
    {
        new_best(query, stem, current_best_dist_sq, current_best_neighbor);
    }
}
