use crate::{distance::*, Node, Tree, utils::{FnntwResult, check_point}};
use ordered_float::NotNan;

impl<'t, const D: usize> Tree<'t, D> {


    #[cfg(not(feature = "do-not-return-position"))]
    pub fn query_nearest(
        &'t self,
        query: &[f64; D]
    ) -> FnntwResult<(f64, u64, &'t [NotNan<f64>; D])> {

        // Check for valid query point
        let query: &[NotNan<f64>; D] = check_point(query)?;

        if let Some(ref boxsize) = self.boxsize {

            // Periodic query
            Ok(self.query_nearest_periodic(query, boxsize))
        } else {

            // Non periodic query
            Ok(self.query_nearest_nonperiodic(query))
        }
    }

    /// Given a query point `query`, query the tree and return point's nearest neighbor.
    /// The value returned is (`distance_to_neighbor: f64`, `neighbor_index: u64`, `neighbor_position: &'t [NotNan<f64>; D]`).
    #[cfg(not(feature = "do-not-return-position"))]
    fn query_nearest_nonperiodic(
        &'t self,
        query: &[NotNan<f64>; D]
    ) -> (f64, u64, &'t [NotNan<f64>; D]) {

        // Get reference to the root node
        let current_node: &Node<'t, D> = &self.root_node;

        // Ledger with info about nodes we've touched, namely the parent and sibling nodes
        // and distance to their associated space in form of (&usize, f64), where usize is
        // the index inside of self.nodes. The root node is checked at the end.
        let mut points_to_check: Vec<(&usize, &[NotNan<f64>; D], f64)> =
            Vec::with_capacity(self.height_hint);

        let mut current_best_dist_sq = std::f64::MAX;
        let mut current_best_neighbor: &'t [NotNan<f64>; D] = current_node
            .stem_position();

        // Recurse down (and then up and down) the stem
        self.check_stem(
            query,
            current_node,
            &mut current_best_dist_sq,
            &mut current_best_neighbor,
            &mut points_to_check,
        );

        let best_idx = self.data_index[current_best_neighbor];

        (current_best_dist_sq, best_idx, current_best_neighbor)
    }

    #[cfg(not(feature = "do-not-return-position"))]
    fn query_nearest_periodic(
        &'t self,
        query: &[NotNan<f64>; D],
        boxsize: &[NotNan<f64>; D],
    ) -> (f64, u64, &'t [NotNan<f64>; D]) {

        // First get real image result
        let (mut best_dist2, mut best_idx, mut best_nn) = self.query_nearest_nonperiodic(query);

        // Find closest dist2 to every side
        let mut closest_side_dist2 = [0.0_f64; D];
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
        let mut images_to_check = Vec::with_capacity(2_usize.pow(D as u32)-1);
        for image in 1..2_usize.pow(D as u32) {
            
            // Closest image in the form of bool array
            let closest_image = (0..D)
                .map(|idx| ((image / 2_usize.pow(idx as u32)) % 2) == 1);

            // Find distance to corresponding side, edge, vertex or other higher dimensional equivalent
            let dist_to_side_edge_or_other: f64 = closest_image
                .clone()
                .enumerate()
                .flat_map(|(side, flag)| if flag {
                    
                    // Get minimum of dist2 to lower and upper side
                    // safety: made safe with const generic
                    Some( unsafe { closest_side_dist2.get_unchecked(side) } )
                } else { None })
                .fold(0.0, |acc, x| acc + x);

            if dist_to_side_edge_or_other < best_dist2 {

                let mut image_to_check = query.clone();
                
                for (idx, flag) in closest_image.enumerate() {

                    // If moving image along this dimension
                    if flag {

                        // Do a single index here. This is equal to distance to lower side
                        // safety: made safe with const generic
                        let query_component: &NotNan<f64> =  unsafe { query.get_unchecked(idx) };

                        // Single index here as well
                        // safety: made safe with const generic
                        let boxsize_component = unsafe { boxsize.get_unchecked(idx) };

                        // safety: made safe with const generic
                        unsafe {
                            if *query_component < boxsize_component / 2.0 {
                                // Add if in lower half of box
                                *image_to_check.get_unchecked_mut(idx) = query_component + boxsize_component
                            } else {
                                // Subtract if in upper half of box
                                *image_to_check.get_unchecked_mut(idx) = query_component - boxsize_component
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
            let (image_best_dist2, image_best_idx, image_nn) = self.query_nearest_nonperiodic(
                // safety: NotNan --> f64, the image will be checked by query_nearest
                unsafe {
                    std::mem::transmute(image)
                }
            );

            if image_best_dist2 < best_dist2 {
                best_dist2 = image_best_dist2;
                best_idx = image_best_idx;
                best_nn = image_nn;
            }
        }

        (best_dist2, best_idx, best_nn)
    }

    #[cfg(feature = "do-not-return-position")]
    pub fn query_nearest(
        &'t self,
        query: &[f64; D]
    ) -> FnntwResult<(f64, u64)> {

        // Check for valid query point
        let query: &[NotNan<f64>; D] = check_point(query)?;

        if let Some(ref boxsize) = self.boxsize {

            // Periodic query
            Ok(self.query_nearest_periodic(query, boxsize))
        } else {

            // Non periodic query
            Ok(self.query_nearest_nonperiodic(query))
        }
    }

    /// Given a query point `query`, query the tree and return point's nearest neighbor.
    /// The value returned is (`distance_to_neighbor: f64`, `neighbor_index: u64`, `neighbor_position: &'t [NotNan<f64>; D]`).
    #[cfg(feature = "do-not-return-position")]
    fn query_nearest_nonperiodic(
        &'t self,
        query: &[NotNan<f64>; D]
    ) -> (f64, u64) {

        // Get reference to the root node
        let current_node: &Node<'t, D> = &self.root_node;

        // Ledger with info about nodes we've touched, namely the parent and sibling nodes
        // and distance to their associated space in form of (&usize, f64), where usize is
        // the index inside of self.nodes. The root node is checked at the end.
        let mut points_to_check: Vec<(&usize, &[NotNan<f64>; D], f64)> =
            Vec::with_capacity(self.height_hint);

        let mut current_best_dist_sq = std::f64::MAX;
        let mut current_best_neighbor: &'t [NotNan<f64>; D] = current_node
            .stem_position();

        // Recurse down (and then up and down) the stem
        self.check_stem(
            query,
            current_node,
            &mut current_best_dist_sq,
            &mut current_best_neighbor,
            &mut points_to_check,
        );

        let best_idx = self.data_index[current_best_neighbor];

        (current_best_dist_sq, best_idx)
    }

    #[cfg(feature = "do-not-return-position")]
    fn query_nearest_periodic(
        &'t self,
        query: &[NotNan<f64>; D],
        boxsize: &[NotNan<f64>; D],
    ) -> (f64, u64) {

        // First get real image result
        let (mut best_dist2, mut best_idx) = self.query_nearest_nonperiodic(query);

        // Find closest dist2 to every side
        let mut closest_side_dist2 = [0.0_f64; D];
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
        let mut images_to_check = Vec::with_capacity(2_usize.pow(D as u32)-1);
        for image in 1..2_usize.pow(D as u32) {
            
            // Closest image in the form of bool array
            let closest_image = (0..D)
                .map(|idx| ((image / 2_usize.pow(idx as u32)) % 2) == 1);

            // Find distance to corresponding side, edge, vertex or other higher dimensional equivalent
            let dist_to_side_edge_or_other: f64 = closest_image
                .clone()
                .enumerate()
                .flat_map(|(side, flag)| if flag {
                    
                    // Get minimum of dist2 to lower and upper side
                    // safety: made safe with const generic
                    Some( unsafe { closest_side_dist2.get_unchecked(side) } )
                } else { None })
                .fold(0.0, |acc, x| acc + x);

            if dist_to_side_edge_or_other < best_dist2 {

                let mut image_to_check = query.clone();
                
                for (idx, flag) in closest_image.enumerate() {

                    // If moving image along this dimension
                    if flag {

                        // Do a single index here. This is equal to distance to lower side
                        // safety: made safe with const generic
                        let query_component: &NotNan<f64> =  unsafe { query.get_unchecked(idx) };

                        // Single index here as well
                        // safety: made safe with const generic
                        let boxsize_component = unsafe { boxsize.get_unchecked(idx) };

                        // safety: made safe with const generic
                        unsafe {
                            if *query_component < boxsize_component / 2.0 {
                                // Add if in lower half of box
                                *image_to_check.get_unchecked_mut(idx) = query_component + boxsize_component
                            } else {
                                // Subtract if in upper half of box
                                *image_to_check.get_unchecked_mut(idx) = query_component - boxsize_component
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
            let (image_best_dist2, image_best_idx) = self.query_nearest_nonperiodic(
                // safety: NotNan --> f64, the image will be checked by query_nearest
                unsafe {
                    std::mem::transmute(image)
                }
            );

            if image_best_dist2 < best_dist2 {
                best_dist2 = image_best_dist2;
                best_idx = image_best_idx;
            }
        }

        (best_dist2, best_idx)
    }

    #[inline(always)]
    /// Upon checking that we are close to some other space during upward traversal of the tree,
    /// this function is called to check candidates in the child space, appending any new candidate spaces
    /// as we go along
    fn check_child<'i, 'o>(
        &'i self,
        query: &[NotNan<f64>; D],
        sibling: &usize,
        // check's the parent of the sibling (also our parent)
        stem_position: &'i [NotNan<f64>; D],
        current_best_dist_sq: &'o mut f64,
        current_best_neighbor: &'o mut &'i [NotNan<f64>; D],
        points_to_check: &'o mut Vec<(&'i usize, &'i [NotNan<f64>; D], f64)>,
    ) where
        'i: 'o,
        't: 'i,
    {
        // safety: indices are valid by construction, with the atomic lock on Vec<Node>
        let sibling = unsafe { self.nodes.get_unchecked(*sibling) };
        match sibling {
            // Sibling is a leaf
            Node::Leaf { points, .. } => {
                // the stem_position here is the position of the parent
                self.check_parent(
                    query,
                    stem_position,
                    current_best_dist_sq,
                    current_best_neighbor,
                );
                self.check_leaf(query, points, current_best_dist_sq, current_best_neighbor)
            }

            // Sibling is a parent (e.g. for unbalanced tree)
            Node::Stem { .. } => {
                self.check_parent(
                    query,
                    stem_position,
                    current_best_dist_sq,
                    current_best_neighbor,
                );
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
        query: &'b [NotNan<f64>; D],
        // leaf_points: &'t Vec<&'t [NotNan<f64>; D]>,
        leaf_points: impl IntoIterator<Item = &'b &'a [NotNan<f64>; D]>,
        current_best_dist_sq: &mut f64,
        current_best_neighbor: &mut &'a [NotNan<f64>; D],
    ) where
        'a: 'b,
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
        query: &[NotNan<f64>; D],
        stem: &'i Node<D>,
        current_best_dist_sq: &'o mut f64,
        current_best_neighbor: &'o mut &'i [NotNan<f64>; D],
        points_to_check: &'o mut Vec<(&'i usize, &'i [NotNan<f64>; D], f64)>,
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
        query: &[NotNan<f64>; D],
        stem_position: &'i [NotNan<f64>; D],
        current_best_dist_sq: &'o mut f64,
        current_best_neighbor: &'o mut &'i [NotNan<f64>; D],
    ) where
        'i: 'o,
        't: 'i,
    {
        new_best(
            query,
            stem_position,
            current_best_dist_sq,
            current_best_neighbor,
        );
    }

    // fn check_point(&self, boxsize: &[NotNan<f64>; D]) -> Result<()> {

    //     // Initialize flag
    //     let mut okay = true;
    
    //     for length in boxsize {
    //         if !(length.is_sign_positive() && length.is_normal()) {
    //             okay = false; 
    //         } else  {

    //         }

    //     }
    
    //     okay
    // }
}
