use crate::{distance::*, Node, Tree, utils::{check_point, FnntwResult}, };
use ordered_float::NotNan;

pub(crate) mod container;
use container::Container;

impl<'t, const D: usize> Tree<'t, D> {

    #[cfg(not(feature = "do-not-return-position"))]
    pub fn query_nearest_k<'q>(
        &'q self,
        query: &'q [f64; D],
        k: usize,
    ) -> FnntwResult<Vec<(f64, u64, &'q [NotNan<f64>; D])>> {

        // Check for valid query point
        let query: &[NotNan<f64>; D] = check_point(query)?;

        if let Some(ref boxsize) = self.boxsize {
            
            // Periodic query
            Ok(self.query_nearest_k_periodic(query, k, boxsize))
        } else {

            // Nonperiodic query
            Ok(self.query_nearest_k_nonperiodic(query, k))
        }
    }

    #[cfg(not(feature = "do-not-return-position"))]
    fn query_nearest_k_nonperiodic<'q>(
        &'q self,
        query: &'q [NotNan<f64>; D],
        k: usize,
    ) -> Vec<(f64, u64, &'q [NotNan<f64>; D])>
    where
        't: 'q,
    {

        // Get reference to the root node
        let current_node: &Node<'t, D> = &self.root_node;

        // Ledger with info about nodes we've touched, namely the parent and sibling nodes
        // and distance to their associated space in form of (&usize, f64), where usize is
        // the index inside of self.nodes. The root node is checked at the end.
        let mut points_to_check: Vec<(&usize, &[NotNan<f64>; D], f64)> =
            Vec::with_capacity(self.height_hint);

        // Initialize candidate container with dummy point
        let mut container = Container::new(k.min(self.data_index.len()));
        container.push((std::f64::MAX, current_node.stem_position()));

        // Recurse down (and then up and down) the stem
        self.check_stem_k(
            query,
            current_node,
            &mut container,
            &mut points_to_check,
        );


        container.index(&self.data_index)
    }

    #[cfg(not(feature = "do-not-return-position"))]
    fn query_nearest_k_periodic<'q, 'i>(
        &'q self,
        query: &'q [NotNan<f64>; D],
        k: usize,
        boxsize: &[NotNan<f64>; D],
    ) -> Vec<(f64, u64, &'i [NotNan<f64>; D])>
    where
        'q: 'i,
        't: 'q,
    {

        // First get real image result
        let mut real_image_container: Container<D> = {

            // Get reference to the root node
            let current_node: &Node<'t, D> = &self.root_node;

            // Ledger with info about nodes we've touched, namely the parent and sibling nodes
            // and distance to their associated space in form of (&usize, f64), where usize is
            // the index inside of self.nodes. The root node is checked at the end.
            let mut points_to_check: Vec<(&usize, &[NotNan<f64>; D], f64)> =
                Vec::with_capacity(self.height_hint);

            // Initialize candidate container with dummy point
            let mut container = Container::new(k.min(self.data_index.len()));
            container.push((std::f64::MAX, current_node.stem_position()));

            // Recurse down (and then up and down) the stem
            self.check_stem_k(
                query,
                current_node,
                &mut container,
                &mut points_to_check,
            );

            container
        };

        // Find closest dist2 to every side
        let mut closest_side_dist2 = [0.0_f64; D];
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
                    // safety: made safe by const generic
                    Some( unsafe { closest_side_dist2.get_unchecked(side) } )
                } else { None })
                .fold(0.0, |acc, x| acc + x);

            if dist_to_side_edge_or_other < *best_real_dist2 {

                let mut image_to_check = query.clone();
                
                for (idx, flag) in closest_image.enumerate() {

                    // If moving image along this dimension
                    if flag {

                        // Do a single index here. This is equal to distance to lower side
                        // safety: made safe by const generic
                        let query_component: &NotNan<f64> =  unsafe { query.get_unchecked(idx) };

                        // Single index here as well
                        // safety: made safe by const generic
                        let boxsize_component = unsafe { boxsize.get_unchecked(idx) };

                        // safety: made safe by const generic
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
        for image in images_to_check {

            // Ledger with info about nodes we've touched, namely the parent and sibling nodes
            // and distance to their associated space in form of (&usize, f64), where usize is
            // the index inside of self.nodes. The root node is checked at the end.
            let mut points_to_check: Vec<(&usize, &[NotNan<f64>; D], f64)> =
                Vec::with_capacity(self.height_hint);

            // Get image result
            self.check_stem_k(
                &image,
                &self.root_node,
                &mut real_image_container,
                &mut points_to_check,
            );
        }

        real_image_container.index(&self.data_index)
    }

    #[cfg(feature = "do-not-return-position")]
    pub fn query_nearest_k<'q>(
        &'q self,
        query: &'q [f64; D],
        k: usize,
    ) -> FnntwResult<Vec<(f64, u64)>> {

        // Check for valid query point
        let query: &[NotNan<f64>; D] = check_point(query)?;

        if let Some(ref boxsize) = self.boxsize {
            
            // Periodic query
            Ok(self.query_nearest_k_periodic(query, k, boxsize))
        } else {

            // Nonperiodic query
            Ok(self.query_nearest_k_nonperiodic(query, k))
        }
    }

    #[cfg(feature = "do-not-return-position")]
    fn query_nearest_k_nonperiodic<'q>(
        &'q self,
        query: &'q [NotNan<f64>; D],
        k: usize,
    ) -> Vec<(f64, u64)>
    where
        't: 'q,
    {

        // Get reference to the root node
        let current_node: &Node<'t, D> = &self.root_node;

        // Ledger with info about nodes we've touched, namely the parent and sibling nodes
        // and distance to their associated space in form of (&usize, f64), where usize is
        // the index inside of self.nodes. The root node is checked at the end.
        let mut points_to_check: Vec<(&usize, &[NotNan<f64>; D], f64)> =
            Vec::with_capacity(self.height_hint);

        // Initialize candidate container with dummy point
        let mut container = Container::new(k.min(self.data_index.len()));
        container.push((std::f64::MAX, current_node.stem_position()));

        // Recurse down (and then up and down) the stem
        self.check_stem_k(
            query,
            current_node,
            &mut container,
            &mut points_to_check,
        );


        container.index(&self.data_index)
    }

    #[cfg(feature = "do-not-return-position")]
    fn query_nearest_k_periodic<'q, 'i>(
        &'q self,
        query: &'q [NotNan<f64>; D],
        k: usize,
        boxsize: &[NotNan<f64>; D],
    ) -> Vec<(f64, u64)>
    where
        'q: 'i,
        't: 'q,
    {

        // First get real image result
        let mut real_image_container: Container<D> = {

            // Get reference to the root node
            let current_node: &Node<'t, D> = &self.root_node;

            // Ledger with info about nodes we've touched, namely the parent and sibling nodes
            // and distance to their associated space in form of (&usize, f64), where usize is
            // the index inside of self.nodes. The root node is checked at the end.
            let mut points_to_check: Vec<(&usize, &[NotNan<f64>; D], f64)> =
                Vec::with_capacity(self.height_hint);

            // Initialize candidate container with dummy point
            let mut container = Container::new(k.min(self.data_index.len()));
            container.push((std::f64::MAX, current_node.stem_position()));

            // Recurse down (and then up and down) the stem
            self.check_stem_k(
                query,
                current_node,
                &mut container,
                &mut points_to_check,
            );

            container
        };

        // Find closest dist2 to every side
        let mut closest_side_dist2 = [0.0_f64; D];
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
                    // safety: made safe by const generic
                    Some( unsafe { closest_side_dist2.get_unchecked(side) } )
                } else { None })
                .fold(0.0, |acc, x| acc + x);

            if dist_to_side_edge_or_other < *best_real_dist2 {

                let mut image_to_check = query.clone();
                
                for (idx, flag) in closest_image.enumerate() {

                    // If moving image along this dimension
                    if flag {

                        // Do a single index here. This is equal to distance to lower side
                        // safety: made safe by const generic
                        let query_component: &NotNan<f64> =  unsafe { query.get_unchecked(idx) };

                        // Single index here as well
                        // safety: made safe by const generic
                        let boxsize_component = unsafe { boxsize.get_unchecked(idx) };

                        // safety: made safe by const generic
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
        for image in images_to_check {

            // Ledger with info about nodes we've touched, namely the parent and sibling nodes
            // and distance to their associated space in form of (&usize, f64), where usize is
            // the index inside of self.nodes. The root node is checked at the end.
            let mut points_to_check: Vec<(&usize, &[NotNan<f64>; D], f64)> =
                Vec::with_capacity(self.height_hint);

            // Get image result
            self.check_stem_k(
                &image,
                &self.root_node,
                &mut real_image_container,
                &mut points_to_check,
            );
        }

        real_image_container.index(&self.data_index)
    }

    #[inline(always)]
    /// Upon checking that we are close to some other space during upward traversal of the tree,
    /// this function is called to check candidates in the child space, appending any new candidate spaces
    /// as we go along
    fn check_child_k<'i, 'o>(
        &'i self,
        query: &'o [NotNan<f64>; D],
        sibling: &usize,
        // check's the parent of the sibling (also our parent)
        stem_position: &'i [NotNan<f64>; D],
        container: &'o mut Container<'i, D>,
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
                self.check_parent_k(
                    query,
                    stem_position,
                    container
                );
                self.check_leaf_k(query, points, container)
            }

            // Sibling is a parent (e.g. for unbalanced tree)
            Node::Stem { .. } => {
                self.check_parent_k(
                    query,
                    stem_position,
                    container
                );
                self.check_stem_k(
                    query,
                    sibling,
                    container,
                    points_to_check,
                )
            }
        }
    }

    #[inline(always)]
    fn check_leaf_k<'i, 'o>(
        &self,
        query: &'o [NotNan<f64>; D],
        leaf_points: impl IntoIterator<Item = &'o &'i [NotNan<f64>; D]>,
        container: &'o mut Container<'i, D>,
    ) where
        'i: 'o,
    {
        // Check all points in leaf
        for candidate in leaf_points.into_iter() {
            new_best_kth(
                query,
                candidate,
                container
            );
        }
    }

    /// If sibling is a stem, then we need to recurse back down
    #[inline(always)]
    fn check_stem_k<'i, 'o>(
        &'i self,
        query: &'o [NotNan<f64>; D],
        stem: &'i Node<'t, D>,
        container: &'o mut Container<'i, D>,
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
        self.check_leaf_k(
            query,
            current_node.iter(),
            container
        );

        // Now we empty out the queue
        while let Some((sibling, parent, dist_sq_to_space)) = points_to_check.pop() {
            let better_dist2 = dist_sq_to_space < *container.best_dist2();
            if better_dist2 {
                self.check_child_k(
                    query,
                    sibling,
                    parent,
                    container,
                    points_to_check,
                );
            }
        }
    }

    #[inline(always)]
    fn check_parent_k<'i, 'o>(
        &self,
        query: &[NotNan<f64>; D],
        stem_position: &'i [NotNan<f64>; D],
        container: &'o mut Container<'i, D>,
    ) where
        'i: 'o,
        't: 'i,
    {
        new_best_kth(
            query,
            stem_position,
            container,
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
