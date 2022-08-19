use ordered_float::NotNan;
use crate::{Tree, Node, distance::*};

impl<'t, const D: usize> Tree<'t, D> {

    /// Given a query point `query`, query the tree and return point's nearest neighbor.
    /// The value returned is (`distance_to_neighbor: f64`, `neighbor_index: u64`, `neighbor_position: &'t [NotNan<f64>; D]`).
    pub fn query_nearest(
        &'t self,
        query: &[NotNan<f64>; D]
    ) -> (f64, u64, &'t [NotNan<f64>; D]) {

        // Initialize a reference to the root node
        let current_node: &Node<'t, D> = unsafe { self.nodes.last().unwrap_unchecked() };
    
        // Ledger with info about nodes we've touched, namely the parent and sibling nodes
        // and distance to their associated space in form of (&usize, f64), where usize is 
        // the index inside of self.nodes. The root node is checked at the end.
        //
        // TODO: test usize vs &usize
        let mut points_to_check: Vec<(&usize, &[NotNan<f64>; D], f64)> = Vec::with_capacity(self.height_hint);

        let mut current_best_dist_sq = std::f64::MAX;
        let mut current_best_neighbor: &'t [NotNan<f64>; D] = unsafe { self.nodes.last().unwrap_unchecked().stem_position() };

        // Recurse down (and then up and down) the stem
        self.check_stem(query, current_node, &mut current_best_dist_sq, &mut current_best_neighbor, &mut points_to_check);

        let best_idx = self.data_index[current_best_neighbor];

        (current_best_dist_sq, best_idx, current_best_neighbor)
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
    ) 
    where
        'i: 'o,
        't: 'i
    {
        let sibling = unsafe { self.nodes.get_unchecked(*sibling) };
        match sibling {
            
            // Sibling is a leaf
            Node::Leaf { points, .. } => {
                // the stem_position here is the position of the parent
                self.check_parent(query, stem_position, current_best_dist_sq, current_best_neighbor);
                self.check_leaf(query, points, current_best_dist_sq, current_best_neighbor)
            },

            // Sibling is a parent (e.g. for unbalanced tree)
            Node::Stem { .. } => {
                self.check_parent(query, stem_position, current_best_dist_sq, current_best_neighbor);
                self.check_stem(query, sibling, current_best_dist_sq, current_best_neighbor, points_to_check)
            }
        }
    }

    #[inline(always)]
    fn check_leaf<'a, 'b>(
        &self,
        query: &'b [NotNan<f64>; D],
        // leaf_points: &'t Vec<&'t [NotNan<f64>; D]>,
        leaf_points: impl IntoIterator<Item = &'b &'a[NotNan<f64>; D]>,
        current_best_dist_sq: &mut f64,
        current_best_neighbor: &mut &'a [NotNan<f64>; D]
    ) 
    where
        'a: 'b
    {
        // Check all points in leaf
        for candidate in leaf_points.into_iter() {   
            new_best(query, candidate, current_best_dist_sq, current_best_neighbor);
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
    ) 
    where
        'i: 'o,
        't: 'i
    {
        
       // Navigate down the stems until we reach a leaf
       let mut current_node = stem;

        while current_node.is_stem() {

            let next_leafnode = match current_node {
            
                Node::Stem { ref split_dim, point, left, ref right, .. } => {
                
                    // Determine left/right split
                    if unsafe { query.get_unchecked(*split_dim) > point.get_unchecked(*split_dim) } {


                        // Record sibling node and the dist_sq to sibling's associated space
                        let (sibling_lower, sibling_upper) = unsafe { self.nodes.get_unchecked(*left) }.get_bounds();
                        let dist_sq_to_space = calc_dist_sq_to_space(query, sibling_lower, sibling_upper);
                        points_to_check.push((left, point, dist_sq_to_space));

                        // Right Branch
                        right

                    } else { 

                        // Record sibling node and the dist_sq to its associated space
                        let (sibling_lower, sibling_upper) = unsafe { self.nodes.get_unchecked(*right) }.get_bounds();
                        let dist_sq_to_space = calc_dist_sq_to_space(query, sibling_lower, sibling_upper);
                        points_to_check.push((right, point, dist_sq_to_space));

                        // Left Branch
                        left
                    }
                },
                _ => unreachable!("we are traversing though stems")
            };

            // Set leafnode
            current_node = unsafe { self.nodes.get_unchecked(*next_leafnode) };
            
        };

        // We are now at a leaf; check it
        self.check_leaf(query, current_node.iter(), current_best_dist_sq, current_best_neighbor);

        // Now we empty out the queue
        // if points_to_check.len() > 1{ println!("{points_to_check:?}"); }
        while let Some((sibling, parent, dist_sq_to_space)) = points_to_check.pop() {
            if dist_sq_to_space < *current_best_dist_sq {
                self.check_child(query, sibling, parent, current_best_dist_sq, current_best_neighbor, points_to_check);
            }
        }
    }

    #[inline(always)]
    fn check_parent<'i, 'o>(
        &self,
        query: &[NotNan<f64>; D],
        stem_position: &'i [NotNan<f64>; D],
        current_best_dist_sq: &'o mut f64,
        current_best_neighbor: &'o mut &'i [NotNan<f64>; D]
    )
    where
        'i: 'o,
        't: 'i,
    {
        new_best(query, stem_position, current_best_dist_sq, current_best_neighbor);
    }
}
