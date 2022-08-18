use ordered_float::NotNan;
use crate::{Tree, Node, distance::*};


impl<'t, const D: usize> Tree<'t, D> {

    // pub fn query_nearest(
    //     &'t self,
    //     query: &'t [NotNan<f64>; D]
    // ) -> &'t [NotNan<f64>; D] {

    //     // Initialize a reference to the root node
    //     let mut current_leafnode: &'t Node<'t, D> = unsafe { self.nodes.last().unwrap_unchecked() };
    
    //     // Navigate down the stems until we reach a leaf
    //     while current_leafnode.is_stem() {

    //         // // Unpack node info
    //         // let (split_dim, point, left, right) = current_leafnode.node_info();
    //         let next_leafnode = match current_leafnode {
            
    //             Node::Stem { ref split_dim, point, ref left, ref right } => {
    //             // Determine left/right split
    //                 if unsafe { query.get_unchecked(*split_dim) > point.get_unchecked(*split_dim) } {

    //                     // Right Branch
    //                     right
    //                 } else { 

    //                     // Left Branch
    //                     left
    //                 }
    //             },
    //             _ => unreachable!("this must be a node")
    //         };

    //         // Set leafnode
    //         current_leafnode = unsafe { self.nodes.get_unchecked(*next_leafnode) };
            
    //     };

    //     // Find closest point in this leaf
    //     // TODO: this is not the nearest neighbor, just the nearest neighbor in this leaf
    //     let neighbor: &'t [NotNan<f64>; D] = current_leafnode
    //         .iter()
    //         .fold((std::f64::MAX, query), |(min_dist, neighbor), node| {
            
    //             // Calculate distance to this point in leaf
    //             let dist = squared_euclidean(query, node);

    //             if min_dist.gt(&dist) {
                    
    //                 // New closest neighbor
    //                 (dist, node)
    //             } else {
    //                 // unchanged
    //                 (min_dist, neighbor)
    //             }

    //         }).1;

    //     neighbor
    // }


    pub fn query_nearest(
        &'t self,
        query: &'t [NotNan<f64>; D]
    ) -> &'t [NotNan<f64>; D] {

        // Initialize a reference to the root node
        let mut current_leafnode: &'t Node<'t, D> = unsafe { self.nodes.last().unwrap_unchecked() };
    
        // Navigate down the stems until we reach a leaf
        while current_leafnode.is_stem() {

            // // Unpack node info
            // let (split_dim, point, left, right) = current_leafnode.node_info();
            let next_leafnode = match current_leafnode {
            
                Node::Stem { ref split_dim, point, ref left, ref right } => {
                // Determine left/right split
                    if unsafe { query.get_unchecked(*split_dim) > point.get_unchecked(*split_dim) } {

                        // Right Branch
                        right
                    } else { 

                        // Left Branch
                        left
                    }
                },
                _ => unreachable!("this must be a node")
            };

            // Set leafnode
            current_leafnode = unsafe { self.nodes.get_unchecked(*next_leafnode) };
            
        };

        // Find closest point in this leaf
        // TODO: this is not the nearest neighbor, just the nearest neighbor in this leaf

        // Initialize best with first entry in leaf
        let mut current_leafnode_iter = current_leafnode.iter();
        let mut current_best_neighbor = unsafe { *current_leafnode_iter.next().unwrap_unchecked() };
        let mut current_best_dist_sq: f64 = squared_euclidean(query, current_best_neighbor);

        // Check all other points in leaf
        for candidate in  current_leafnode_iter {   
            new_best(query, candidate, &mut current_best_dist_sq, &mut current_best_neighbor);
        }

        current_best_neighbor
    }
}


