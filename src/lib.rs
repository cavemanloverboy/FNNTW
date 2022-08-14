#![feature(slice_split_at_unchecked)]

use ordered_float::NotNan;
use std::cmp::Ordering;

#[cfg(feature = "timing")]
use num_format::{Locale, ToFormattedString};



// mod medians;
#[cfg(feature = "timing")]
static LEAF_VEC_ALLOC: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "timing")]
static LEAF_WRITE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "timing")]
static NODE_MEDIAN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "timing")]
static NODE_FIRST_SPLIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "timing")]
static NODE_SECOND_SPLIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(feature = "timing")]
static NODE_WRITE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub struct Tree<'t, const D: usize> {
    data: &'t [[NotNan<f64>; D]],
    leafsize: usize,
    nodes: Vec<LeafNode<'t, D>>,
}

#[derive(Debug)]
enum LeafNode<'t, const D: usize> {
    Node {
        split_dim: usize,
        point: &'t [NotNan<f64>; D],
        left: usize,
        right: usize,
    },
    Leaf(Leaf<'t, D>),
}

type Leaf<'t, const D: usize> = Vec<&'t [NotNan<f64>; D]>;

impl<'t, const D: usize> Tree<'t, D> {
    
    pub fn new(
        data: &'t [[NotNan<f64>; D]],
        leafsize: usize,
    ) -> Result<Tree<'t, D>, &'static str> {
        
        // Nonzero Length
        let data_len =  data.len();
        if data_len == 0 {
            return Err("data has zero length")
        }

        // Unsafe operations require leafsize to be at least 4
        // Also probably a good idea to keep above 4 anyway.
        if leafsize < 4 {
            return Err("Choose a leafsize >= 4")
        }

        // Initialize variables for recursive function
        let split_level: usize = 0;
        #[cfg(feature = "timing")]
        let timer = std::time::Instant::now();
        let vec_ref: &mut [&'t [NotNan<f64>; D]] = &mut data.iter().collect::<Vec<_>>();
        #[cfg(feature = "timing")]
        let initial_vec_ref = timer.elapsed().as_nanos();
        let mut nodes = vec![];

        // Run recursive parallel build
        Tree::get_leafnodes(vec_ref, split_level, leafsize, &mut nodes);

        #[cfg(feature = "timing")]
        {
            let total = {
             LEAF_VEC_ALLOC.load(std::sync::atomic::Ordering::SeqCst)
             + LEAF_WRITE.load(std::sync::atomic::Ordering::SeqCst)
             + NODE_MEDIAN.load(std::sync::atomic::Ordering::SeqCst)
             + NODE_FIRST_SPLIT.load(std::sync::atomic::Ordering::SeqCst)
             + NODE_SECOND_SPLIT.load(std::sync::atomic::Ordering::SeqCst)
             + NODE_WRITE.load(std::sync::atomic::Ordering::SeqCst)
            };

            println!("\nINITIAL_VEC_REF = {}", initial_vec_ref.to_formatted_string(&Locale::en));
            println!("LEAF_VEC_ALLOC = {}", LEAF_VEC_ALLOC.load(std::sync::atomic::Ordering::SeqCst).to_formatted_string(&Locale::en));
            println!("LEAF_WRITE = {}", LEAF_WRITE.load(std::sync::atomic::Ordering::SeqCst).to_formatted_string(&Locale::en));
            println!("NODE_MEDIAN = {}", NODE_MEDIAN.load(std::sync::atomic::Ordering::SeqCst).to_formatted_string(&Locale::en));
            println!("NODE_FIRST_SPLIT = {}", NODE_FIRST_SPLIT.load(std::sync::atomic::Ordering::SeqCst).to_formatted_string(&Locale::en));
            println!("NODE_SECOND_SPLIT = {}", NODE_SECOND_SPLIT.load(std::sync::atomic::Ordering::SeqCst).to_formatted_string(&Locale::en));
            println!("NODE_WRITE = {}\n", NODE_WRITE.load(std::sync::atomic::Ordering::SeqCst).to_formatted_string(&Locale::en));
        }

        Ok(Tree {
            data,
            leafsize,
            nodes,
        })
    }


    // A recursive private function.
    fn get_leafnodes<'a>(
        mut subset: &'a mut[&'t [NotNan<f64>; D]],
        mut split_level: usize,
        leafsize: usize,
        nodes: &mut Vec<LeafNode<'t, D>>,
    ) -> usize {

        // Increment split level
        split_level += 1;

        // Get split dimension
        let split_dim = split_level % D;

        // Determine leaf-ness
        let is_leaf =  subset.len() <= leafsize;
        
        match is_leaf {
            true => {

                #[cfg(feature = "timing")]
                let timer = std::time::Instant::now();
                let leaf = LeafNode::Leaf(subset.to_vec());
                #[cfg(feature = "timing")]
                let vec_alloc = timer.elapsed().as_nanos();
                #[cfg(feature = "timing")]
                LEAF_VEC_ALLOC.fetch_add(vec_alloc as usize, std::sync::atomic::Ordering::SeqCst);

                #[cfg(feature = "timing")]
                let timer = std::time::Instant::now();
                let leaf_index = nodes.len();
                nodes.push(leaf);
                #[cfg(feature = "timing")]
                let write_time = timer.elapsed().as_nanos();
                #[cfg(feature = "timing")]
                LEAF_WRITE.fetch_add(write_time as usize, std::sync::atomic::Ordering::SeqCst);

                leaf_index
            },
            false => {
                
                // Calculate index of median
                let median_index = subset.len() / 2;

                #[cfg(feature = "timing")]
                let timer = std::time::Instant::now();
                // Select median in this subset based on split_dim component
                subset.select_nth_unstable_by(median_index, |a, b| { 
                    unsafe { a.get_unchecked(split_dim).cmp(&b.get_unchecked(split_dim)) }
                });
                #[cfg(feature = "timing")]
                let node_median = timer.elapsed().as_nanos();
                #[cfg(feature = "timing")]
                NODE_MEDIAN.fetch_add(node_median as usize, std::sync::atomic::Ordering::SeqCst);

                // let median: &'t [NotNan<f64>; D] = subset.swap_remove(median_index);
                // let left: Vec<&'t [NotNan<f64>; D]> = subset.drain(0..median_index).collect();
                // let right: Vec<&'t [NotNan<f64>; D]> = subset;

                #[cfg(feature = "timing")]
                let timer = std::time::Instant::now();
                let (left, right) = unsafe { subset.split_at_mut_unchecked(median_index) };
                #[cfg(feature = "timing")]
                let node_first_split = timer.elapsed().as_nanos();
                #[cfg(feature = "timing")]
                NODE_FIRST_SPLIT.fetch_add(node_first_split as usize, std::sync::atomic::Ordering::SeqCst);


                #[cfg(feature = "timing")]
                let timer = std::time::Instant::now();
                let (median, right) = unsafe { right.split_at_mut_unchecked(1) };
                let median = unsafe { median.get_unchecked(0) };
                #[cfg(feature = "timing")]
                let node_second_split = timer.elapsed().as_nanos();
                #[cfg(feature = "timing")]
                NODE_SECOND_SPLIT.fetch_add(node_second_split as usize, std::sync::atomic::Ordering::SeqCst);


                let left_handle = Tree::get_leafnodes(left, split_level, leafsize, nodes);
                let right_handle = Tree::get_leafnodes(right, split_level, leafsize, nodes);


                let node = LeafNode::Node {
                    split_dim,
                    point: median,
                    left: left_handle,
                    right: right_handle,
                };

                #[cfg(feature = "timing")]
                let timer = std::time::Instant::now();
                let node_index = nodes.len();
                nodes.push(node);
                #[cfg(feature = "timing")]
                let node_write = timer.elapsed().as_nanos();
                #[cfg(feature = "timing")]
                NODE_WRITE.fetch_add(node_write as usize, std::sync::atomic::Ordering::SeqCst);

                node_index
            }
        }
    }

    pub fn size(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {

    use crate::Tree;
    use ordered_float::NotNan;
    use concat_idents::concat_idents;
    use seq_macro::seq;

    // Generate 1..16 dimensional size=1 kd tree unit tests
    macro_rules! size_one_kdtree {
        ($d:ident) => {
            concat_idents!(test_name = test_make_, $d, _, dtree, {
                #[test]
                fn test_name() {
            
                    let leafsize = 16; 

                    let data: Vec<_> = (0..leafsize).map(|x| {
                        [NotNan::new(x as f64).unwrap(); $d]
                    }).collect();
            
            
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

        let data: Vec<[NotNan<f64>; 1]> = [
            [(); 32].map(|_| unsafe { [NotNan::new_unchecked(0.1)] }).as_ref(),
            [(); 1].map(|_| unsafe { [NotNan::new_unchecked(0.5)] }).as_ref(),
            [(); 32].map(|_| unsafe { [NotNan::new_unchecked(0.9)] }).as_ref(),
        ].concat();

        let leafsize = 32;

        let tree = Tree::new(&data, leafsize).unwrap();
        for node in &tree.nodes {
            println!("{node:?}")
        }
        assert_eq!(tree.size(), 3);
    }
}

