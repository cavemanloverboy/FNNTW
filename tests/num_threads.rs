use rayon::prelude::{IntoParallelIterator, ParallelIterator};

#[test]
fn test_num_threads() {
    (0..1000).into_par_iter().for_each(|_| {
        assert!(rayon::current_thread_index().unwrap() <= rayon::max_num_threads());
    })
}
