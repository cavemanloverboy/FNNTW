use std::error::Error;
use fnstw::Tree;
use ordered_float::NotNan;

use std::time::{Instant, Duration};
use num_format::{Locale, ToFormattedString};

const D: usize = 3;


fn tree_build_bench() {

    const RUNS: u128 = 100;

    let mut build_time = 0;

    for _ in 0..RUNS {

        // Bench building tree
        for ndata in [5].map(|p| 10_usize.pow(p)){

            let data: Vec<[NotNan<f64>; D]> = (0..ndata)
                .map(|_| [(); D].map(|_| unsafe { NotNan::new_unchecked(rand::random()) } ))
                .collect();

            let time = Instant::now();
            let tree = Tree::new(&data, 32).unwrap();
            build_time += time.elapsed().as_nanos();
        }
    }

    println!("average build time is {} nanos", (build_time / RUNS).to_formatted_string(&Locale::en));
}

fn main() {
    tree_build_bench()
}