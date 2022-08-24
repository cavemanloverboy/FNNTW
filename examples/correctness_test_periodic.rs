use fnntw::Tree;
use ndarray_npy::write_npy;
use ordered_float::NotNan;
use rayon::prelude::*;
use std::error::Error;

// const DIMS: usize = 2;
// const NDATA: usize = 10;
// const NQUERY: usize = 1000;
const DIMS: usize = 3;
const NDATA: usize = 100_000;
const NQUERY: usize = 1_000_000;
const DATA_FILE: &'static str = "data.npy";
const QUERY_FILE: &'static str = "query.npy";
const RESULT_FILE: &'static str = "results.npy";
const INDICES_FILE: &'static str = "indices.npy";

fn main() -> Result<(), Box<dyn Error>> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(48)
        .build_global()?;

    // Gather data and query points
    let data = get_data();
    let queries = get_queries();
    save_data_queries(&data, &queries)?;

    // Build tree
    let leafsize = 4;
    println!("Built tree");
    let tree = Tree::new_parallel(&data, leafsize, 3).unwrap();
    
    // Query tree, in parallel
    let (sqdists, indices): (Vec<f64>, Vec<u64>) = queries
        .par_iter()
        // .iter()
        .map_with(&tree, |t, q| {
            let result = t.query_nearest_periodic(q, &[1.0; 3]);
            (result.0, result.1)
        })
        // .map(|q| tree.query_nearest(q))
        .unzip();
    println!("Queried");

    save_results(sqdists, indices)
}

fn get_data() -> Vec<[f64; DIMS]> {
    gen_points::<NDATA>()
}

fn get_queries() -> Vec<[f64; DIMS]> {
    gen_points::<NQUERY>()
}

fn gen_points<const N: usize>() -> Vec<[f64; DIMS]> {
    (0..N)
        .map(|_| [(); DIMS].map(|_| rand::random()))
        .collect()
}

fn save_results(sqdists: Vec<f64>, indices: Vec<u64>) -> Result<(), Box<dyn Error>> {
    write_npy(RESULT_FILE, &ndarray::Array1::from_vec(sqdists))?;
    write_npy(INDICES_FILE, &ndarray::Array1::from_vec(indices))?;
    Ok(())
}

fn save_data_queries(
    data: &Vec<[f64; DIMS]>,
    queries: &Vec<[f64; DIMS]>,
) -> Result<(), Box<dyn Error>> {
    let flat_data: Vec<f64> = data.iter().cloned().flatten().collect();
    let flat_query: Vec<f64> = queries.iter().cloned().flatten().collect();
    write_npy(
        DATA_FILE,
        &ndarray::Array2::from_shape_vec((NDATA, DIMS), flat_data)?,
    )?;
    write_npy(
        QUERY_FILE,
        &ndarray::Array2::from_shape_vec((NQUERY, DIMS), flat_query)?,
    )?;
    Ok(())
}
