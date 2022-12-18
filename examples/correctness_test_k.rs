use fnntw::Tree;
use ndarray::Array2;
use ndarray_npy::write_npy;
use rayon::prelude::*;
use std::error::Error;

const K: usize = 128;
const DIMS: usize = 3;
const NDATA: usize = 100_000;
const NQUERY: usize = 100_000;
const DATA_FILE: &'static str = "data.npy";
const QUERY_FILE: &'static str = "query.npy";
const RESULT_FILE: &'static str = "results.npy";
const INDICES_FILE: &'static str = "indices.npy";

fn main() -> Result<(), Box<dyn Error>> {
    // Gather data and query points
    let data = get_data();
    let queries = get_queries();
    save_data_queries(&data, &queries)?;

    // Build tree
    let leafsize = 32;
    let tree = Tree::new_parallel(&data, leafsize, 1).unwrap();
    println!("Built tree");

    // Query tree, in parallel
    let (sqdists, indices): (Array2<f64>, Array2<u64>) = {
        let result: (Vec<f64>, Vec<u64>) = queries
            .par_iter()
            .map_with(&tree, |t, q| {
                let result = t.query_nearest_k(q, K).unwrap();
                result
                    .iter()
                    .map(|n| (n.0, n.1))
                    .collect::<Vec<(f64, u64)>>()
            })
            .flatten()
            .unzip();

        (
            Array2::from_shape_vec((NQUERY, K), result.0).unwrap(),
            Array2::from_shape_vec((NQUERY, K), result.1).unwrap(),
        )
    };
    println!("Queried");

    save_results(sqdists, indices)?;

    drop(tree);
    Ok(())
}

#[inline]
fn get_data() -> Vec<[f64; DIMS]> {
    gen_points::<NDATA>()
}

fn get_queries() -> Vec<[f64; DIMS]> {
    gen_points::<NQUERY>()
}

#[inline]
fn gen_points<const N: usize>() -> Vec<[f64; DIMS]> {
    (0..N).map(|_| [(); DIMS].map(|_| rand::random())).collect()
}

fn save_results(sqdists: Array2<f64>, indices: Array2<u64>) -> Result<(), Box<dyn Error>> {
    write_npy(RESULT_FILE, &sqdists)?;
    write_npy(INDICES_FILE, &indices)?;
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
