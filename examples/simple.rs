use approx_eq::assert_approx_eq;
use fnntw::Tree;

type T = f64;

fn main() {
    // Define some data
    let data = [
        [0.6, 0.2],
        [0.1, 0.3],
        [0.4, 0.9],
        [0.7, 0.5],
        [0.7, 0.5],
        [0.7, 0.5],
        [0.7, 0.5],
    ];

    // Build the tree (non_parallel build for small tree)
    let leafsize = 1;
    let tree = Tree::new_parallel(&data, leafsize, 0).expect(
        "doesn't fail if data.len() > 4, leafsize => 4 
                (and you aren't oom or something)",
    );

    // Define some query point
    let query = [0.6, 0.1];

    // Query the tree
    #[cfg(not(feature = "no-position"))]
    let (distance_squared, index, neighbor) = tree.query_nearest(&query).unwrap();
    #[cfg(feature = "no-position")]
    let (distance_squared, index) = tree.query_nearest(&query).unwrap();

    // Check that the distance squared is what we expect
    const TOLERANCE: T = 1e-6;
    assert_approx_eq!(distance_squared, 0.1 * 0.1, TOLERANCE);

    // Check that the index is what we expect
    assert_eq!(index, 0);

    // Check that the neighbor is the one we expect
    #[cfg(not(feature = "no-position"))]
    assert_eq!(neighbor, &data[0]);

    println!("Success")
}
