use approx_eq::assert_approx_eq;
use fnntw::{NotNan, Tree};

fn main() {
    // Define some data
    let data = [
        [NotNan::new(0.6).unwrap(), NotNan::new(0.2).unwrap()],
        [NotNan::new(0.1).unwrap(), NotNan::new(0.3).unwrap()],
        [NotNan::new(0.4).unwrap(), NotNan::new(0.9).unwrap()],
        [NotNan::new(0.7).unwrap(), NotNan::new(0.5).unwrap()],
        [NotNan::new(0.7).unwrap(), NotNan::new(0.5).unwrap()],
        [NotNan::new(0.7).unwrap(), NotNan::new(0.5).unwrap()],
        [NotNan::new(0.7).unwrap(), NotNan::new(0.5).unwrap()],
    ];

    // Build the tree (non_parallel build for small tree)
    let leafsize = 4;
    let tree = Tree::new(&data, leafsize).expect(
        "doesn't fail if data.len() > 4, leafsize => 4 
                (and you aren't oom or something)",
    );

    // Define some query point
    let query = [NotNan::new(0.6).unwrap(), NotNan::new(0.1).unwrap()];

    // Query the tree
    let (distance_squared, index, neighbor) = tree.query_nearest(&query);

    // Check that the distance squared is what we expect
    const TOLERANCE: f64 = 1e-6;
    assert_approx_eq!(distance_squared, 0.1 * 0.1, TOLERANCE);

    // Check that the index is what we expect
    assert_eq!(index, 0,);

    // Check that the neighbor is the one we expect
    assert_eq!(neighbor, &data[0],);

    println!("Success")
}
