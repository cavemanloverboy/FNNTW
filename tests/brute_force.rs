use fnntw::{Tree, NotNan, distance::squared_euclidean};
use rand::{rngs::ThreadRng, Rng};



const NDATA: usize = 100_000;
const NQUERY: usize = 1_000_000;
const D: usize = 3;

fn main() {

    // Random number generator
    let mut rng = rand::thread_rng();

    // Generate random data, query
    let mut data = Vec::with_capacity(NDATA);
    let mut query = Vec::with_capacity(NQUERY);
    for _ in 0..NDATA {
        data.push(random_point(&mut rng));
    }
    for _ in 0..NQUERY {
        query.push(random_point(&mut rng));
    }

    // Construct tree
    let tree = Tree::<'_, D>::new(&data, 32).unwrap();

    // Query tree
    let mut results = Vec::with_capacity(NQUERY);
    for q in &query {
        results.push(tree.query_nearest(q));
    }

    // Brute force check results
    for (i, q) in query.iter().enumerate() {
        assert_eq!(results[i], brute_force(q, &data))
    }

}


fn random_point<const D: usize>(rng: &mut ThreadRng) -> [NotNan<f64>; D] {
    [(); D].map(|_| unsafe { std::mem::transmute::<f64, NotNan<f64>>(rng.gen()) })
}


fn brute_force<'d, const D: usize>(
    q: &[NotNan<f64>; D],
    data: &'d [[NotNan<f64>; D]],
) -> (f64, u64, &'d[NotNan<f64>; D]) {

    let mut best_dist = std::f64::MAX;
    let mut best = (std::f64::MAX, std::u64::MAX, data.get(0).unwrap());
    for (d, i) in data.iter().zip(0..) {
        
        let dist = squared_euclidean(q, d);

        if dist < best_dist {
            best_dist = dist;
            best = (best_dist, i, d)
        }
    }

    best
}