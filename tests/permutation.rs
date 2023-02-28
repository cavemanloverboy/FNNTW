use permutation::Permutation;

// This is here just to test my understanding of permutation crate
#[test]
fn test_permutation() {
    let mut dists = [10_u8, 0, 20, 40, 30];
    let mut indices = [10, 7, 3, 5, 4];

    let mut perm = permutation::sort_unstable_by(&dists[..], |a, b| a.partial_cmp(&b).unwrap());
    println!("{:?}", perm);
    assert_eq!(perm, Permutation::oneline([1, 0, 2, 4, 3]));

    perm.apply_slice_in_place(&mut indices);
    perm.apply_slice_in_place(&mut dists);

    assert_eq!(dists, [0, 10, 20, 30, 40]);
    assert_eq!(indices, [7, 10, 3, 4, 5]);
}
