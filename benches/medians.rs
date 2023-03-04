use adqselect::nth_element;
use kth::SliceExtKth;
use num_format::{Locale, ToFormattedString};
use ordered_float::NotNan;
use pdqselect::select_by_key;
use rand::random;

use std::time::Instant;

type T = f64;

fn main() {
    bench_medians()
}

fn bench_medians() {
    const NUM_DATA: usize = 100_000;
    const D: usize = 3;
    const RUNS: usize = 1_000;

    let mut adq_time = 0;
    let mut pdq_time = 0;
    let mut std_time = 0;
    let mut std_time2 = 0;
    let mut fr_time = 0;
    let mut os_time = 0;
    let mut kth_time = 0;

    for _ in 0..RUNS {
        let data = [(); NUM_DATA]
            .map(|_| [(); 3].map(|_| unsafe { NotNan::new_unchecked(random::<T>()) }))
            .to_vec();

        let mut adq_data = data.clone();
        let mut pdq_data = data.clone();
        let mut std_data = data.clone();
        let mut std_data2 = data.clone();
        let mut fr_data = data.clone();
        let mut os_data = data.clone();
        let mut kth_data = data.clone();

        let timer = Instant::now();
        nth_element(&mut adq_data, NUM_DATA / 2, &mut |a, b| {
            a[D / 2].cmp(&b[D / 2])
        });
        adq_time += timer.elapsed().as_nanos();

        let timer = Instant::now();
        select_by_key(&mut pdq_data, NUM_DATA / 2, |a| unsafe {
            *a.get_unchecked(D / 2)
        });
        pdq_time += timer.elapsed().as_nanos();

        let timer = Instant::now();
        std_data.select_nth_unstable_by_key(NUM_DATA / 2, |a| unsafe { *a.get_unchecked(D / 2) });
        std_time += timer.elapsed().as_nanos();

        let timer = Instant::now();
        std_data2.select_nth_unstable_by(NUM_DATA / 2, |a, b| unsafe {
            a.get_unchecked(D / 2).cmp(&b.get_unchecked(D / 2))
        });
        std_time2 += timer.elapsed().as_nanos();

        let timer = Instant::now();
        floydrivest::nth_element(&mut fr_data, NUM_DATA / 2, &mut |a, b| {
            a[D / 2].cmp(&b[D / 2])
        });
        fr_time += timer.elapsed().as_nanos();

        let timer = Instant::now();
        order_stat::kth_by(&mut os_data, NUM_DATA / 2, |a, b| a[D / 2].cmp(&b[D / 2]));
        os_time += timer.elapsed().as_nanos();

        let timer = Instant::now();
        kth_data.partition_by_kth(NUM_DATA / 2);
        kth_time += timer.elapsed().as_nanos();
    }

    println!(
        "adq time  = {} nanos",
        adq_time.to_formatted_string(&Locale::en)
    );
    println!(
        "pdq time  = {} nanos",
        pdq_time.to_formatted_string(&Locale::en)
    );
    println!(
        "std time  = {} nanos",
        std_time.to_formatted_string(&Locale::en)
    );
    println!(
        "std time2 = {} nanos",
        std_time2.to_formatted_string(&Locale::en)
    );
    println!(
        "fr time = {} nanos",
        fr_time.to_formatted_string(&Locale::en)
    );
    println!(
        "os time = {} nanos",
        os_time.to_formatted_string(&Locale::en)
    );

    println!(
        "kth time = {} nanos",
        kth_time.to_formatted_string(&Locale::en)
    );
}

#[test]
fn test_median() {
    use adqselect::nth_element;

    let mut v = vec![10, 7, 9, 7, 2, 8, 8, 1, 9, 4];
    nth_element(&mut v, 3, &mut Ord::cmp);

    assert_eq!(v[3], 7);
    println!("{v:?}");
}

#[test]
fn test_median_ref() {
    use adqselect::nth_element;

    let data = vec![10, 7, 9, 7, 2, 8, 8, 1, 9, 4];

    let mut v: Vec<&u8> = data.iter().collect();
    nth_element(&mut v, 3, &mut Ord::cmp);

    assert_eq!(v[3], &7);
    println!("{v:?}");
}

#[test]
fn test_median_ref_not_copy_by() {
    use adqselect::nth_element;

    let data: Vec<Vec<u8>> = vec![
        vec![3, 5],
        vec![3, 1],
        vec![5, 2],
        vec![1, 12],
        vec![8, 5],
        vec![6, 6],
    ];

    let mut v: Vec<&Vec<u8>> = data.iter().collect();
    nth_element(&mut v, 3, &mut Ord::cmp);

    assert_eq!(v[3], &vec![5, 2]);
    println!("{v:?}");

    let mut v: Vec<&Vec<u8>> = data.iter().collect();
    nth_element(&mut v, 3, &mut |a, b| {
        a[1].cmp(&b[1])
            // This part is only here to get
            // deterministic result for this test
            .then(a[0].cmp(&b[0]))
    });

    assert_eq!(v[3], &vec![8, 5]);
    println!("{v:?}");
}
