use fnntw::Tree as FNNTWTree;
use ndarray::Array2;
use numpy::ndarray::{Array1, ArrayView2};
use numpy::*;
use ouroboros::*;
use pyo3::{exceptions, prelude::*};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn pyfnntw(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyclass]
    struct Tree(Box<dyn FNTree + Send + Sync>);

    #[pymethods]
    impl Tree {
        #[new]
        fn new(
            data: PyReadonlyArray2<'_, f64>,
            leafsize: usize,
            boxsize: Option<PyReadonlyArray1<'_, f64>>,
            par_split_level: Option<usize>,
        ) -> PyResult<Tree> {
            // If a boxsize is specified for periodic boundary conditions,
            // check that it is the same dimension as the data
            if let Some(ref boxsize) = boxsize {
                if boxsize.shape()[0] != data.shape()[1] {
                    return Err(PyErr::new::<exceptions::PyTypeError, _>(
                        "the dimensions of boxsize does not match the dimensions of data.",
                    ));
                }
            }

            // Enable parallelism
            let threads = std::thread::available_parallelism()?.into();
            match rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
            {
                Ok(_) => println!("Parallelism activated: {threads} threads"),
                err => {
                    let err = err.unwrap_err();
                    if !err
                        .to_string()
                        .contains("The global thread pool has already been initialized")
                    {
                        println!("Unable to activate parallelism: {err}")
                    }
                }
            }

            // Check dimensions of data
            let dims: [usize; 2] = data
                .shape()
                .try_into()
                .expect("2D array should definitely have 2 dims");

            // Check dimensionality is as expected
            match dims[1] {
                2 => {
                    // Transform the data into a slice of arrays,
                    // which is what the library expects
                    //
                    // safety: converting to static as part of python requirements.
                    // can lead to (read-only) dangling pointers if data is collected by python gc.
                    let data: &[[f64; 2]] = unsafe {
                        std::mem::transmute(slice_as_chunks::<f64, 2>(
                            data.as_array().as_slice().unwrap(),
                        ))
                    };

                    // Build the tree
                    let tree: Tree2 = Tree2Builder {
                        data,
                        tree_builder: |data: &&[[f64; 2]]| {
                            // Build the tree
                            let mut tree = if let Some(psl) = par_split_level {
                                FNNTWTree::<'_, 2>::new_parallel(*data, leafsize, psl)
                                    .expect("failed to build tree")
                            } else {
                                FNNTWTree::<'_, 2>::new(data, leafsize)
                                    .expect("failed to build tree")
                            };

                            // Add a boxsize for peridic boundary conditions, if specified
                            if let Some(boxsize) = boxsize {
                                tree = tree
                                    .with_boxsize(boxsize.as_slice().unwrap().try_into().unwrap())
                                    .expect("failed to apply periodic boundary conditions");
                            }

                            tree
                        },
                    }
                    .build();

                    Ok(Tree(Box::new(tree)))
                }
                3 => {
                    // Transform the data into a slice of arrays,
                    // which is what the library expects
                    //
                    // safety: converting to static as part of python requirements.
                    // can lead to (read-only) dangling pointers if data is collected by python gc.
                    let data: &[[f64; 3]] = unsafe {
                        std::mem::transmute(slice_as_chunks::<f64, 3>(
                            data.as_array().as_slice().unwrap(),
                        ))
                    };

                    // Build the tree
                    let tree: Tree3 = Tree3Builder {
                        data,
                        tree_builder: |data: &&[[f64; 3]]| {
                            // Build the tree
                            let mut tree = if let Some(psl) = par_split_level {
                                FNNTWTree::<'_, 3>::new_parallel(*data, leafsize, psl)
                                    .expect("failed to build tree")
                            } else {
                                FNNTWTree::<'_, 3>::new(data, leafsize)
                                    .expect("failed to build tree")
                            };

                            // Add a boxsize for peridic boundary conditions, if specified
                            if let Some(boxsize) = boxsize {
                                tree = tree
                                    .with_boxsize(boxsize.as_slice().unwrap().try_into().unwrap())
                                    .expect("failed to apply periodic boundary conditions");
                            }

                            tree
                        },
                    }
                    .build();

                    Ok(Tree(Box::new(tree)))
                }
                _ => {
                    return Err(PyErr::new::<exceptions::PyTypeError, _>(
                        "Only 2D, and 3D are supported at the moment.",
                    ))
                }
            }
        }

        fn query(
            slf: PyRef<'_, Self>,
            query: PyReadonlyArray2<'_, f64>,
            k: Option<usize>,
        ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
            if let Some(k) = k {
                // unflattened branch
                let (distances, indices) = slf.0.query_k(query.as_array(), k)?;
                let distances = Array2::from_shape_vec((query.shape()[0], k), distances)
                    .expect("shape is known");
                let indices =
                    Array2::from_shape_vec((query.shape()[0], k), indices).expect("shape is known");
                Ok((
                    distances.into_pyarray(slf.py()).into(),
                    indices.into_pyarray(slf.py()).into(),
                ))
            } else {
                // k = 1 flattened branch
                let (distances, indices) = slf.0.query(query.as_array())?;
                let distances = Array1::from_vec(distances);
                let indices = Array1::from_vec(indices);
                Ok((
                    distances.into_pyarray(slf.py()).into(),
                    indices.into_pyarray(slf.py()).into(),
                ))
            }
        }
    }

    m.add_class::<Tree>()?;
    Ok(())
}

// #[pyclass]
#[self_referencing]
struct Tree2<'a> {
    // data: Array1<[NotNan<f64>; 2]>,
    data: &'a [[f64; 2]],
    #[borrows(data)]
    #[covariant]
    pub tree: FNNTWTree<'this, 2>,
}

// #[pyclass]
#[self_referencing]
struct Tree3<'a> {
    data: &'a [[f64; 3]],
    #[borrows(data)]
    #[covariant]
    pub tree: FNNTWTree<'this, 3>,
}

impl<'a> FNTree for Tree2<'a> {
    fn query(&self, query: ArrayView2<f64>) -> PyResult<(Vec<f64>, Vec<u64>)> {
        // Check dimensions of data
        let dims: [usize; 2] = query
            .shape()
            .try_into()
            .expect("2D array should definitely have 2 dims");

        // Return error if not 2D
        if dims[1] != 2 {
            return Err(PyErr::new::<exceptions::PyTypeError, _>(
                "A 3D tree can only be queried with 3D data",
            ));
        }

        // Transform slice of floats into slice of arrays
        let query: &[[f64; 2]] = slice_as_chunks::<f64, 2>(query.as_slice().unwrap());

        let tree = self.borrow_tree();
        let query_size = query.len();
        let mut distances = Vec::with_capacity(query_size);
        let mut indices = Vec::with_capacity(query_size);
        let dist_ptr_usize = distances.as_mut_ptr() as usize;
        let idx_ptr_usize = indices.as_mut_ptr() as usize;
        query.into_par_iter().enumerate().for_each_with(
            (dist_ptr_usize, idx_ptr_usize),
            |(d, i), (j, q)| {
                let result = tree.query_nearest(q).expect("error occurred during query");
                let d = *d as *mut f64;
                let i = *i as *mut u64;
                unsafe {
                    *d.add(j) = result.0;
                    *i.add(j) = result.1;
                }
            },
        );

        unsafe {
            distances.set_len(query.len());
            indices.set_len(query.len());
        }
        Ok((distances, indices))
    }

    fn query_k(&self, query: ArrayView2<f64>, k: usize) -> PyResult<(Vec<f64>, Vec<u64>)> {
        // Check dimensions of data
        let dims: [usize; 2] = query
            .shape()
            .try_into()
            .expect("2D array should definitely have 2 dims");

        // Return error if not 2D
        if dims[1] != 2 {
            return Err(PyErr::new::<exceptions::PyTypeError, _>(
                "A 3D tree can only be queried with 3D data",
            ));
        }

        // Transform slice of floats into slice of arrays
        let query: &[[f64; 2]] = slice_as_chunks::<f64, 2>(query.as_slice().unwrap());

        let tree = self.borrow_tree();
        let mut distances = Vec::with_capacity(query.len() * k);
        let mut indices = Vec::with_capacity(query.len() * k);
        let dist_ptr_usize = distances.as_mut_ptr() as usize;
        let idx_ptr_usize = indices.as_mut_ptr() as usize;
        query.into_par_iter().enumerate().for_each_with(
            (dist_ptr_usize, idx_ptr_usize),
            |(d, i), (j, q)| {
                let result = tree
                    .query_nearest_k(q, k)
                    .expect("error occurred during query");
                unsafe {
                    let d = *d as *mut f64;
                    let i = *i as *mut u64;
                    for kk in 0..k {
                        let query_k = result.get_unchecked(kk);
                        *d.add(k * j + kk) = query_k.0;
                        *i.add(k * j + kk) = query_k.1;
                    }
                }
            },
        );

        unsafe {
            distances.set_len(query.len() * k);
            indices.set_len(query.len() * k);
        }

        Ok((distances, indices))
    }
}

impl<'a> FNTree for Tree3<'a> {
    fn query(&self, query: ArrayView2<f64>) -> PyResult<(Vec<f64>, Vec<u64>)> {
        // Check dimensions of data
        let dims: [usize; 2] = query
            .shape()
            .try_into()
            .expect("2D array should definitely have 2 dims");

        // Return error if not 3D
        if dims[1] != 3 {
            return Err(PyErr::new::<exceptions::PyTypeError, _>(
                "A 3D tree can only be queried with 3D data",
            ));
        }

        // Transform slice of floats into slice of arrays
        let query: &[[f64; 3]] = slice_as_chunks::<f64, 3>(query.as_slice().unwrap());

        let tree = self.borrow_tree();
        let query_size = query.len();
        let mut distances = Vec::with_capacity(query_size);
        let mut indices = Vec::with_capacity(query_size);
        let dist_ptr_usize = distances.as_mut_ptr() as usize;
        let idx_ptr_usize = indices.as_mut_ptr() as usize;
        query.into_par_iter().enumerate().for_each_with(
            (dist_ptr_usize, idx_ptr_usize),
            |(d, i), (j, q)| {
                let result = tree.query_nearest(q).expect("error occurred during query");
                let d = *d as *mut f64;
                let i = *i as *mut u64;
                unsafe {
                    *d.add(j) = result.0;
                    *i.add(j) = result.1;
                }
            },
        );

        unsafe {
            distances.set_len(query.len());
            indices.set_len(query.len());
        }

        // Safe alternative (comparable; barely slower)
        // query
        //     .into_par_iter()
        //     .map_with(tree, |t, q| {
        //         t.query_nearest(q).expect("error occurred during query")
        //     })
        //     .unzip_into_vecs(&mut distances, &mut indices);
        Ok((distances, indices))
    }

    fn query_k(&self, query: ArrayView2<f64>, k: usize) -> PyResult<(Vec<f64>, Vec<u64>)> {
        // Check dimensions of data
        let dims: [usize; 2] = query
            .shape()
            .try_into()
            .expect("2D array should definitely have 2 dims");

        // Return error if not 3D
        if dims[1] != 3 {
            return Err(PyErr::new::<exceptions::PyTypeError, _>(
                "A 3D tree can only be queried with 3D data",
            ));
        }

        // Transform slice of floats into slice of arrays
        let query: &[[f64; 3]] = slice_as_chunks::<f64, 3>(query.as_slice().unwrap());

        let tree = self.borrow_tree();

        // Unsafe
        let mut distances = Vec::with_capacity(query.len() * k);
        let mut indices = Vec::with_capacity(query.len() * k);
        let dist_ptr_usize = distances.as_mut_ptr() as usize;
        let idx_ptr_usize = indices.as_mut_ptr() as usize;
        query.into_par_iter().enumerate().for_each_with(
            (dist_ptr_usize, idx_ptr_usize),
            |(d, i), (j, q)| {
                let result = tree
                    .query_nearest_k(q, k)
                    .expect("error occurred during query");
                unsafe {
                    let d = *d as *mut f64;
                    let i = *i as *mut u64;
                    for kk in 0..k {
                        let query_k = result.get_unchecked(kk);
                        *d.add(k * j + kk) = query_k.0;
                        *i.add(k * j + kk) = query_k.1;
                    }
                }
            },
        );

        unsafe {
            distances.set_len(query.len() * k);
            indices.set_len(query.len() * k);
        }

        // // Safe alternative (very slow)
        // let (distances, indices) = query
        //     .into_par_iter()
        //     .map_with(tree, |t, q| {
        //         t.query_nearest_k(q, k)
        //             .expect("error occurred during query")
        //     })
        //     .flatten()
        //     .unzip();

        Ok((distances, indices))
    }
}

trait FNTree {
    /// The inner vec is a 1d array containing the distances/indices
    /// for every neighbor
    fn query(&self, query: ArrayView2<f64>) -> PyResult<(Vec<f64>, Vec<u64>)>;

    /// The inner vec is a flattened 2d array!
    fn query_k(&self, query: ArrayView2<f64>, k: usize) -> PyResult<(Vec<f64>, Vec<u64>)>;
}

#[allow(unused)]
unsafe fn slice_to_array<T, const K: usize>(xs: &[T]) -> &[T; K] {
    ::core::mem::transmute(xs.as_ptr())
}

/// ONLY USE FOR EXACT CHUNKS!!!! REMAINDER IS DISCARDED !!!
pub fn slice_as_chunks<T, const N: usize>(slice: &[T]) -> &[[T; N]] {
    assert_ne!(N, 0);
    let len = slice.len() / N;
    let (multiple_of_n, _) = slice.split_at(len * N);
    // SAFETY: We already panicked for zero, and ensured by construction
    // that the length of the subslice is a multiple of N.
    let array_slice = unsafe { as_chunks_unchecked(multiple_of_n) };
    // (array_slice, remainder)
    array_slice
}

#[inline]
#[allow(unused_unsafe)]
pub unsafe fn as_chunks_unchecked<T, const N: usize>(slice: &[T]) -> &[[T; N]] {
    // SAFETY: Caller must guarantee that `N` is nonzero and exactly divides the slice length
    let new_len = unsafe {
        // assert_unsafe_precondition!(N != 0 && self.len() % N == 0);
        slice.len() / N
    };
    // SAFETY: We cast a slice of `new_len * N` elements into
    // a slice of `new_len` many `N` elements chunks.
    unsafe { core::slice::from_raw_parts(slice.as_ptr().cast(), new_len) }
}

#[cfg(test)]
mod tests {
    use super::slice_as_chunks;
    use ordered_float::NotNan;

    #[test]
    fn test_transmute_float() {
        // Define some val
        let a: f64 = 123.45;
        let b: NotNan<f64> = NotNan::new(a).unwrap();
        let c: f64 = unsafe { std::mem::transmute(b) };

        assert_eq!(a, c,);
    }

    #[test]
    fn test_transmute_array_of_arrays() {
        // Define some vals
        let a: [[f64; 3]; 5] = [
            [1.0, 3.0, 5.0],
            [4.0, 3.0, 3.0],
            [0.0, 6.0, 1.0],
            [1.0, 2.0, 5.0],
            [7.0, 3.0, 3.0],
        ];
        let b: [[NotNan<f64>; 3]; 5] = unsafe {
            [
                [
                    NotNan::new_unchecked(1.0),
                    NotNan::new_unchecked(3.0),
                    NotNan::new_unchecked(5.0),
                ],
                [
                    NotNan::new_unchecked(4.0),
                    NotNan::new_unchecked(3.0),
                    NotNan::new_unchecked(3.0),
                ],
                [
                    NotNan::new_unchecked(0.0),
                    NotNan::new_unchecked(6.0),
                    NotNan::new_unchecked(1.0),
                ],
                [
                    NotNan::new_unchecked(1.0),
                    NotNan::new_unchecked(2.0),
                    NotNan::new_unchecked(5.0),
                ],
                [
                    NotNan::new_unchecked(7.0),
                    NotNan::new_unchecked(3.0),
                    NotNan::new_unchecked(3.0),
                ],
            ]
        };
        let c: [[f64; 3]; 5] = unsafe { std::mem::transmute(b) };

        assert_eq!(a, c,);
    }

    #[test]
    fn test_transmute_slice() {
        // Now slices
        let a: &[f64] = &[1.0, 2.0, 5.0];
        let b: &[NotNan<f64>] = &unsafe {
            [
                NotNan::new_unchecked(1.0),
                NotNan::new_unchecked(2.0),
                NotNan::new_unchecked(5.0),
            ]
        };
        let c: &[f64] = unsafe { std::mem::transmute(b) };
        assert_eq!(a, c);
    }

    #[test]
    fn test_transmute_mut_slice() {
        // Now mut slices
        let a: &mut [f64] = &mut [1.0, 2.0, 5.0];
        let b: &mut [NotNan<f64>] = &mut unsafe {
            [
                NotNan::new_unchecked(1.0),
                NotNan::new_unchecked(2.0),
                NotNan::new_unchecked(5.0),
            ]
        };
        let c: &mut [f64] = unsafe { std::mem::transmute(b) };
        assert_eq!(a, c);
    }

    #[test]
    fn test_transmute_slice_to_array() {
        // Let's try slice -> array ref
        let a: &[f64] = &[1.0, 2.0, 5.0];
        let b: &[f64; 3] = unsafe { slice_to_array(a) };

        a.iter().zip(b).for_each(|(ax, bx)| assert_eq!(ax, bx));
    }

    #[test]
    fn test_transmute_slice_to_slice_of_arrays() {
        // Let's try slice -> array of array ref
        let a: &[f64] = &[1.0, 2.0, 5.0, 6.0];
        let b: &[[f64; 2]] = unsafe { std::mem::transmute(slice_as_chunks::<f64, 2>(a)) };

        for i in 0..4 {
            let (bi, bj) = (i / 2, i % 2);
            assert_eq!(a[i], b[bi][bj]);
        }
    }

    #[test]
    fn test_slice_as_chunks() {
        let a = &[1, 2, 3, 4];
        let b = &[[1, 2], [3, 4]];

        assert_eq!(slice_as_chunks::<u8, 2>(a), b)
    }

    unsafe fn slice_to_array<T, const K: usize>(xs: &[T]) -> &[T; K] {
        ::core::mem::transmute(xs.as_ptr())
    }
}
