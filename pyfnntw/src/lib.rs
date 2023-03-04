use fnntw::Tree as FNNTWTree;
use fnntw::{point::Float, utils::FnntwResult};
use ndarray::Array2;
use numpy::*;
use ouroboros::*;
use pyo3::exceptions::PyValueError;
use pyo3::{exceptions, prelude::*};

use numpy::ndarray::{Array1, ArrayView2};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn pyfnntw(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyclass]
    struct Treef32(Box<dyn FNTree<f32> + Send + Sync + 'static>);
    #[pyclass]
    struct Treef64(Box<dyn FNTree<f64> + Send + Sync + 'static>);

    #[pymethods]
    impl Treef32 {
        #[new]
        fn new<'py>(
            data: PyReadonlyArray2<'py, f32>,
            leafsize: usize,
            par_split_level: Option<usize>,
            boxsize: Option<PyReadonlyArray1<f32>>,
        ) -> PyResult<Treef32> {
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
                .expect("2D array should have two dimensions");

            // Check dimensionality is as expected
            match dims[1] {
                2 => {
                    // SAFETY: this data is owned by python and so the reference will remain valid for longer than this scope.
                    // Note that it is still very possible to use-after-free but there's not much we can do about it without making
                    // a copy of the data
                    let data: &[[f32; 2]] = unsafe {
                        std::mem::transmute(slice_as_chunks::<f32, 2>(
                            data.as_array().as_slice().unwrap(),
                        ))
                    };

                    let tree: Tree2<f32> =
                        Tree2TryBuilder {
                            data,
                            tree_builder: |data: &&[[f32; 2]]| -> Result<
                                FNNTWTree<f32, 2>,
                                Box<dyn std::error::Error>,
                            > {
                                let tree = if let Some(psl) = par_split_level {
                                    FNNTWTree::<'_, f32, 2>::new_parallel(data, leafsize, psl)?
                                } else {
                                    FNNTWTree::<'_, f32, 2>::new(data, leafsize)?
                                };
                                if let Some(boxsize) = boxsize {
                                    let boxsize: [f32; 2] = boxsize.as_slice()?.try_into()?;
                                    Ok(tree.with_boxsize(&boxsize)?)
                                } else {
                                    Ok(tree)
                                }
                            },
                        }
                        .try_build()
                        .map_err(|e| PyValueError::new_err(format!("failed to build tree: {e}")))?;

                    Ok(Treef32(Box::new(tree)))
                }
                3 => {
                    // SAFETY: this data is owned by python and so the reference will remain valid for longer than this scope.
                    // Note that it is still very possible to use-after-free but there's not much we can do about it without making
                    // a copy of the data
                    let data: &[[f32; 3]] = unsafe {
                        std::mem::transmute(slice_as_chunks::<f32, 3>(
                            data.as_array().as_slice().unwrap(),
                        ))
                    };

                    let tree: Tree3<f32> =
                        Tree3TryBuilder {
                            data,
                            tree_builder: |data: &&[[f32; 3]]| -> Result<
                                FNNTWTree<f32, 3>,
                                Box<dyn std::error::Error>,
                            > {
                                let tree = if let Some(psl) = par_split_level {
                                    FNNTWTree::<'_, f32, 3>::new_parallel(data, leafsize, psl)?
                                } else {
                                    FNNTWTree::<'_, f32, 3>::new(data, leafsize)?
                                };
                                if let Some(boxsize) = boxsize {
                                    let boxsize: [f32; 3] = boxsize.as_slice()?.try_into()?;
                                    Ok(tree.with_boxsize(&boxsize)?)
                                } else {
                                    Ok(tree)
                                }
                            },
                        }
                        .try_build()
                        .map_err(|e| PyValueError::new_err(format!("failed to build tree: {e}")))?;
                    Ok(Treef32(Box::new(tree)))
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
            query: PyReadonlyArray2<'_, f32>,
            k: Option<usize>,
            axis: Option<usize>,
        ) -> PyResult<(PyObject, PyObject)> {
            if let Some(axis) = axis {
                let k = k.unwrap_or(1);
                let (axis, nonaxis) = slf.0.query_k_axis(query.as_array(), k, axis)?;
                let distances =
                    unsafe { Array2::from_shape_vec_unchecked((query.shape()[0], k), axis) };
                let indices =
                    unsafe { Array2::from_shape_vec_unchecked((query.shape()[0], k), nonaxis) };
                Ok((
                    distances.into_pyarray(slf.py()).into(),
                    indices.into_pyarray(slf.py()).into(),
                ))
            } else {
                if k == Some(1) || k.is_none() {
                    let (distances, indices) = slf.0.query(query.as_array())?;
                    let distances = Array1::from_vec(distances);
                    let indices = Array1::from_vec(indices);
                    Ok((
                        distances.into_pyarray(slf.py()).into(),
                        indices.into_pyarray(slf.py()).into(),
                    ))
                } else {
                    let k = k.unwrap();
                    let (distances, indices) = slf.0.query_k(query.as_array(), k)?;
                    // SAFETY: the shape has been checked aldready
                    let distances = unsafe {
                        Array2::from_shape_vec_unchecked((query.shape()[0], k), distances)
                    };
                    let indices =
                        unsafe { Array2::from_shape_vec_unchecked((query.shape()[0], k), indices) };
                    Ok((
                        distances.into_pyarray(slf.py()).into(),
                        indices.into_pyarray(slf.py()).into(),
                    ))
                }
            }
        }
    }

    #[pymethods]
    impl Treef64 {
        #[new]
        fn new<'py>(
            data: PyReadonlyArray2<'py, f64>,
            leafsize: usize,
            par_split_level: Option<usize>,
            boxsize: Option<PyReadonlyArray1<f64>>,
        ) -> PyResult<Treef64> {
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
                .expect("2D array should have two dimensions");

            // Check dimensionality is as expected
            match dims[1] {
                2 => {
                    // SAFETY: this data is owned by python and so the reference will remain valid for longer than this scope.
                    // Note that it is still very possible to use-after-free but there's not much we can do about it without making
                    // a copy of the data
                    let data: &[[f64; 2]] = unsafe {
                        std::mem::transmute(slice_as_chunks::<f64, 2>(
                            data.as_array().as_slice().unwrap(),
                        ))
                    };

                    let tree: Tree2<f64> =
                        Tree2TryBuilder {
                            data,
                            tree_builder: |data: &&[[f64; 2]]| -> Result<
                                FNNTWTree<f64, 2>,
                                Box<dyn std::error::Error>,
                            > {
                                let tree = if let Some(psl) = par_split_level {
                                    FNNTWTree::<'_, f64, 2>::new_parallel(data, leafsize, psl)
                                        .unwrap()
                                } else {
                                    FNNTWTree::<'_, f64, 2>::new(data, leafsize).unwrap()
                                };
                                if let Some(boxsize) = boxsize {
                                    let boxsize: [f64; 2] = boxsize.as_slice()?.try_into()?;
                                    Ok(tree.with_boxsize(&boxsize)?)
                                } else {
                                    Ok(tree)
                                }
                            },
                        }
                        .try_build()
                        .map_err(|e| PyValueError::new_err(format!("failed to build tree: {e}")))?;

                    Ok(Treef64(Box::new(tree)))
                }
                3 => {
                    // SAFETY: this data is owned by python and so the reference will remain valid for longer than this scope.
                    // Note that it is still very possible to use-after-free but there's not much we can do about it without making
                    // a copy of the data
                    let data: &[[f64; 3]] = unsafe {
                        std::mem::transmute(slice_as_chunks::<f64, 3>(
                            data.as_array().as_slice().unwrap(),
                        ))
                    };

                    let tree: Tree3<f64> =
                        Tree3TryBuilder {
                            data,
                            tree_builder: |data: &&[[f64; 3]]| -> Result<
                                FNNTWTree<f64, 3>,
                                Box<dyn std::error::Error>,
                            > {
                                let tree = if let Some(psl) = par_split_level {
                                    FNNTWTree::<'_, f64, 3>::new_parallel(data, leafsize, psl)
                                        .unwrap()
                                } else {
                                    FNNTWTree::<'_, f64, 3>::new(data, leafsize).unwrap()
                                };
                                if let Some(boxsize) = boxsize {
                                    let boxsize: [f64; 3] = boxsize.as_slice()?.try_into()?;
                                    Ok(tree.with_boxsize(&boxsize)?)
                                } else {
                                    Ok(tree)
                                }
                            },
                        }
                        .try_build()
                        .map_err(|e| PyValueError::new_err(format!("failed to build tree: {e}")))?;

                    Ok(Treef64(Box::new(tree)))
                }
                _ => {
                    return Err(PyErr::new::<exceptions::PyTypeError, _>(
                        "Only 2D, and 3D are supported at the moment.",
                    ))
                }
            }
        }

        fn query<'py>(
            slf: PyRef<'_, Self>,
            query: PyReadonlyArray2<'_, f64>,
            k: Option<usize>,
            axis: Option<usize>,
        ) -> PyResult<(PyObject, PyObject)> {
            // if k == Some(1) || k.is_none() {
            //     let (distances, indices) = slf.0.query(query.as_array())?;
            //     // let distances = Array1::from_vec(distances);
            //     // let indices = Array1::from_vec(indices);
            //     // Ok((
            //     //     distances.into_pyarray(slf.py()).into(),
            //     //     indices.into_pyarray(slf.py()).into(),
            //     // ))
            //     let distances =
            //         unsafe { Array2::from_shape_vec_unchecked((query.shape()[0], 1), distances) };
            //     let indices =
            //         unsafe { Array2::from_shape_vec_unchecked((query.shape()[0], 1), indices) };
            //     Ok((
            //         distances.into_pyarray(py).into(),
            //         indices.into_pyarray(py).into(),
            //     ))
            // } else {
            //     let k = k.unwrap();
            //     let (distances, indices) = slf.0.query_k(query.as_array(), k)?;
            //     // SAFETY: the shape has been checked aldready
            //     let distances =
            //         unsafe { Array2::from_shape_vec_unchecked((query.shape()[0], k), distances) };
            //     let indices =
            //         unsafe { Array2::from_shape_vec_unchecked((query.shape()[0], k), indices) };
            //     Ok((
            //         distances.into_pyarray(py).into(),
            //         indices.into_pyarray(py).into(),
            //     ))
            // }

            if let Some(axis) = axis {
                let k = k.unwrap_or(1);
                let (axis, nonaxis) = slf.0.query_k_axis(query.as_array(), k, axis)?;
                let distances =
                    unsafe { Array2::from_shape_vec_unchecked((query.shape()[0], k), axis) };
                let indices =
                    unsafe { Array2::from_shape_vec_unchecked((query.shape()[0], k), nonaxis) };
                Ok((
                    distances.into_pyarray(slf.py()).into(),
                    indices.into_pyarray(slf.py()).into(),
                ))
            } else {
                if k == Some(1) || k.is_none() {
                    let (distances, indices) = slf.0.query(query.as_array())?;
                    let distances = Array1::from_vec(distances);
                    let indices = Array1::from_vec(indices);
                    Ok((
                        distances.into_pyarray(slf.py()).into(),
                        indices.into_pyarray(slf.py()).into(),
                    ))
                } else {
                    let k = k.unwrap();
                    let (distances, indices) = slf.0.query_k(query.as_array(), k)?;
                    // SAFETY: the shape has been checked aldready
                    let distances = unsafe {
                        Array2::from_shape_vec_unchecked((query.shape()[0], k), distances)
                    };
                    let indices =
                        unsafe { Array2::from_shape_vec_unchecked((query.shape()[0], k), indices) };
                    Ok((
                        distances.into_pyarray(slf.py()).into(),
                        indices.into_pyarray(slf.py()).into(),
                    ))
                }
            }
        }
    }

    m.add_class::<Treef32>()?;
    m.add_class::<Treef64>()?;
    Ok(())
}

// #[pyclass]
#[self_referencing]
struct Tree2<'a, T: Float> {
    // data: Array1<[NotNan<f64>; 2]>,
    data: &'a [[T; 2]],
    #[borrows(data)]
    #[covariant]
    pub tree: FNNTWTree<'this, T, 2>,
}

// #[pyclass]
#[self_referencing]
struct Tree3<'a, T: Float> {
    data: &'a [[T; 3]],
    #[borrows(data)]
    #[covariant]
    pub tree: FNNTWTree<'this, T, 3>,
}

// impl<'a, T: Float> FNTree<T> for Tree2<'a, T> {
//     fn query(&self, query: ArrayView2<T>) -> PyResult<(Vec<T>, Vec<u64>)> {
//         // Check dimensions of data
//         let dims: [usize; 2] = query
//             .shape()
//             .try_into()
//             .expect("2D array should definitely have 2 dims");

//         // Return error if not 2D
//         if dims[1] != 2 {
//             return Err(PyErr::new::<exceptions::PyTypeError, _>(
//                 "A 3D tree can only be queried with 3D data",
//             ));
//         }

//         let (distances, indices): (Vec<T>, Vec<u64>) = query
//             .axis_iter(Axis(0))
//             .par_bridge()
//             .map(|q| {
//                 let q: &[T; 2] = q.as_slice().unwrap().try_into().unwrap();
//                 let (distance, index, _) = self
//                     .borrow_tree()
//                     .query_nearest(q)
//                     .expect("you likely have a nan");
//                 (distance, index)
//             })
//             .unzip();

//         Ok((distances, indices))
//     }
// }

// impl<'a> FNTree for Tree3<'a> {
//     fn query(&self, query: ArrayView2<f64>) -> PyResult<(Vec<f64>, Vec<u64>)> {
//         // Check dimensions of data
//         let dims: [usize; 2] = query
//             .shape()
//             .try_into()
//             .expect("2D array should definitely have 2 dims");

//         // Return error if not 3D
//         if dims[1] != 3 {
//             return Err(PyErr::new::<exceptions::PyTypeError, _>(
//                 "A 3D tree can only be queried with 3D data",
//             ));
//         }

//         let query: &[[f64; 3]] = slice_as_chunks::<f64, 3>(query.as_slice().unwrap());

//         let tree = self.borrow_tree();
//         let query_size = query.len();
//         let mut distances = Vec::with_capacity(query_size);
//         let mut indices = Vec::with_capacity(query_size);
//         query
//             .into_par_iter()
//             .map_with(tree, |t, q| {
//                 let (d, i, _) = t.query_nearest(q).expect("you likely have a nan");
//                 (d, i)
//             })
//             .unzip_into_vecs(&mut distances, &mut indices);

//         Ok((distances, indices))
//     }
// }

trait FNTree<T: Float> {
    fn query(&self, query: ArrayView2<T>) -> PyResult<(Vec<T>, Vec<u64>)>;
    fn query_k(&self, query: ArrayView2<T>, k: usize) -> PyResult<(Vec<T>, Vec<u64>)>;
    fn query_k_axis(
        &self,
        query: ArrayView2<T>,
        k: usize,
        axis: usize,
    ) -> PyResult<(Vec<T>, Vec<T>)>;
}

macro_rules! tree_impl {
    ($float:ty, $dim:literal) => {
        concat_idents::concat_idents!(tree = Tree, $dim {
            impl<'a> FNTree<$float> for tree<'a, $float> {
                fn query(&self, query: ArrayView2<$float>) -> PyResult<(Vec<$float>, Vec<u64>)> {
                    // Check dimensions of data
                    let dims: [usize; 2] = query
                        .shape()
                        .try_into()
                        .expect("2D array should definitely have 2 dims");

                    // Return error if not 3D
                    if dims[1] != $dim {
                        return Err(PyErr::new::<exceptions::PyTypeError, _>(
                            "Your data is not the right dimension",
                        ));
                    }

                    let query: &[[$float; $dim]] = slice_as_chunks::<$float, $dim>(query.as_slice().unwrap());

                    let kdtree = self.borrow_tree();
                    let query_size = query.len();
                    let mut distances = Vec::with_capacity(query_size);
                    let mut indices = Vec::with_capacity(query_size);
                    query
                        .into_par_iter()
                        .map_with(kdtree, |t, q| {
                            t.query_nearest(q).expect("you likely have a nan")
                        })
                        .unzip_into_vecs(&mut distances, &mut indices);

                    Ok((distances, indices))
                }

                fn query_k(
                    &self,
                    query: ArrayView2<$float>,
                    k: usize,
                ) -> PyResult<(Vec<$float>, Vec<u64>)> {

                    // Check dimensions of data
                    let dims: [usize; 2] = query
                        .shape()
                        .try_into()
                        .expect("2D array should definitely have 2 dims");

                    // Return error if not 2D
                    if dims[1] != $dim {
                        return Err(PyErr::new::<exceptions::PyTypeError, _>("A 3D tree can only be queried with 3D data"))
                    }

                    // Transform slice of floats into slice of arrays
                    let query: &[[$float; $dim]] = slice_as_chunks::<$float, $dim>(query
                        .as_slice()
                        .unwrap()
                    );

                    {
                    // let kdtree = self.borrow_tree();
                    // let (distances, indices): (Vec<$float>, Vec<u64>) = query
                    //     .into_par_iter()
                    //     .flat_map(|q| {
                    //         let result = kdtree.query_nearest_k(q, k)
                    //             .expect("error occurred during query");
                    //         result
                    //     }
                    //     ).unzip();

                    // Ok((distances, indices))
                    }

                    // current best
                    {
                        let mut distances = Vec::with_capacity(query.len() * k);
                        let mut indices = Vec::with_capacity(query.len() * k);
                        let dist_ptr_usize = distances.as_mut_ptr() as usize;
                        let idx_ptr_usize = indices.as_mut_ptr() as usize;
                        let kdtree = self.borrow_tree();
                        query.into_par_iter().enumerate().try_for_each_with(
                            (dist_ptr_usize, idx_ptr_usize),
                            |(d, i), (j, q)| -> FnntwResult<(), $float> {
                                let result = kdtree
                                    .query_nearest_k(q, k)?;

                                unsafe {
                                    let d = *d as *mut $float;
                                    let i = *i as *mut u64;
                                    d.add(k * j).copy_from_nonoverlapping(result.0.as_ptr(), k);
                                    i.add(k * j).copy_from_nonoverlapping(result.1.as_ptr(), k);
                                }
                                Ok(())
                            },
                        ).map_err(|e| {
                            PyValueError::new_err(format!("query failed {e}"))
                        })?;
                        unsafe {
                            distances.set_len(query.len() * k);
                            indices.set_len(query.len() * k);
                        }
                        Ok((distances, indices))
                    }

                    // {
                    //     let kdtree = self.borrow_tree();
                    //     Ok(
                    //         kdtree.query_nearest_k_parallel(&query, k)
                    //         .map_err(|e|
                    //             PyValueError::new_err(format!("query error: {e}"))
                    //         )?
                    //     )
                    // }

                }

                fn query_k_axis(
                    &self,
                    query: ArrayView2<$float>,
                    k: usize,
                    axis: usize
                ) -> PyResult<(Vec<$float>, Vec<$float>)> {

                    // Check dimensions of data
                    let dims: [usize; 2] = query
                        .shape()
                        .try_into()
                        .expect("2D array should definitely have 2 dims");

                    // Return error if not 2D
                    if dims[1] != $dim {
                        return Err(PyErr::new::<exceptions::PyTypeError, _>("A 3D tree can only be queried with 3D data"))
                    }

                    // Transform slice of floats into slice of arrays
                    let query: &[[$float; $dim]] = slice_as_chunks::<$float, $dim>(query
                        .as_slice()
                        .unwrap()
                    );

                    self.borrow_tree().query_nearest_k_parallel_axis(query, k, axis).map_err(|e| {
                        PyValueError::new_err(format!("query failed {e}"))
                    })
                }
            }
        });
    }
}

tree_impl!(f32, 3);
tree_impl!(f64, 3);
tree_impl!(f32, 2);
tree_impl!(f64, 2);

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
unsafe fn as_chunks_unchecked<T, const N: usize>(slice: &[T]) -> &[[T; N]] {
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
    use fnntw::NotNan;

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
