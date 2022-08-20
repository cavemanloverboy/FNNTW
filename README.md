# FNNTW: Fastest Nearest Neighbor (in the) West

Fastest Neighbor Nearest Neighbor (in the) West is an in-development kDtree library that aims to be one of the most, if not the most, performant parallel kNN libraries that exist.

There are several key components of the building and querying process of the kDtrees that allows for its performance.

### 1. Building The Tree
##### a. Using Quickselect
By using the quickselect in the form of `select_nth_unstable_by` in Rust's `core::slice` instead of an average or even some other median algorithm, we get the `left` and `right` subsets for free in the same O(N) algorithm instead of having to do yet another O(N) comparisons to obtain the `left` and `right` bins. As such, the build is still O(N log(N)), but removes a whole O(N) operation that is found in many other libraries during each splitting.

##### b. Parallel build
Many kD-Tree build implementations are not parallel. Every branch (and subbranch) of the tree is an independent kD-tree. After some number of splittings, one could throw the builds of the remaining branches on different threads. This is done here at the user-specified `par_split_level`, which is the tree depth at which the parallelism begins.

### 2. Querying the tree
Most libraries return either the kNN distance or the distance and the corresponding data point's index. Here we return the distance, the index, and the neighbor. The way in which we obtain the index is new.
##### a. `HashMap<&'[NotNan<f64>; D], u64>`
We attempted to do what other libraries do, which is to flinging around the data's index in some struct or tuple, e.g. `&(usize, [NotNan<f64>; D]`, but found poor performance. By using the `ordered_float` crate's `NotNan<T>`, struct, floating points become hashable. Then, one can do all querying operations using just the query and candiate positions, and then do an O(1) lookup at the end to retrieve the closest neighbor's index. This `HashMap` is constructed at build time **in parallel** with the tree itself, adding only the overhead of the thread spawn.

### 3. Unsafe Accesses (Maybe.. Probably Minor, Not Benchmarked)
I debated about whether I should even mention this one... Because we know the shape of all arrays (i.e. the dimension of the tree) at compile time, the `unsafe` methods `get_unchecked` and `get_unchecked_mut` are used liberally throughout the code. Although the impact of this has not been meausured explicitly, theoretically this means virtually no bounds checks are done. It is possible that LLVM optimizes away lots of the bounds checks anyway. There are also several `unwrap_unchecked` calls to several `Option<T>` that are guaranteed by the algorithm to be nonempty. (Going against best practices, these two use cases were pre-optimizations).


# Present State of the Code
At present, the library only serves 1NN queries. In the very near future, this functionality will be expanded to kNN (k>1) queries. In addition, support for periodic boundary conditions will be added. Finally, there are plans to add generic parameters to allow for monomorphization over `f32` and `f64` (at present only `f64` is supported).


# Benchmark Against Other Codes
This library is intended to be used by the author to calculate summary statistics in cosmology. As such, the parameters of the benchmark chosen are close to those that would be used in analyzing the output of a cosmological simulation. We use 
 - A mock dataset of 100,000 uniform random points in the unit cube.
 - A query set of 1,000,000 uniform random points in the unit cube.

In such an application, often many subsamples or simulation boxes are used. So, the combined build + query time is important since many different trees may be constructed in an analysis.

Over 100 realizations of the datasets and query points, the following results are obtained for the average build (serial) and 1NN query times on an AMD EPYC 7502P using 48 threads. The results are sorted by the combined build and query time.

|  Code | Build (ms)| Query (ms) | Total (ms) |
|---|---|---|---|
| FNNTW| 11 | 28 | 39 |
| pykdtree (python)| 12 | 36 | 48  |
| nabo-rs (rust)| 25 | 30  | 55 |
| Scipy's cKDTree (python) | 31 | 47 | 78 |
| kiddo (rust)| 26 | 84 | 110 |

With FNNTW's parallel build capability, the build time goes as low as 8.7 ms on the AMD EPYC 7502P. Since the overhead of the parallelism and atomic operations slows down the build when the number of datapoints is small, both a parallel build and non_parallel build are available via `Tree:new(..)` and `Tree::new_parallel(..)`. The latter takes the aforementioned parameter `par_split_level`, which is the split depth at which the parallelism begins. For our applications of O(1e5) points, we see the biggest improvement for `par_split_level = 1` (and we recommend this value for datasets of this size or smaller), but we suspect that the optimal `par_split_level` will increase with the size of the dataset.
