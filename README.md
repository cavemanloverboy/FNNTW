# FNNTW: Fastest Nearest Neighbor (in the) West

Fastest Neighbor Nearest Neighbor (in the) West is an in-development kD-tree library that aims to be one of the most, if not the most, performant parallel kNN libraries that exist.

There are several key components of the building and querying process of the kD-trees that allows for its performance.

### 1. Building The Tree
Many kD-Tree build implementations are not parallel despite being very easy to parallelize. Here we discuss parallelization strategies for individual subtrees and across subtrees.

##### a. Using Quickselect
By using the quickselect in the form of `select_nth_unstable_by` in Rust's `core::slice` instead of an average or even some other median algorithm, we get the `left` and `right` subsets for free in the same O(N) algorithm instead of having to do yet another O(N) comparisons to obtain the `left` and `right` bins. As such, the build is still O(N log(N)), but removes a whole O(N) operation that is found in many other libraries during each splitting. For trees with >=100,000 nodes, a homemade parallel approximate median finder is used to find the split point for the subtrees. This speeds up the building of large trees tremendously, as ≈95% of a sequential tree build is spent finding medians.

##### b. Parallel build
Every subtree of a kD-tree is an independent kD-tree, so at each splitting one could build each subtree in parallel up to some minimum subtree size. This is done here with the user-specified `par_split_level`, which is the tree depth at which the parallelism begins. See note in **Benchmark Against Other Codes** about recommended values. 


### 2. Unsafe Accesses
Because we know the shape of all arrays (i.e. the dimension of the tree) at compile time, and we know the tree size and topology post-build at run time, the `unsafe` methods `get_unchecked` and `get_unchecked_mut` are used liberally throughout the code. This means virtually no bounds checks are done.


### 3. Allocators
We tested many allocators, including the default allocator, `jemalloc`, `tcmalloc`, `snmalloc`, `mimalloc`, `rpmalloc`, and found the best performance with `tcmalloc`. 

# Benchmark Against Other Codes
This library is intended to be used by the author to calculate summary statistics in cosmology. As such, the parameters of the benchmark chosen are close to those that would be used in analyzing the output of a cosmological simulation. In such an application, often many subsamples or simulation boxes are used. So, the combined build + query time is important since many different trees may be constructed in an analysis. We use 
 - A mock dataset of 100,000 uniform random points in the unit cube.
 - A query set of 1,000,000 uniform random points in the unit cube.

Over 100 realizations of the datasets and query points, the following results are obtained for the average build (serial) and 1NN query times on an AMD EPYC 7502P using 48 threads. The results are sorted by the combined build and query time.

|  Code | Build (ms)| Query (ms) | Total (ms) |
|---|---|---|---|
| FNNTW | 12 | 22 | 34 |
| pykdtree (python)| 12 | 35 | 47  |
| nabo-rs (rust)| 25 | 30  | 55 |
| Scipy's cKDTree (python) | 31 | 38 | 69 |
| kiddo (rust)| 26 | 84 | 110 |

With FNNTW's parallel build capability, the build time can go as low as 5 ms on the AMD EPYC 7502P (at `split_level = 2`) (and under 5 ms at single precision). Since the overhead of the parallelism and atomic operations slows down the build when the number of datapoints is small, both a parallel build and non_parallel build are available via `Tree:new(..)` and `Tree::new_parallel(..)`. The latter takes the aforementioned parameter `par_split_level`, which is the split depth after which the parallelism stops. Although for our applications of O(1e5) points we see the biggest improvement for `par_split_level = 2`, we expect that the optimal `par_split_level` will increase with the size of the dataset. For tree sizes of O(1e8) points, for example, we see peak performance at `par_split_level = 4` (16 threads).
