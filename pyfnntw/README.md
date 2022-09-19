# FNNTW: Fastest Nearest Neighbor (in the) West

Fastest Neighbor Nearest Neighbor (in the) West is an in-development kD-tree library that aims to be one of the most, if not the most, performant parallel kNN libraries that exist.


### Basic Usage
```python
import pyfnntw;
import numpy as np;
from scipy.spatial import cKDTree as Tree
from time import time

ND = 10**5
NQ = 10**6
RUNS = 10
TRIALS = 10
WARMUP = 5

# Get some data and queries
data = np.random.uniform(size=(ND, 3))
query = np.random.uniform(size=(NQ, 3))

# Build and query the pynntw tree
tree1 = pyfnntw.Tree(data, 32, 1)
(_, ids1) = tree1.query(query)

# Build and query the scipy tree
tree2 = pyfnntw.Tree(data, 32, 1)
(_, ids2) = tree2.query(query)

if np.all(ids1 == ids2):
    print("Success")
else:
    print("Failure")
```

There are several key components of the building and querying process of the kD-trees that allows for its performance.

### 1. Building The Tree
##### a. Using Quickselect
By using the quickselect in the form of `select_nth_unstable_by` in Rust's `core::slice` instead of an average or even some other median algorithm, we get the `left` and `right` subsets for free in the same O(N) algorithm instead of having to do yet another O(N) comparisons to obtain the `left` and `right` bins. As such, the build is still O(N log(N)), but removes a whole O(N) operation that is found in many other libraries during each splitting.

##### b. Parallel build
Many kD-Tree build implementations are not parallel. Every subtree of a kD-tree is an independent kD-tree. So, after some number of splittings one could throw the builds of the remaining subtrees on different threads. This is done here at the user-specified `par_split_level`, which is the tree depth at which the parallelism begins. See note in **Benchmark Against Other Codes** about recommended values. 

### 2. Querying the tree
Most libraries return either the kNN distance or the distance and the corresponding data point's index. Here we return the distance, the index, and the neighbor. The way in which we obtain the index is new.
##### a. `HashMap<&'[NotNan<f64>; D], u64>`
We attempted to do what other libraries do, which is to carry around the data's index in some struct or tuple, e.g. `&(usize, [NotNan<f64>; D])`, but found poor performance. By using the `ordered_float` crate's `NotNan<T>`, struct, floating points become hashable. Then, one can do all querying operations using just the query and candiate positions, and then do an O(1) lookup at the end to retrieve the closest neighbor's index. This `HashMap` is constructed at build time **in parallel** with the tree itself, adding only the minimal overhead of a thread spawn.

### 3. Unsafe Accesses
Although a `const fn` can be used to calculate the size of a particular tree at compile time, that is not done in this library (to support multiple trees in a single program, as well as trees for datasets made at runtime). As such, the size of the vector containing the nodes belonging to the tree is not known to the compiler at compile time. However, valid indices to children nodes are generated at runtime when building the tree and so the `unsafe` methods `get_unchecked` and `get_unchecked_mut` speed up access times by omitting bounds checks.

# Present State of the Code
At present, the library serves 1NN and kNN queries with separate APIs. **Note that the f64 returned represents the squared euclidean distance to the neighbor, not the euclidean distance**. The library supports periodic boundary conditions via the `with_boxsize` method of `Tree`. Finally, there are plans to add generic parameters to allow for monomorphization over `f32` and `f64` (at present only `f64` is supported).

The accompanying pyfnntw module also supports 1NN and kNN queries (nonperiodic and periodic), but presently the kNN query is slow. More work will be done to make it on par with the rust library's performance. The 1NN query is on par with the rust library. Note that setting k=1 uses the kNN query method; to use the 1NN query just omit the optional `k` parameter, or set it to `None`.


# Benchmark Against Other Codes
This library is intended to be used by the author to calculate summary statistics in cosmology. As such, the parameters of the benchmark chosen are close to those that would be used in analyzing the output of a cosmological simulation. In such an application, often many subsamples or simulation boxes are used. So, the combined build + query time is important since many different trees may be constructed in an analysis. We use 
 - A mock dataset of 100,000 uniform random points in the unit cube.
 - A query set of 1,000,000 uniform random points in the unit cube.

Over 100 realizations of the datasets and query points, the following results are obtained for the average build (serial) and 1NN query times on an AMD EPYC 7502P using 48 threads. The results are sorted by the combined build and query time. 

|  Code | Build (ms)| Query (ms) | Total (ms) |
|---|---|---|---|
| FNNTW | 11 | 28 | 39 |
| pykdtree (python)| 12 | 36 | 48  |
| nabo-rs (rust)| 25 | 30  | 55 |
| Scipy's cKDTree (python) | 31 | 47 | 78 |
| kiddo (rust)| 26 | 84 | 110 |
(These results are for the base Rust code. Calling the code within python adds a bit of overhead. Using the parallel build we measure about 9-10ms for the query, and about 32ms for the query).

With FNNTW's parallel build capability, the build time goes as low as 8.7 ms on the AMD EPYC 7502P (at `split_level = 1`). Since the overhead of the parallelism and atomic operations slows down the build when the number of datapoints is small, both a parallel build and non_parallel build are available via `Tree:new(..)` and `Tree::new_parallel(..)`. The latter takes the aforementioned parameter `par_split_level`, which is the split depth at which the parallelism begins. Although for our applications of O(1e5) points we see the biggest improvement for `par_split_level = 1`, we suspect that the optimal `par_split_level` will increase with the size of the dataset. 


