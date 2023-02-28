
# 0.3.0
- Removed HashMap build trick to handle identical points. This slightly slows down the build but also slightly speeds up queries.
- Fixed parallel build bug, improving performance dramatically for parallel builds.
- Added several new public methods
- Added intrinsics optimizations (especially for periodic boundary conditions)
- Added support for `f32` (previously only supported `f64`)
This release reworked some internals to increase performance for the target benchmark in the README. With parallel builds, all optimizations, and with single precision, the Rust code now performs the target benchmark (1e5 uniform point build + 1e6 uniform point query) in approximately 28 ms with periodic boundary conditions, and about 26 ms for non-periodic boundary conditions.

# 0.2.0
This release introduces breaking changes while adding functionality. FNNTW now supports
- k-Nearest Neighbor queries
- Periodic Boundary Conditions.