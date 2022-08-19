import numpy as np
from time import time
from pykdtree.kdtree import KDTree
from scipy.spatial import cKDTree as Tree

# USE_SCIPY = True
USE_SCIPY = False

NRAND = 10**5
NQUERY = 10**6
RUNS = 100
LS = 32

# Generate data 
DATA = np.random.uniform(size=(NRAND, 3))

query_time = 0
build_time = 0
kdtree = None
for run in range(RUNS):

    # Generate queries
    queries = np.random.uniform(size=(NQUERY, 3))

    # Build tree
    start = time()
    if USE_SCIPY:
        kdtree = Tree(DATA, leafsize=LS)
    else:
        kdtree = KDTree(DATA, leafsize=LS)
    bt = (time() - start)*1000
    build_time += bt

    # Query tree
    start = time()
    if USE_SCIPY:
        dist, idx = kdtree.query(queries, workers=-1)        
    else:
        dist, idx = kdtree.query(queries, sqr_dists=False)
    qt = (time() - start)*1000
    query_time += qt
    
    # Print run results
    print(f"Run {run}: {bt=:.3f} ms; {qt=:.3f} ms")

avg_build = build_time / RUNS
avg_query = query_time / RUNS
name = "scipy" if USE_SCIPY else "pykdtree"
print(f"{name} results (d={NRAND:,}, q={NQUERY:,}, ls={LS}, runs={RUNS})")
print(f"non-pbc: {avg_build=:.3f} ms; {avg_query=:.3f} ms")



