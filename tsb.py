import numpy as np
from time import time
import pyfnntw


NRAND = 10**5
NQUERY = 10**6
RUNS = 100
LS = 32
KK = [1, 2, 4, 8]

# Generate data 
DATA = np.random.uniform(size=(NRAND, 3))

for lib in ["fnntw"]:
    print()
    print(f"{lib} results (d={NRAND:,}, q={NQUERY:,}, ls={LS}, runs={RUNS})")
    for K in KK:
        query_time = 0
        build_time = 0
        kdtree = None
        # Generate queries
        queries = np.random.uniform(size=(NQUERY, 3))
        for run in range(RUNS):


            # Build tree
            start = time()
            kdtree = pyfnntw.Treef64(DATA, leafsize=LS, par_split_level=3)
            bt = (time() - start)*1000
            build_time += bt

            # Query tree
            start = time()
            if lib == "scipy":
                dist, idx = kdtree.query(queries, k=K, workers=-1)        
            elif lib == "pykdtree":
                dist, idx = kdtree.query(queries, k=K, sqr_dists=False)
            elif lib == "fnntw":
                dist, idx = kdtree.query(queries, K)
            else:
                assert False
            qt = (time() - start)*1000
            query_time += qt
            
            # Print run results
            # print(f"Run {run}: {bt=:.3f} ms; {qt=:.3f} ms")

        avg_build = build_time / RUNS
        avg_query = query_time / RUNS
        print(f"{lib} k={K}: {avg_build=:.3f} ms; {avg_query=:.3f} ms")
    print()

print("periodic results")
for lib in ["fnntw"]:
    print()
    print(f"{lib} results (d={NRAND:,}, q={NQUERY:,}, ls={LS}, runs={RUNS})")
    for K in KK:
        query_time = 0
        build_time = 0
        kdtree = None
        # Generate queries
        queries = np.random.uniform(size=(NQUERY, 3))
        for run in range(RUNS):

            # Build tree
            start = time()
            kdtree = pyfnntw.Treef64(DATA, boxsize = np.array([1.0]*3), leafsize=LS, par_split_level=3)
            bt = (time() - start)*1000
            build_time += bt

            # Query tree
            start = time()
            dist, idx = kdtree.query(queries, K)
            qt = (time() - start)*1000
            query_time += qt
            
            # Print run results
            # print(f"Run {run}: {bt=:.3f} ms; {qt=:.3f} ms")

        avg_build = build_time / RUNS
        avg_query = query_time / RUNS
        print(f"{lib} k={K}: {avg_build=:.3f} ms; {avg_query=:.3f} ms")
    print()

for lib in ["fnntw"]:
    print()
    print(f"{lib} single precision results (d={NRAND:,}, q={NQUERY:,}, ls={LS}, runs={RUNS})")
    DATA = DATA.astype(np.float32)
    for K in KK:
        query_time = 0
        build_time = 0
        kdtree = None
        # Generate queries
        queries = np.random.uniform(size=(NQUERY, 3)).astype(np.float32)
        for run in range(RUNS):


            # Build tree
            start = time()
            kdtree = pyfnntw.Treef32(DATA, leafsize=LS, par_split_level=3)
            bt = (time() - start)*1000
            build_time += bt

            # Query tree
            start = time()
            dist, idx = kdtree.query(queries, K)
            qt = (time() - start)*1000
            query_time += qt
            
            # Print run results
            # print(f"Run {run}: {bt=:.3f} ms; {qt=:.3f} ms")

        avg_build = build_time / RUNS
        avg_query = query_time / RUNS
        print(f"{lib} k={K}: {avg_build=:.3f} ms; {avg_query=:.3f} ms")
    print()