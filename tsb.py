import numpy as np
from time import perf_counter as time
import pyfnntw
import os
os.environ.setdefault("TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD", f"{100*2**30}")


NRAND = 10**5
NQUERY = 10**7
RUNS = 10
WARMUP = 10
LS = 32
KK = [1, 32]

# Generate data 
DATA = np.random.uniform(size=(NRAND, 3))
QUERIES = np.random.uniform(size=(NQUERY, 3))

for lib in ["fnntw"]:
    print()
    print(f"{lib} results (d={NRAND:,}, q={NQUERY:,}, ls={LS}, runs={RUNS})")

    build_time = 0
    kdtree = None
    # Generate queries
    for run in range(-WARMUP, RUNS):
        # Build tree
        start = time()
        kdtree = pyfnntw.Treef64(DATA, leafsize=LS, par_split_level=3)
        bt = (time() - start)*1000
        if run > 0:
            build_time += bt
    avg_build = build_time / RUNS
    print(f"{lib}: {avg_build=:.3f} ms")

    for K in KK:
        query_time = 0
        qt = 0
        for run in range(-WARMUP, RUNS):

            # Query tree
            if lib == "scipy":
                start = time()
                dist, idx = kdtree.query(QUERIES, k=K, workers=-1)        
                qt = (time() - start)*1000
            elif lib == "pykdtree":
                start = time()
                dist, idx = kdtree.query(QUERIES, k=K, sqr_dists=False)
                qt = (time() - start)*1000
            elif lib == "fnntw":
                start = time()
                dist, idx = kdtree.query(QUERIES, K)
                qt = (time() - start)*1000
            else:
                assert False
            if run > 0:
                query_time += qt
            
            # Print run results
            # print(f"Run {run}: {bt=:.3f} ms; {qt=:.3f} ms")

        avg_query = query_time / RUNS
        print(f"{lib} k={K}: {avg_query=:.3f} ms")
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
        QUERIES = np.random.uniform(size=(NQUERY, 3))
        kdtree = pyfnntw.Treef64(DATA, boxsize = np.array([1.0]*3), leafsize=LS, par_split_level=3)

        for run in range(RUNS):
            # Query tree
            start = time()
            dist, idx = kdtree.query(QUERIES, K)
            qt = (time() - start)*1000
            query_time += qt
            
            # Print run results
            # print(f"Run {run}: {bt=:.3f} ms; {qt=:.3f} ms")

        avg_query = query_time / RUNS
        print(f"{lib} k={K}: {avg_query=:.3f} ms")
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
        QUERIES = np.random.uniform(size=(NQUERY, 3)).astype(np.float32)
        for run in range(RUNS):


            # Build tree
            start = time()
            kdtree = pyfnntw.Treef32(DATA, leafsize=LS, par_split_level=3)
            bt = (time() - start)*1000
            build_time += bt

            # Query tree
            start = time()
            dist, idx = kdtree.query(QUERIES, K)
            qt = (time() - start)*1000
            query_time += qt
            
            # Print run results
            # print(f"Run {run}: {bt=:.3f} ms; {qt=:.3f} ms")

        avg_build = build_time / RUNS
        avg_query = query_time / RUNS
        print(f"{lib} k={K}: {avg_build=:.3f} ms; {avg_query=:.3f} ms")
    print()