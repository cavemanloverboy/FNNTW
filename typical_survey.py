import numpy as np
from time import perf_counter as time
import pyfnntw
from pykdtree.kdtree import KDTree as pykdTree
from scipy.spatial import cKDTree as scipyTree


NRAND = 4*10**6
NQUERY = 4*10**6
WARMUP = 20
RUNS = 100
LS = 32
KK = [1, 10]

print(f"(d={NRAND:,}, q={NQUERY:,}, ls={LS}, runs={RUNS}")

# Generate data 
DATA = np.random.uniform(size=(NRAND, 3))

times = []
for _ in range((len(KK) + 1)):
    times.append([])
libs = ["scipy", "pykdtree", "fnntw"]
for lib in libs:
    print()
    print(f"{lib} results")
 
    build_time = 0
    kdtree = None
    # Generate queries
    for run in range(-WARMUP, RUNS):
        # Build tree
        start = time()
        if lib == "scipy":
            kdtree = scipyTree(DATA, leafsize=LS)#, boxsize = 1.0)
        elif lib == "pykdtree":
            kdtree = pykdTree(DATA, leafsize=LS)
        elif lib == "fnntw":
            kdtree = pyfnntw.Treef64(DATA, leafsize=LS, par_split_level=4)
        else:
            assert False
        bt = (time() - start)*1000
        if run >= 0:
            build_time += bt
    avg_build = build_time / RUNS
    times[0].append(avg_build)
    print(f"{lib}: {avg_build=:.3f} ms")

    for (i, K) in enumerate(KK):
        query_time = 0
        qt = 0
        queries = np.random.uniform(size=(NQUERY, 3))
        for run in range(-WARMUP, RUNS):
            # Query tree
            if lib == "scipy":
                start = time()
                dist, idx = kdtree.query(queries, k=K, workers=-1)        
                qt = (time() - start)*1000
            elif lib == "pykdtree":
                start = time()
                dist, idx = kdtree.query(queries, k=K, sqr_dists=False)
                qt = (time() - start)*1000
            elif lib == "fnntw":
                start = time()
                dist, idx = kdtree.query(queries, K)
                qt = (time() - start)*1000
            else:
                assert False
            if run >= 0:
                query_time += qt

        avg_query = query_time / RUNS
        times[i+1].append(avg_query)
        print(f"{lib} k={K}: {avg_query=:.3f} ms")
    print()
print('\x1b[6;30;42m' + 'build & nonpbc winners:' + '\x1b[0m')
print(f"build winner: {libs[np.argmin(times[0])]} @ {np.min(times[0]):.3f} ms")
for (i, K) in enumerate(KK):
    print(f"{K=} query winner: {libs[np.argmin(times[1+i])]}: @ {np.min(times[i+1]):.3f} ms")

print()
print("periodic results")
times = []
for _ in range(len(KK)):
    times.append([])
libs = ["scipy", "fnntw"]
for lib in libs:
    print()
    print(f"{lib} results")

    # Build tree (no need to benchmark)
    kdtree = None
    if lib == "scipy":
        kdtree = scipyTree(DATA, boxsize = 1.0, leafsize=LS)
    else:
        kdtree = pyfnntw.Treef64(DATA, boxsize = np.array([1.0]*3), leafsize=LS, par_split_level=4)

    for (i, K) in enumerate(KK):
        query_time = 0
        qt = 0
        queries = np.random.uniform(size=(NQUERY, 3))
        for run in range(-WARMUP, RUNS):
            # Query tree
            if lib == "scipy":
                start = time()
                dist, idx = kdtree.query(queries, k=K, workers=-1)        
                qt = (time() - start)*1000
            else:
                start = time()
                dist, idx = kdtree.query(queries, K)
                qt = (time() - start)*1000
            if run >= 0:
                query_time += qt

        avg_query = query_time / RUNS
        times[i].append(avg_query)
        print(f"{lib} k={K}: {avg_query=:.3f} ms")
    print()
print('\x1b[6;30;42m' + 'pbc winners:' + '\x1b[0m')
for (i, K) in enumerate(KK):
    print(f"{K=} query winner: {libs[np.argmin(times[i])]} @ {np.min(times[i]):.3f} ms", )


times = []
for _ in range(len(KK)):
    times.append([])
libs = ["scipy", "pykdtree", "fnntw"]
DATA = DATA.astype(np.float32)
for lib in libs:
    print()
    print(f"{lib} single precision results")

    build_time = 0
    kdtree = None
    # Generate queries
    queries = np.random.uniform(size=(NQUERY, 3)).astype(np.float32)
    # Build tree (no need to benchmark)
    if lib == "scipy":
        kdtree = scipyTree(DATA, leafsize=LS)#, boxsize = 1.0)
    elif lib == "pykdtree":
        kdtree = pykdTree(DATA, leafsize=LS)
    else:
        kdtree = pyfnntw.Treef32(DATA, leafsize=LS, par_split_level=4)
        
    for (i, K) in enumerate(KK):
        query_time = 0
        qt = 0
        queries = np.random.uniform(size=(NQUERY, 3)).astype(np.float32)
        for run in range(-WARMUP, RUNS):
            # Query tree
            if lib == "scipy":
                start = time()
                dist, idx = kdtree.query(queries, k=K, workers=-1)        
                qt = (time() - start)*1000
            elif lib == "pykdtree":
                start = time()
                dist, idx = kdtree.query(queries, k=K, sqr_dists=False)
                qt = (time() - start)*1000
            else:
                start = time()
                dist, idx = kdtree.query(queries, K)
                qt = (time() - start)*1000
            if run >= 0:
                query_time += qt
            
        avg_query = query_time / RUNS
        times[i].append(avg_query)
        print(f"{lib} k={K}: {avg_query=:.3f} ms")
    print()
print('\x1b[6;30;42m' + 'single precision winners:' + '\x1b[0m')
for (i, K) in enumerate(KK):
    print(f"{K=} query winner: {libs[np.argmin(times[i])]}  @ {np.min(times[i]):.3f} ms", )
