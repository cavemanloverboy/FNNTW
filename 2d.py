from time import perf_counter as time
import pyfnntw
import numpy as np
import scipy


RAND_QUERY = [
    (10**6, 10**5),
    (4*10**6, 4*10**6),
]
RUNS = 1
WARMUP = 0
LS = 32
K = 10

def get_trans_par(data, query, k):
    start = time()
    xtree = scipy.spatial.cKDTree(data, boxsize=1.0)
    bt = (time()-start)*1000

    start = time()
    _, disi = xtree.query(query, k=k)
    qt = (time()-start)*1000

    dis_trans = -np.ones((len(query), k))
    dis_par = -np.ones((len(query), k))
    for ik in range(k):
        ineighs = disi[:, ik]
        delta_pos = data[ineighs] - query
        delta_trans = np.sqrt(delta_pos[:, 0]**2 + delta_pos[:, 1]**2)
        delta_par = abs(delta_pos[:, 2])
        dis_trans[:, ik] = delta_trans          
        dis_par[:, ik] = delta_par

    return dis_trans, dis_par, bt, qt

def get_trans_par_fnntw(data, query, k):
    start = time()
    xtree = pyfnntw.Treef64(data, leafsize=LS, boxsize=np.array([1.0, 1.0, 1.0]))
    bt = (time()-start)*1000

    start = time()
    par, trans = xtree.query(query, k=k, axis=2)
    qt = (time()-start)*1000

    return trans, par, bt, qt

for (rand, query) in RAND_QUERY:
    for lib in ["scipy", "fnntw"]:
        query_time = 0
        build_time = 0
        total_time = 0
        bt = 0; qt = 0
        for run in range(-WARMUP, RUNS):
            DATA = np.random.uniform(size=(rand, 3))
            QUERIES = np.random.uniform(size=(query, 3))
            # Query tree
            if lib == "scipy":
                trans, par, bt, qt = get_trans_par(DATA, QUERIES, K)        
            elif lib == "fnntw":
                trans, par, bt, qt = get_trans_par_fnntw(DATA, QUERIES, K)
            if run >= 0:
                query_time += qt
                build_time += bt
                total_time += qt+bt
        avg_build = build_time / RUNS
        avg_query = query_time / RUNS
        avg_total = total_time / RUNS
        print(f"{lib:8}: build = {avg_build:.3f} ms; query = {avg_query:.3f} ms; total = {avg_total:.3f} ms")

    scipyt,scipyp,_,_= get_trans_par(DATA, QUERIES, K)   
    fnntwt,fnntwp,_,_= get_trans_par_fnntw(DATA, QUERIES, K)   
    print(f"{scipyt.shape=}; {scipyp.shape=}")
    print(f"{fnntwt.shape=}; {fnntwp.shape=}")
    print(f"{scipyt[:2]=};")
    print(f"{fnntwt[:2]=};")
    print(f"{scipyp[:2]=};")
    print(f"{fnntwp[:2]=};")
