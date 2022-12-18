import pyfnntw;
import numpy as np;
from time import perf_counter as time

ND = 10**5
NQ = 10**6
RUNS_PER_TRIAL = 10
TRIALS = 10
WARMUP = 5
D = 3
K = None

overall_build_time = 0
overall_query_time = 0
print("Warming up")
for trial in range(1-WARMUP,TRIALS+1):
    trial_build_time = 0
    trial_query_time = 0
    for _ in range(RUNS_PER_TRIAL):
        
        # Define data, query
        data = np.random.uniform(size=(ND, D))
        query = np.random.uniform(size=(NQ, D))

        # Build tree
        start = time()
        tree = pyfnntw.Tree(data, 32, par_split_level=1)#, boxsize=np.array([1.0, 1.0, 1.0]))
        build_time = (time() - start) * 1000
        trial_build_time += build_time

        # query tree
        start = time()
        (r, ids) = tree.query(query, k=K)
        query_time = (time() - start) * 1000
        trial_query_time += query_time

        #print(f"{build_time=:.2f} ms, {query_time=:.2f} ms")

    if trial > 0:
        overall_build_time += trial_build_time
        overall_query_time += trial_query_time
        trial_avg_build = trial_build_time / RUNS_PER_TRIAL
        trial_avg_query = trial_query_time / RUNS_PER_TRIAL
        print(f"Trial {trial} results: {trial_avg_build=:.2f} ms, {trial_avg_query=:.2f} ms")

overall_avg_build = overall_build_time / RUNS_PER_TRIAL / TRIALS
overall_avg_query = overall_query_time / RUNS_PER_TRIAL / TRIALS
print(f"Overall Results: {overall_avg_build=:.2f} ms, {overall_avg_query=:.2f} ms")
