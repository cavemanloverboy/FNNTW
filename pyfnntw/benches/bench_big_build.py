import pyfnntw;
import numpy as np;
from time import perf_counter as time

ND = 10**7
RUNS = 4
TRIALS = 10
WARMUP = 1

overall_build_time = 0
print("Warming up")
for trial in range(1-WARMUP,TRIALS+1):
    trial_build_time = 0
    for _ in range(RUNS):
        
        # Define data, query
        data = np.random.uniform(size=(ND, 3))

        # Build tree
        start = time()
        tree = pyfnntw.Tree(data, 32, 2)
        build_time = (time() - start) * 1000
        trial_build_time += build_time

        print(f"{build_time=:.2f} ms")

    if trial > 0:
        overall_build_time += trial_build_time
        trial_avg_build = trial_build_time / RUNS
        print(f"Trial {trial} results: {trial_avg_build=:.2f} ms")

overall_avg_build = overall_build_time / RUNS / TRIALS
print(f"Overall Results: {overall_avg_build=:.2f} ms")
