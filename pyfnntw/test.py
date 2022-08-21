import pyfnntw;
import numpy as np;
from time import time

ND = 10**5
NQ = 10**6
RUNS = 100

overall_build_time = 0
overall_query_time = 0
for _ in range(RUNS):
    
    # Define data, query
    data = np.random.uniform(size=(ND, 3))
    query = np.random.uniform(size=(NQ, 3))

    # Build tree
    start = time()
    tree = pyfnntw.Tree(data, 32)
    build_time = (time() - start) * 1000
    overall_build_time += build_time

    # query tree
    start = time()
    (r, ids) = tree.query(query)
    query_time = (time() - start) * 1000
    overall_query_time += query_time

    print(f"{build_time=:.2f} ms, {query_time=:.2f} ms")

avg_build = overall_build_time / RUNS
avg_query = overall_query_time / RUNS

print(f"Results: {avg_build=:.2f} ms, {avg_query=:.2f} ms")
