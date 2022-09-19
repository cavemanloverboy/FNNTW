import pyfnntw;
import numpy as np;
from scipy.spatial import cKDTree as Tree
from time import time

ND = 10**5
NQ = 10**6
RUNS = 10
TRIALS = 10
WARMUP = 5
K = 8

# Get some data and queries
data = np.random.uniform(size=(ND, 3))
query = np.random.uniform(size=(NQ, 3))

# Build and query the pynntw tree (nonperiodic)
tree1 = pyfnntw.Tree(data, 32)
(_, ids1) = tree1.query(query)

# Build and query the scipy tree (nonperiodic)
tree2 = Tree(data, 32)
(_, ids2) = tree2.query(query)

if np.all(ids1 == ids2):
    print("Nonperiodic k=1: Success")
else:
    print("Nonperiodic k=1: Failure")

# Query the same trees for k>1
(_, ids1) = tree1.query(query, K)
(_, ids2) = tree2.query(query, K)
if np.all(ids1 == ids2):
    print(f"Nonperiodic k=K: Success")
else:
    print(f"Nonperiodic k=K: Failure")



# Build and query the pynntw tree (periodic)
ptree1 = pyfnntw.Tree(data, 32, np.array([1.0, 1.0, 1.0]))
(_, ids1) = tree1.query(query)

# Build and query the scipy tree (periodic)
ptree2 = Tree(data, 32, boxsize=1)
(_, ids2) = tree2.query(query)

if np.all(ids1 == ids2):
    print("Periodic k=1: Success")
else:
    print("Periodic k=1: Failure")

# Query the same trees for k>1
(_, ids1) = tree1.query(query, K)
(_, ids2) = tree2.query(query, K)
if np.all(ids1 == ids2):
    print(f"Periodic k=K: Success")
else:
    print(f"Periodic k=K: Failure")

