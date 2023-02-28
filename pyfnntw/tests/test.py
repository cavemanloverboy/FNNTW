import pyfnntw;
import numpy as np;
from scipy.spatial import cKDTree as Tree
from time import time

ND = 10**3
NQ = 10**3
RUNS = 10
TRIALS = 10
WARMUP = 5

# Get some data and queries
data = np.random.uniform(size=(ND, 3))
query = np.random.uniform(size=(NQ, 3))

# Build and query the pynntw tree
tree1 = pyfnntw.Treef64(data, 32, 1)
(_, ids1) = tree1.query(query)

# Build and query the scipy tree
tree2 = Tree(data, 32)
(_, ids2) = tree2.query(query)

if np.all(ids1 == ids2):
    print("Success")
else:
    print("Failure")
