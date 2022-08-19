from scipy.spatial import cKDTree as Tree
import matplotlib.pyplot as plt
import numpy as np

# Load Data
data = np.load("data.npy")
query = np.load("query.npy")
results = np.load("results.npy")
indices = np.load("indices.npy")

# Check shapes are what we expect
DIMS = 3
QUERY = 10**6
DATA = 10**5
assert data.shape == (DATA, DIMS), "incorrect data shape"
assert query.shape == (QUERY, DIMS), "incorrect query shape"
assert results.shape == (QUERY, ), "incorrect results shape"

# Construct and query tree
tree = Tree(data, leafsize=32)
(expected, idxs) = tree.query(query, workers=-1)

# Check results
# TOL = 1e-6
correct1 = np.isclose(expected**2, results)
correct = (indices == idxs)
if np.all(correct):
    print("Success")
else:
    num = np.invert(correct).sum()
    num1 = np.invert(correct1).sum()
    print(f"Failure: {num:,}/{num:,} of {len(query):,}")

# incorrect = []
# for i in range(len(query)):

#     if idxs[i] != indices[i]:
#         print(i, idxs[i], expected[i]**2, "!=", indices[i], results[i])
#         incorrect.append((i, idxs[i], indices[i]))
#         assert results[i] > expected[i]**2

#     # else:
#         # print(idxs[i], expected[i]**2, "==", indices[i], results[i])


# min = np.zeros(data.shape[1])
# max = np.ones(data.shape[1])

# # print(data)
# # print(query)

# split0 = 0.3289118679348304
# split1 = 0.49667444751900025


# plt.figure(figsize=(12,8))
# plt.plot(data[:,0], data[:,1], 'o', label="data")
# plt.plot(query[:,0], query[:,1], 'x', label="query")
# plt.plot([split0]*2, [0,1], 'k')
# plt.plot([split0, 1], [split1]*2, 'k')
# plt.legend()
# for (query_idx, right, wrong) in incorrect:
#     q = query[query_idx]
#     r = data[right]
#     w = data[wrong]
#     plt.plot([q[0], r[0]], [q[1], r[1]], "g")
#     plt.plot([q[0], w[0]], [q[1], w[1]], "r")
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.savefig("kdtree.png")