import numpy as np

x = [[1.0, 2.0], [3.0, 4.0]]     # a nested list, not yet an array
shape = (4, 3)                   # desired shape: 4 rows, 3 columns
n = 5                            # a length, or the side of a square matrix
seed = 0                         # integer that fixes the random sequence
a = np.zeros((4, 3))             # the array you are operating on
b = np.ones((4, 3))              # a second array, broadcastable with a
v = np.zeros(4)                  # a 1D array, used where a shape like (n,) is shown
i = 0                            # a row index
j = 1                            # a column index
rows = np.array([0, 2, 3])       # one row index per selection
cols = np.array([1, 0, 2])       # the column index paired with each row
cond = a > 0                     # a boolean array, true or false per element
idx = np.array([0, 2, 1])        # integer indices, one per row
k = 3                            # number of columns in the identity matrix
lo, hi = 0.0, 1.0                # smallest and largest value allowed
axis = 0                         # which dimension to reduce; 0 is down the rows
keepdims = True                  # leave the reduced dimension in place, with size 1

np.array(x)                         # turn a list into an array
np.zeros(shape)                     # array of zeros
np.ones(shape)                      # array of ones
np.eye(n)                           # identity matrix, n by n
np.arange(n)                        # 0, 1, ..., n-1

rng = np.random.default_rng(seed)   # reproducible random source
rng.standard_normal(shape)          # normal samples, mean 0, variance 1
rng.random(shape)                   # uniform samples on [0, 1)
rng.integers(low, high, shape)      # random ints, low inclusive, high exclusive

a.shape                             # size of each axis
a.T                                 # swap the two axes
a.reshape(shape)                    # change the shape; -1 fills in one axis
a.ravel()                           # collapse to one axis
a[:, None]                          # turn (n,) into (n, 1)
a[None, :]                          # turn (n,) into (1, n)
np.squeeze(a)                       # drop axes whose size is 1
np.concatenate([a, b], axis=0)      # join along an existing axis
np.stack([a, b], axis=0)            # join along a new axis

a @ b                               # matrix multiply
a * b                               # multiply matching elements
a + b                               # add; a smaller array stretches to fit
a / b                               # divide matching elements
a ** n                              # raise each element to n

np.sum(a, axis=axis, keepdims=keepdims)     # add along one axis; keepdims leaves it as size 1
np.mean(a, axis=axis, keepdims=keepdims)    # average along one axis
np.max(a, axis=axis, keepdims=keepdims)     # largest value along one axis
np.argmax(a, axis=axis)                     # index of the largest value
np.exp(a)                           # e to the power of each element
np.log(a)                           # natural log of each element
np.sqrt(a)                          # square root of each element
np.tanh(a)                          # hyperbolic tangent of each element
np.abs(a)                           # absolute value of each element
np.sign(a)                          # -1, 0, or 1 for each element
np.maximum(a, b)                    # larger of the two values, element by element
np.clip(a, lo, hi)                  # force every element into [lo, hi]

a[i]                                # row i
a[i, j]                             # one element
a[:, j]                             # column j
a[rows, cols]                       # one element per pair of indices
np.where(cond, x, y)                # x where cond is true, otherwise y
np.eye(k)[idx]                      # one-hot rows for integer indices

a.copy()                            # a separate array, not a view
np.allclose(a, b)                   # true if every element is nearly equal