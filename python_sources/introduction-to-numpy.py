#!/usr/bin/env python
# coding: utf-8

# # Introduction to NumPy
# **NumPy**: short for Numerical Python, the fundamental package required for high performance scientific computing and analysis
# 
# **Utilities provided by NumPy**:
# * `ndarray`, a fast and space-efficient multi-dimensional array for vectorized arithmetic operations with sophisticated broadcasting mechanism
# * Standard mathematical operations and matrix operations
# * Linear algebra, random number generators, Fourier transform, etc.
# 
# **Setup**:
# * Install NumPy

# In[ ]:


get_ipython().system('pip install numpy')


# * Import NumPy

# In[ ]:


import numpy as np


# # NumPy ndarray
# ## Introduction to ndarray
# **ndarray**: a N-dimensional array object, which is the key features of NumPy
# * Advantages: 
#     * Fast, flexible container for datasets in Python
#     * Allow us to perform mathematical operations
# 
# 
# **Basic attributes of a ndarray**:
# * `ndarray.shape`: return shape of the array
# * `ndarray.dtype`: return the data type of the data in the array
# 
# **Example code**:
# * Create one array representing a scalar $c = 1$ with type `float32`

# In[ ]:


c = np.array(1, dtype=np.float32)
print(f"c: shape {c.shape} - dtype: {c.dtype}")
print(c)


# * Create one array representing a row vector $\textbf{v} = \begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$ with type `int32`

# In[ ]:


v = np.array([1.0, 2.0, 3.0], dtype=np.int32)
print(f"v: shape {v.shape} - dtype: {v.dtype}")
print(v)


# * Create one array representing a matrix $\textbf{A} = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 3 & 4 & 5 & 6 \\ 6 & 7 & 8 & 9 \end{bmatrix}$ whose type is inferred by NumPy

# In[ ]:


A = np.array([[1, 2, 3, 4], [3, 4, 5, 6], [6, 7, 8, 9]])
print(f"A: shape {A.shape} - dtype: {A.dtype}")
print(A)


# ## Creating ndarray
# **Create pre-specified elements**: we can create the matrix $\begin{bmatrix} 1 & 2 \\ 4 & 5 \end{bmatrix}$ with the code below

# In[ ]:


data = np.array([[1, 2], [4, 5]])
print(data)


# **Create some special arrays**:
# * Create an array with all zeros (e.g. $\begin{bmatrix} 0 & 0 & 0 & 0 & 0 \end{bmatrix}$)[](http://)

# In[ ]:


data = np.zeros(5)
print(data)


# * Create an array with all ones (e.g. $\begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$)

# In[ ]:


data = np.ones((3, 3))
print(data)


# * Create an array with elements ranging from $0$ to $4$ (i.e. $\begin{bmatrix} 0 & 1 & 2 & 3 & 4 \end{bmatrix}$)

# In[ ]:


data = np.arange(5)
print(data)


# **Other options for creating a ndarray**:
# 
# | Function | Description |
# | --- | --- |
# | `array` | Convert input data (list, tuple, etc.) to an ndarray. The input data is copied by default |
# | `asarray` | Convert input to ndarray without copying if the input is an ndarray already |
# | `ones`, `ones_like` | Produce an array of ones given shape and dtype |
# | `zeros`, `zeros_like` | Produce an array of ones given shape and dtype |
# | `empty`, `empty_like` | Create new arrays by allocating new memory with random values |
# | `eye`, `identity` | Create a square $N \times N$ matrix |
# 
# # Data types for ndarrays
# **Data types**:
# * Numerical data type:
#     * Integer: `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`
#         * Explain: 
#             * `int8` is signed 8-bit integer type
#             * `uint8` is unsigned 8-bit integer type
#     * Float: `float16`, `float32`, `float64`, `float128`
#     * Complex numbers: `complex64`, `complex128`, `complex256`
# * Other types:
#     * Boolean: `bool`
#     * Python object type: `object`
#     * String type: `string_`
#     * Unicode: `unicode_`
# 
# **Data type casting**: consider the array `arr` $= \begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$

# In[ ]:


arr = np.array([1, 2, 3])
print(f"Original data type: {arr.dtype}")


# If we want to convert `arr.dtype` to `np.float32`, we have two options:
# * Option 1: use `np.float32(arr)` if we want to cast `arr.dtype` to `np.float32`

# In[ ]:


arr = np.float32(arr)
print(f"Converted data type: {arr.dtype}")


# * Option 2: use `arr.astype(np.float32)` if we want to cast `arr.dtype` to `np.float32`

# In[ ]:


arr = arr.astype(np.float64)
print(f"Converted data type: {arr.dtype}")


# # Indexing and slicing
# ## 1D arrays
# Consider array `arr` $= \begin{bmatrix} 0 & 1 & \cdots & 8 & 9 \end{bmatrix}$

# In[ ]:


arr = np.arange(10)
print(arr)


# ### Indexing
# **Single element indexing**:
# * Get the 2nd element of `arr` (i.e. element $1$)

# In[ ]:


print(arr[1])


# * Get the 3rd last element of `arr` (i.e. the 3rd element from the right)

# In[ ]:


print(arr[-3])


# **Multiple element indexing**:
# * Get the 1st and the 2nd elements of `arr` (i.e. elements $0$ and $1$)

# In[ ]:


print(arr[[0, 1]])


# >**NOTE**: multiple element indexing isn't applied to Python `list`

# ### Slicing
# * Get a slide from the 5th element to the 8th element (i.e. $\begin{bmatrix} 5 & 6 & 7 \end{bmatrix}$)

# In[ ]:


print(arr[5:8])


# * Slicing the first 3 elements (i.e. $\begin{bmatrix} 0 & 1 & 2 \end{bmatrix}$)

# In[ ]:


print(arr[:3])


# * Slicing the elements from the 3rd element till the last element (i.e. $\begin{bmatrix} 3 & 4 & \cdots 8 & 9 \end{bmatrix}$)

# In[ ]:


print(arr[3:])


# * Slicing the whole array (i.e. $\begin{bmatrix} 0 & 1 & \cdots & 8 & 9 \end{bmatrix}$)

# In[ ]:


print(arr[:])


# ## Higher-dimensional indexing
# ### 2D arrays
# Consider `arr` $= \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}$

# In[ ]:


arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


# **Indexing**:
# * Get the 1st row of `arr` (i.e. $\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$)

# In[ ]:


print(f"Get a row: {arr[0]} or {arr[0, :]}")


# * Get the 2nd col of `arr` (i.e. $\begin{bmatrix} 2 \\ 5 \\ 8 \end{bmatrix}$)

# In[ ]:


print(f"Get a col: {arr[:, 1]}")


# * Get element `arr`$_{0, 1}$ (i.e. $2$)

# In[ ]:


print(f"Get an element: {arr[0, 1]} or {arr[0][1]}")


# * Get the submatrix $\begin{bmatrix} 2 & 3 \\ 5  & 6 \end{bmatrix}$ with slicing

# In[ ]:


print(f"Indexing with slices: {arr[:2, 1:]}")


# ### N-d arrays
# Consider a random array with shape $3 \times 3 \times 3$

# In[ ]:


arr = np.empty((3, 3, 3))
print(arr)


# 
# **Indexing**:

# In[ ]:


print(f"Get element indexed 0 in the 2nd dimension:")
print(arr[:, 0, :])
print("---")
print(f"Get element indexed 0 in the first dimension:")
print(arr[0, :, :])
print("or")
print(arr[0, ...])
print("---")
print(f"Get element indexed 0 in the last dimension:")
print(arr[:, :, 0])
print("or")
print(arr[..., 0])
print("---")


# ## Advanced indexing
# **Boolean indexing**: if we have an array of student names $\begin{bmatrix} a & b & a \end{bmatrix}$ and an array of their IDs $\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$

# In[ ]:


ids = np.array([1, 2, 3])
names = np.array(["a", "b", "a"])


# We can get the IDs of students named $a$ by the following code

# In[ ]:


print(ids[names == "a"])


# # Basic operations
# **Operations between arrays and scalars**:
# * Any operations between equal-size arrays applies the operation element-wise

# In[ ]:


arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(f"addition: {arr1 + arr2}")
print("---")
print(f"compare: {arr1 > arr2}")


# * Any operations with scalars propagating the scalar to each element

# In[ ]:


arr = np.array([1, 2, 3])
scalar = 3

print(f"addition: {arr+scalar}")
print("---")
print(f"multiplication: {arr*scalar}")
print("---")
print(f"power: {arr**scalar}")
print("---")
print(f"compare: {arr == scalar}")


# >**NOTE**: Python keywords `and` and `or` don't work with boolean arrays
# 
# **Fast element-wise array functions**: 

# In[ ]:


arr1 = np.arange(5)
arr2 = 5 * np.random.rand(5)
print(f"original arrays:")
print(arr1)
print("and")
print(arr2)
print("---")

print(f"square root: {np.sqrt(arr1)}")
print("---")
print(f"exponential root: {np.exp(arr1)}")
print("---")
print(f"max: {np.maximum(arr1, arr2)}")
print("---")


# * Unary element-wise functions (example):
# 
# | Function | Description |
# | --- | --- |
# | `abs` | Compute the absolute value element-wise |
# | `square` | Compute the square of each element |
# | `exp` | Compute the exponent $e^x$ of each element |
# | `log`, `log10`, `log2` | Compute the logarithms of each element |
# | `sign` | Compute the sign of each element |
# | `floor`, `ceil` | Compute the floor and ceiling of each element |
# | `round` | Round each element |
# | `isnan` | Check if each element is `np.nan` (not a number) or not |
# | `isfinite`, `isinf` | Check if each element is finite or infinite |
# | `cos`, `cosh`, `sin`, `sinh`, `tan`, `tanh` | Regular and hyperbolic trigonometric functions |
# | `arccos`, `arccosh`, `arcsin`, `arcsinh`, `arctan`, `arctanh` | Inverse trigonometric functions |
# | `logical_not` | Logical not element-wise |
# 
# * Binary element-wise functions (example):
# 
# | Function | Description |
# | --- | --- |
# | `add`, `subtract`, `multiply`, `divide`, `mod` | Basic operations element-wise |
# | `floor_divide` | Floor divide (truncate the remainder) elementwise |
# | `power` | Power elementwise | 
# | `maximum`, `minimum` | Maximum and minimum elementwise |
# | `logical_and`, `logical_or`, `logical_xor` | Binary logical functions elementwise |

# # Transposing arrays, swapping axes, and concatenation
# * Reshape array

# In[ ]:


arr = np.arange(10)
print(f"original shape: {arr.shape}")
print(f"original array: {arr}")

arr = arr.reshape((2, 5))
print(f"new shape: {arr.shape}")
print(f"new array: {arr}")


# * Transpose array

# In[ ]:


arr = np.arange(10)
arr = arr.reshape((2, 5))
print(f"original array: {arr}")

print(f"transposed array: {arr.T} or {arr.transpose((1, 0))}")

print(f"swap two axes: {arr.swapaxes(1, 0)}")


# * Concatenate arrays

# In[ ]:


mat1 = np.random.randint(low=0, high=10, size=(3, 3))
mat2 = np.random.randint(low=0, high=10, size=(3, 3))
print(f"matrix 1: {mat1}")
print(f"matrix 2: {mat2}")
print("---")

print(f"concatenated matrix (along axis 0): {np.concatenate([mat1, mat2], axis=0)}")
print(f"concatenated matrix (along axis 1): {np.concatenate([mat1, mat2], axis=1)}")


# # Data processing using arrays
# ## Random number generation
# We can use methods of module `numpy.random`
# 
# | Method | Description |
# | --- | --- |
# | `seed` | Seed the random number generator |
# | `permutation` | Return a random permutation of a sequence |
# | `shuffle` | Randomly permute a sequence inplace |
# | `rand` | Draw samples from a uniform distribution | 
# | `randint` | Draw integers from a given low-to-high range |
# | `randn` | Draw samples from a standard normal distribution |
# | `binomial` | Draw samples from a binomial distribution |
# | `normal` | Draw samples from a normal distribution |
# | `beta` | Draw samples from a beta distribution |
# | `chisquare` | Draw samples from a chisquare distribution |
# | `gamma` | Draw samples from a gamma distribution |
# | `uniform` | Draw samples from a uniform distribution |
# 
# * Example 1: generate random set of points $\{\textbf{x}_i\}_{i=1}^{30}$ where $\textbf{x}_i = (x^{(1)}_i, x^{(2)}_i)$ with $x^{(1)}_i$ and $x^{(2)}_i$ are integers

# In[ ]:


import matplotlib.pyplot as plt

# generate random numbers
X = np.random.randint(low=0, high=5, size=(30, 2))

# visualize generated numbers
plt.plot(X[:, 0], X[:, 1], "ro", alpha=0.5)
plt.show()


# * Example 2: generate random set of points $\{\textbf{x}_i\}_{i=1}^{1000}$ where $\textbf{x}_i = (x^{(1)}_i, x^{(2)}_i)$ from multivariate Gaussian with mean $\begin{bmatrix} 2 \\ 2\end{bmatrix}$ and covariance $\begin{bmatrix} 3 & 0 \\ 0 & 1\end{bmatrix}$

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# generate random numbers
means = [[2, 2]]
cov = [[3, 0], [0, 1]]
N = 1000
X0 = np.random.multivariate_normal(means[0], cov, N)

# visualize generated numbers
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].plot(X0[:, 0], X0[:, 1], "ro", alpha=0.2)
ax[0].set_title("Random 2D points")
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")

ax[1] = sns.kdeplot(X0[:, 0], X0[:, 1], shade=True)
ax[1].set_title("Underlying Gaussian")
ax[1].set_xlabel("X")
ax[1].set_ylabel("Y")

plt.show()


# * Example 3: generate random permutation of integers $0, 1, 2, ..., 9$

# In[ ]:


print(np.random.permutation(10))


# * Example 4: generate random numbers from Standard Gaussian

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# generate random numbers
X = np.random.randn(500, 1)

# visualize generated numbers
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(X[:, 0], [7e-3]*500, "ro", alpha=0.1)
ax[0].set_title("Generated data points")
ax[0].set_xlim([-3, 3])

ax[1] = sns.kdeplot(X[:, 0], shade=True)
ax[1].set_title("Underlying Gaussian")
ax[1].set_xlim([-3, 3])
plt.show()


# * Example 5: generate random numbers from uniform distribution within interval $[0, 10]$

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# generate random numbers
X = np.random.uniform(low=0, high=10, size=(100, 1))

# visualize generated numbers
plt.plot(X[:, 0], [7e-3]*100, "ro", alpha=0.5)
plt.xlim([0, 10])
plt.show()


# ## Mathematical and statistical methods
# 
# | Method | Description |
# | --- | --- |
# | `sum` | Sum of all elements in the array or along an axis. Zero-length arrays have sum $0$ |
# | `mean` | Arithmetic mean. Zero-length arrays have NaN mean |
# | `std`, `var` | Standard deviation and variance |
# | `min`, `max` | Minimum and maximum element in the array |
# | `argmin`, `argmax` | Indices of minimum and maximum elements |
# | `cumsum` | Cumulative sum of elements starting from $0$ |
# | `cumprod` | Cumulative product of elements starting from $1$ |
# 
# ### Mathematical methods
# Consider the underlying array

# In[ ]:


arr = np.random.randint(low=0, high=10, size=(2, 2))
print(arr)


# Here is some mathematical methods provided by NumPy to work with `arr`

# In[ ]:


print(f"sum: {arr.sum()} or {np.sum(arr)}")
print(f"sum along axis 0: {arr.sum(axis=0)} or {np.sum(arr, axis=0)}")
print(f"argmin: {arr.argmin()} or {np.argmin(arr)}")
print(f"cumsum: {arr.cumsum()} or {np.cumsum(arr)}")


# ### Statistical methods
# Consider the underlying array

# In[ ]:


arr = np.random.randn(300)


# First, let's visualize the generated numbers in `arr` and their underlying distribution

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# visualize generated numbers
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(arr, [7e-3]*300, "ro", alpha=0.1)
ax[0].set_title("Generated data points")
ax[0].set_xlim([-3, 3])

ax[1] = sns.kdeplot(arr, shade=True)
ax[1].set_title("Underlying Gaussian")
ax[1].set_xlim([-3, 3])
plt.show()


# We can compute some statistical properties of `arr` using `numpy`

# In[ ]:


print(f"mean: {arr.mean()} or {np.mean(arr)}")
print(f"std: {arr.std()} or {np.std(arr)}")
print(f"var: {arr.var()} or {np.var(arr)}")


# ## Sorting
# * Sort (for 1D array): `arr.sort()`
# * Sort along the 1st axis: `arr.sort(1)`

# In[ ]:


mat1 = np.random.randint(low=0, high=10, size=(3, 3))
print(f"original matrix:")
print(mat1)
print("---")

mat1.sort()
print(f"sorted matrix:")
print(mat1)
print("---")

mat1.sort(axis=0)
print(f"sorted along axis 0")
print(mat1)
print("---")


# ## Unique and other set logic
# 
# | Method | Description |
# | --- | --- |
# | unique(x) | Compute the sorted, unique elements in `x` |
# | `intersect1d(x, y)` | Compute the sorted, common elements in 1D arrays `x` and `y` |
# | `union1d(x, y)` | Compute the sorted, union of elements in 1D arrays `x` and `y` |
# | `in1d(x, y)` | Compute a boolean array indicating whether each element of `x` is in `y` |
# | `setdiff1d(x, y)` | Set difference, elements in `x` and not in `y` |

# In[ ]:


mat1 = np.random.randint(low=0, high=10, size=10)
mat2 = np.random.randint(low=0, high=10, size=8)
print(f"matrix 1: {mat1}")
print(f"matrix 2: {mat2}")
print("---")

print(f"unique elements of mat1: {np.unique(mat1)}")
print("---")
print(f"intersection of mat1 and mat2: {np.intersect1d(mat1, mat2)}")
print("---")
print(f"union of mat1 and mat2: {np.union1d(mat1, mat2)}")
print("---")
print(f"whether each element of mat1 is in mat2 or not: {np.in1d(mat1, mat2)}")
print("---")
print(f"elements in mat1 but not in mat2: {np.setdiff1d(mat1, mat2)}")
print("---")


# ## Linear algebra
# We can use the methods of module `numpy.linalg`
# 
# | Method | Description |
# | --- | --- |
# | `diag` | Return the diagonal elements of a square matrix as a 1D array, or convert a 1D array into a diagonal matrix |
# | `dot` | Matrix multiplication |
# | `trace` | Compute matrix trace |
# | `det` | Compute matrix determinant |
# | `eig` | Compute the eigenvalues and eigenvectors of a square matrix | 
# | `inv` | Compute the inverse of a square matrix |
# | `pinv` | Compute the pseudo-inverse of a square matrix |
# | `qr` | Compute the QR decomposition of a matrix | 
# | `svd` | Compute the SVD of a matrix |
# | `solve` | Solve the linear system $A x = b$ |
# | `lstsq` | Compute the least-squares solution to $y = X b$ |`

# In[ ]:


mat1 = np.random.randint(low=0, high=10, size=(2, 2))
mat2 = np.random.randint(low=0, high=10, size=(2, 2))
print(f"matrix 1:")
print(mat1)
print(f"matrix 2:")
print(mat2)
print("---")

print(f"matrix multiplication")
print(np.dot(mat1, mat2))
print("---")
print(f"matrix trace")
print(np.trace(mat1))
print("---")
print(f"matrix determinant")
print(np.linalg.det(mat1))
print("---")
print("---")
print(f"matrix pseudo-inverse")
print(np.linalg.pinv(mat1))
print("---")


# # CAUTION
# If you don't know something, Google it!
# 
# # References
# * [NumPy documentation](https://numpy.org/doc/)
# * [NumPy broadcasting](https://docs.scipy.org/doc/numpy-1.15.0/user/basics.broadcasting.html) (advanced)
# * Python for Data Analysis (O'reilly)

# In[ ]:




