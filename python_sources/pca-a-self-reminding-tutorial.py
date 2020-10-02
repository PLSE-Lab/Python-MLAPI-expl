#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This document is derived from the following video: https://youtu.be/1cDSlY5Q-Sw
# 
# <a href="http://www.youtube.com/watch?feature=player_embedded&v=1cDSlY5Q-Sw" target="_blank"><img src="http://i3.ytimg.com/vi/1cDSlY5Q-Sw/maxresdefault.jpg" 
# alt="Machine Intelligence - Lecture 3 (PCA, AI and Data)"/></a>
# 
# - **Principal Component Analysis (Main Features Selection)** which components(=features) are important to keep?
# - Significance = variance
# - Intelligence = recognizing the significance
# 
# Starting point is a file with a table:
# 
# |  #  | $X_1$ | $X_2$ | $X_3$ | $X_4$ | ... | $X_n$ |
# | --- | ----- | ----- | ----- | ----- | --- | ----- |
# |  1  |       |       |       |       |     |       |
# |  2  |       |       |       |       |     |       |
# |  3  |       |       |       |       |     |       |
# | ... |       |       |       |       |     |       |
# |  m  |       |       |       |       |     |       |
# 
# You need data of some sort (e.g. csv, excel, etc.). The columns are called as *features* and rows are called as *observations*. If some of the columns are not changing, why should I use that feature?
# 
# How to calculate variance?
# 
# $$
# \sigma^2 = \frac{1}{n - 1} X X^T \\
# n = \mid X \mid
# $$
# 
# This can be computed only if the $X$ has **zero mean**. Thus, you should subtract the mean from $\boldsymbol{X}$.
# 
# $$
# X = \begin{bmatrix}
# 1 \\
# 2 \\
# 3
# \end{bmatrix}, 
# X^T = \begin{bmatrix}
# 1 & 2 & 3
# \end{bmatrix} \\
# \sigma^2 = \frac{1}{3} [1 + 4 + 9]
# $$
# 
# Let's first get a simple dataset from scikit-learn to test the idea. First, we should import the libraries that we want to use throughout the notebook.

# In[ ]:


import numpy as np
import scipy as sp
from sklearn import datasets
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


# Now, let's download the dataset and store the features in the $X$ matrix and the target observations in the $y$ vector.

# In[ ]:


X, y = datasets.load_iris(return_X_y=True)


# If we print the first 5 observations of $X$, you will see the following table.

# In[ ]:


print(X[:5, :])


# Here, each column represents a feature and each row represents an observation. The target is not important for the PCA so we will not print it.
# 
# Let's calculate the mean of each column. The result will be a vector of 4 elements since we have 4 features and subtract it from each observation.

# In[ ]:


EX = np.mean(X, axis=0).reshape(1, -1)
print(EX)


# We will call the mean as *Expected Value* ($E[X]$) from since. Because we may not have the full population data. Thus, calling it as the mean is a bold move. If we increase the number of observations at hand, we expect that the *Expected Value* should approach to the population mean.

# In[ ]:


X_zero_mean = X - EX
print(X_zero_mean[:5, :])


# # Covariance Matrix Calculation
# 
# As you can see, we create a new variable called *X_zero_mean* which is the version of X with the subtracted *Expected Value*. At this point, we can define a covariance matrix $\Sigma$ which includes all the variances between all possible combinations of features.
# 
# $$
# \Sigma = E[(X - E[X]) \cdot (X - E[X])^T] \\
# C = E[X X^T]
# $$
# 
# $$
# \sigma^2 = var(X) = E[(X - E[X])^2]
# $$
# 
# Is the variance and covariance the same? The answer is: Fundametally yes. They are the same things. They try to measure the cange. However, variance is 1-dimensional and covariance is 2-dimensional. In covariance, the question is does the $X_3$ change while $X_2$ is changing? In varaince, the question is does the $X_3$ change?
# 
# Suppose that you have only 3 features and you calculate the covariance matrix as follows.
# 
# $$
# \Sigma = X X^T
# $$
# 
# Then,
# 
# $$
# \Sigma = \begin{bmatrix}
# x_1^2 & x_1x_2 & x_1x_3 \\
# x_2x_1 & x_2^2 & x_2x_3 \\
# x_3x_1 & x_3x_2 & x_3^2
# \end{bmatrix}
# $$
# 
# The resulting matrix is completely symmetric. The diagonal is the variance and everything else is measuring the covariance.
# 
# Now, let's calculate the same operation by hand with the actual data.

# In[ ]:


C = np.dot(X_zero_mean.T, X_zero_mean) / (len(X_zero_mean) - 1)
print(C)


# Numpy also provides a function to calculate the covariance matrix. We can validate our result with this function.

# In[ ]:


C = np.cov(X, rowvar=False)
print(C)


# As you can see, our result and the Numpy's are completely same. One thing to note is, Numpy assumes that the observations are in the columns and features are in the rows. Because of this, we need to pass the keyword argument *rowvar* as *False*.
# 
# > Principal Componenets = Significant things that change.
# 
# If you do not normalize your data by subtracting the *Expected Value*, you cannot trust the first significant component of your data. Because it may shift itself to the average.
# 
# $$
# C = E[X X^T]
# $$
# 
# > Diagonalizing $C$ using a suitable **orthogonal** transformation matrix $A$ by obtaining $N$ **orthogonal** *special vectors* $u_i$ with *special parameters* $\lambda_i$.
# 
# **Orthogonal** means *perpendicular*. If your matrices are multidimensional and perpendicular to each other, then we call them as orthogonal.
# 
# $A^{-1} = A^T$ for orthogonal matrices.
# 
# We want to decrease the redundancy. If two vectors are perpendicular to each they are independant. On the other hand, if the angle between two vectors are less than 90 degrees, then there are some reduntant information between them. Thus, we have to look orthogonal principal components.
# 
# $$
# C u_i = \lambda u_i
# $$
# 
# We can prove that such a vector and a scalar exist.
# 
# $$
# \begin{bmatrix}
# 2 & 1 \\
# 1 & 2
# \end{bmatrix} \cdot
# \begin{bmatrix}
# 1 \\
# 1
# \end{bmatrix} = 
# 3 \cdot \begin{bmatrix}
# 1 \\
# 1
# \end{bmatrix} = 
# \begin{bmatrix}
# 3 \\
# 3
# \end{bmatrix} \\
# C \cdot u_i = \lambda \cdot u_i
# $$
# 
# $u_i$ is called *eigenvector* and $\lambda$ is called *eigenvalue*. Here another example:
# 
# $$
# \begin{bmatrix}
# 2 & 3 \\
# 2 & 1
# \end{bmatrix} \cdot
# \begin{bmatrix}
# 6 \\
# 4
# \end{bmatrix} = 
# 4 \cdot \begin{bmatrix}
# 6 \\
# 4
# \end{bmatrix} = \begin{bmatrix}
# 24 \\
# 16
# \end{bmatrix}
# $$
# 
# Here, $\big[\begin{smallmatrix}
# 6\\
# 4
# \end{smallmatrix}\big]$ is the *eigenvector* $u_i$ and $4$ is the *eigenvalue* $\lambda$.
# 
# ## Linear Transformation
# 
# $$
# u_i = A (X_i - m) \\
# X_i = m + A^T u_i
# $$
# 
# Since $A^{-1} = A^T$ for orthogonal matrices, so $C' = A C A^T$ such that $C'=\big[\begin{smallmatrix}
# \lambda_1 & 0 & 0 \\
# 0 & \lambda_2 & 0 \\
# 0 & 0 & \lambda_n \\
# \end{smallmatrix}\big]$.

# In[ ]:


eig_vals, eig_vecs = np.linalg.eig(C)
C_prime = np.diag(eig_vals)
print("C' = ")
print(C_prime)


# In an orthogonal transformation, the *trace* of a matrix remains the same.
# 
# $$
# trace(C) = trace(C') = \sum_{i=1}^N \lambda_i \\
# = \sum_{i=1}^N \sigma_i^2
# $$

# In[ ]:


print("trace(C) =", np.trace(C))
print("trace(C') =", np.trace(C_prime))


# As you can see, they are almost same. The difference between them is most certainly negligable. The largest *eigenvalue* is the most important and the smallest one is the least important.
# 
# $$
# C'=\begin{bmatrix}
# \lambda_1 & 0 & 0 \\
# 0 & \lambda_2 & 0 \\
# 0 & 0 & \lambda_n \\
# \end{bmatrix}
# $$
# 
# How the diagonal is sorted? Because we just started to calculate from the first components. Naturally, it computed as sorted.
# 
# # Principal Components
# 
# $\lambda_1, \lambda_2, \lambda_3, ..., \lambda_{n-2}, \lambda_{n-1}, \lambda_n$
# 
# Important <------------> Useless
# 
# Pick $N' << N$. $N'$ should be much smaller than $N$.
# 
# This process is called *dimensionality reduction*.
# 
# PCA is:
# - a linear transformation
# - unsupervised
# - uses statistics and calculus
# - a dimensionality reduction algorithm
# - a visualization algorithm (extremely important)
# - intelligent (because it recognizes significance)

# In[ ]:


two_max_component_idx = np.argsort(eig_vals)[-2:]
two_max_components = eig_vecs[:, list(reversed(two_max_component_idx))]
print("Transformation matrix =\n", two_max_components)


# Apply the chosen transformation matrix to the data.

# In[ ]:


reduced_X = np.dot(X_zero_mean, two_max_components)


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 8))
ax.scatter(reduced_X[:, 0], reduced_X[:, 1], c=y, cmap='tab10')
plt.show()


# # Validation

# In[ ]:


pca = PCA(n_components=2)
X_2_comp = pca.fit_transform(X)


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 8))
ax.scatter(X_2_comp[:, 0], X_2_comp[:, 1], c=y, cmap='tab10')
plt.show()

