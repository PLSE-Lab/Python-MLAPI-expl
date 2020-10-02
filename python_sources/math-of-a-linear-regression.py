#!/usr/bin/env python
# coding: utf-8

# # Linear Regression
# 
# Here I will present an implementation of a gradient descent for a linear regression.
# 
# ## Notation
# 
# Italic lowercase $a$ will denote scalars  
# Vectors will be formatted as bold lowercase: $\textbf{a}$  
# Bold uppercase font such as this $\textbf{A}$ will be used for matrices
# 
# The following variables will be used below:  
# $m$ - number of features (columns)  
# $n$ - number of samples (rows)  
# $^{(i)}$ - $i$-th sample, where $1 \leq i \leq n$  
# $y$ - target (true values)  
# $\hat{y}$ - prediction  
# $w_j$ - weight for $j$-th feature  
# $b$ - bias  
# $\textbf{X}$ - features matrix of all samples  
# $\textbf{y}$ - target vector  
# $\hat{\textbf{y}}$ - predicted vector  
# $\textbf{w}$ - weights vector  
# $\textbf{1}$ - all-ones vector of length n  
# $^T$ - [transpose](https://en.wikipedia.org/wiki/Transpose) operation
# 
# ## Unvectorized form
# 
# The general equation of a linear regression is following:
# 
# \begin{align*}
# \hat{y}^{(i)} = x_{1}^{(i)}w_1 + \ldots + x_{m}^{(i)}w_m + b \tag 1
# \end{align*}
# 
# Let's define a cost function, we will use a [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error):
# 
# \begin{align*}
# J(w_1,\ldots,w_m,b) = \frac{1}{n} \sum\limits_{i=1}^{n} ( \hat{y}^{(i)} - y^{(i)} )^{2} = \frac{1}{n} \sum\limits_{i=1}^{n} ( x_{1}^{(i)} w_1 + \ldots + x_{m}^{(i)} w_m + b - y^{(i)} )^{2} \tag 2
# \end{align*}
# 
# Now we need to find a gradient of function $J$. It's a vector of the steapest ascent direction of the function and basically is just partial derivatives for weights and a bias of the previous formula:
# 
# \begin{align*}
# \frac{\partial J}{\partial w_j} = \frac{2}{n} \sum\limits_{i=1}^{n} ( x_{1}^{(i)} w_1 + \ldots + x_{m}^{(i)} w_m + b - y^{(i)} ) \cdot x_{j}^{(i)} = \frac{2}{n} \sum\limits_{i=1}^{n} ( \hat{y}^{(i)} - y^{(i)} ) \cdot x_{j}^{(i)} \tag 3
# \end{align*}
# 
# \begin{align*}
# \frac{\partial J}{\partial b} = \frac{2}{n} \sum\limits_{i=1}^{n} ( x_{1}^{(i)} w_1 + \ldots + x_{m}^{(i)} w_m + b - y^{(i)} ) = \frac{2}{n} \sum\limits_{i=1}^{n} ( \hat{y}^{(i)} - y^{(i)} )\tag 4
# \end{align*}
# 
# Now the final step are rules updation for a gradient descent. We should minimize the cost function, so we need to move in an opposite direction from a gradient and we are subtracting our derivatives:
# 
# \begin{align*}
# w_j \leftarrow w_j - \alpha \frac{\partial J}{\partial w_j} \tag 5
# \end{align*}
# 
# \begin{align*}
# b \leftarrow b - \alpha \frac{\partial J}{\partial b} \tag 6
# \end{align*}
# 
# The symbol $\leftarrow$ means that the left-hand side is updated to take the value on the right-hand side; the constant $\alpha$ is known as a learning rate. The larger it is, the larger a step we take. If we chose the small learning rate and perform this update operations many times then we will find an optimal solution of weights and the bias.
# 
# ## Vectorization
# 
# Now we will rewrite these equations in a vector form. Equation 1 in a vectorized form will become:
# 
# \begin{align*}
# \hat{\textbf{y}} = \textbf{Xw} + b\textbf{1} \tag 7
# \end{align*}
# 
# Cost function (see equation 2):
# \begin{align*}
# J(\textbf{w},b) = \frac{1}{n} {\left\lVert \textbf{Xw} + b \textbf{1} - \textbf{y}\right\rVert}^2 = \frac{1}{n} {\left\lVert \hat{\textbf{y}} - \textbf{y}\right\rVert}^2 \tag 8
# \end{align*}
# 
# Two vertical bars mean a [$L^2$ norm](http://mathworld.wolfram.com/L2-Norm.html) operation which is basically a square root of the sum of squared elements of the vector. If we raise it to the power of two then we will get just the sum of squared elements of the vector. And that is what a sum in equation 2 does.
# 
# Partial derivatives (see equations 3 and 4):
# 
# \begin{align*}
# \frac{\partial J}{\partial \textbf{w}} = \frac{2}{n} \textbf{X}^{T} (\textbf{Xw} + b \textbf{1} - \textbf{y}) = \frac{2}{n} \textbf{X}^{T} (\hat{\textbf{y}} - \textbf{y}) \tag 9
# \end{align*}
# 
# \begin{align*}
# \frac{\partial J}{\partial b} = \frac{2}{n} (\textbf{Xw} + b \textbf{1} - \textbf{y}) = \frac{2}{n} (\hat{\textbf{y}} - \textbf{y}) \tag{10}
# \end{align*}
# 
# Update rules (see equations 5 and 6):
# 
# \begin{align*}
# \textbf{w} \leftarrow \textbf{w} - \alpha \frac{\partial J}{\partial \textbf{w}} \tag{11}
# \end{align*}
# 
# \begin{align*}
# b \leftarrow b - \alpha \frac{\partial J}{\partial b} \tag{12}
# \end{align*}
# 
# **Check dimensions for every equation and rewrite them in a long form to be sure that they are correct representations of unvectorized formulas.**
# 
# ## Example
# 
# For example, we have the following five samples with two features $x_1$ and $x_2$, target $y$ and we want to find the best fitting linear function for this data:
# 
# | $x_1$       | $x_2$        | $y$         |
# |-------------|--------------|-------------|
# | 4           | 7            | 37          |
# | 1           | 8            | 24          |
# | -5          | -6           | -34         |
# | 3           | -1           | 16          |
# | 0           | 9            | 21          |

# You can check that the answer is $w_1 = 5$, $w_2 = 2$ and $b = 3$. If you multiply the first column by $w_1$, the second by $w_2$, sum them together and add $b$, you will get exactly our target column. Let's show how to find this solution using python and vecorized formulas above.
# 
# We will implement functions for formulas 7, 8, 9, and 10.

# In[ ]:


import numpy as np

def predict(X, w, b):
    """Make a prediction according to the equation number 7.
    
    Args:
        X: a features matrix.
        w: weights (a column vector).
        b: a bias.
      
    Returns:
        vector: a prediction with the same dimensions as a target column vector (n by 1).
    """
    
    # .dot() is a matrix multiplication in Numpy.
    # We can ommit all-ones vector because Numpy can add a scalar to a vector directly.
    return X.dot(w) + b

def J(y_hat, y):
    """Calculate a cost of this solution (equation 8).
    
    Args:
        y_hat: a prediction vector.
        y: a target vector.
    
    Returns:
        scalar: a cost of this solution.
    """
    # **2 - raise to the power of two.
    # .mean() - calculate a mean value of vector elements.
    return ((y_hat - y)**2).mean()

def dw(X, y_hat, y):
    """Calculate a partial derivative of J with respect to w (equation 9).
    
    Args:
        X: a features matrix.
        y_hat: a prediction vector.
        y: a target vector.
      
    Returns:
        vector: a partial derivative of J with respect to w.
    """
    # .transpose() - transpose matrix.
    return 2 * X.transpose().dot(y_hat - y) / len(y)

def db(y_hat, y):
    """Calculate a partial derivative of J with respect to b (equation 10).
    
    Args:
        y_hat: a prediction vector.
        y: a target vector.
    
    Returns:
        vector: a partial derivative of J with respect to b.
    """
    return 2 * (y_hat - y).mean()


# Now let's run gradient descent for our example.

# In[ ]:


# A features matrix.
X = np.array([
                 [4, 7],
                 [1, 8],
                 [-5, -6],
                 [3, -1],
                 [0, 9]
             ])

# A target column vector.
y = np.array([
                 [37],
                 [24],
                 [-34], 
                 [16],
                 [21]
             ])

# Initialize weights and bias with zeros.
w = np.zeros((X.shape[1], 1))
b = 0

# How much gradient descent steps we will perform.
num_epochs = 50

# A learning rate.
alpha = 0.01

# Here will be stored J for each epoch.
J_array = []

for epoch in range(num_epochs):
    # Equation 7.
    y_hat = predict(X, w, b)

    # Equation 8.
    J_array.append(J(y_hat, y))
    
    # Equation 11.
    w = w - alpha * dw(X, y_hat, y)
    
    # Equation 12.
    # b converges slower than w, so we increased alpha for it by a factor of 10. It's not mandatory though.
    b = b - alpha * db(y_hat, y) * 10


# Visualize how the cost drops after each epoch.

# In[ ]:


import matplotlib.pyplot as plt

plt.plot(J_array)
plt.xlabel('epoch')
plt.ylabel('J')
plt.show()

# {:.3} - round to three significant figures for f-string.
print(f"w1 = {w[0][0]:.3}")
print(f"w2 = {w[1][0]:.3}")
print(f"b = {b:.3}")
print(f"J = {J_array[-1]:.3}")


# WIth cost $J$ approaching zero the results are almost the same as we stated before: $w_1 = 5$, $w_2 = 2$ and $b = 3$.  
# If you increase a number of epochs you will get even closer to the true values.
