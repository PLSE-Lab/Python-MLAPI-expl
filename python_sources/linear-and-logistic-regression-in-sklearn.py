#!/usr/bin/env python
# coding: utf-8

# # Major classes of machine learning problems
# Assume that we have a training dataset ${\cal{D}}_\text{train} = \{(x_i, y_i)\}_{i=1}^N$ for learning, we also have some new data ${\cal{D}}_\text{test} = \{x_i\}_{i=1}^M$ and we have to predict the corresponding $y_i$ value of each $x_i \in {\cal{D}}_\text{test}$. In summary, we have
# * Training set: ${\cal{D}}_\text{train} = \{(x_i, y_i)\}_{i=1}^N$
# * Test set: ${\cal{D}}_\text{test} = \{(x_i, y_i)\}_{i=1}^M$
# 
# There are three main classes of problems following this form
# 
# 
# * Regression problem: when $\{y_i\}_{i=1}^N$ are continuous
# <image src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/1200px-Linear_regression.svg.png" width="400px">
# 
#     * Example 1: 
#         * $x_i$ is the information about some person's health
#         * $y_i$ is his height 
#     * Example 2: 
#         * $x_i$ is the information about some person's highshcool marks
#         * $y_i$ is the probability that he will be able to get into FTU
#     * Example 3:
#         * $x_i$ is the information about a house (e.g. area, location, etc.)
#         * $y_i$  is the price of the house
# * Classification problem: when $\{y_i\}_{i=1}^N$ are discrete and the number of possible values of them are finite
# <image src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR6wQfnkIma64VyKE8m0HJxbUYj-zyvJpfm_kBvIKEQehpvtZiX&s" width="400px">
# 
#     * Example 1:
#         * $x_i$ is an image
#         * $y_i$ is whether the image contains a car or not
#     * Example 2:
#         * $x_i$ is the information about a car
#         * $y_i$ is the type of the car (e.g. sport car, etc.)
# 
#     >**NOTE**: when the number of possible values of $y_i$ can be infinite (e.g. face recognition), it's not a **classification** problem, it's a **recognition** problem
# 
# * Clustering problem: when $\{y_i\}_{i=1}^N$ aren't given in ${\cal{D}}_\text{train}$
# <image src="https://media.geeksforgeeks.org/wp-content/uploads/k-means-copy.jpg" width="400px">
# 
#     * Example:
#         * $x_i$ is the information about a customer
#         * $y_i$ (not given in the training set but must be predicted) is his class of customer

# # Introduction to sklearn
# **Introduction to `sklearn`**
# 
# `sklearn` (stands for Sci-kit learn) is a well-known open-source Python package which provides various machine learning algorithms and techniques for the three classes of problems above
# * Classification
# * Regression
# * Clustering
# 
# `sklearn` also provides a wide range of utilities for machine learning projects, including Dimensionality reduction (e.g. PCA), Model selection (e.g. grid search, cross valation), and Data preprocessing (e.g. feature extraction).
# 
# **Limitation of `sklearn`**
# 
# Despite of its easy-to-use and famous, `sklearn` still cannot be able to be applied in large-scale machine learning project due to a crucial fact: the algorithms and techniques implemented in `sklearn` are all using CPUs for computation. However, `sklearn` still plays an important role in machine learning projects, especially in phases where runtime is not a key problem.
# 
# **Install `sklearn` for Python**

# In[ ]:


get_ipython().system('pip install sklearn')


# Next, we should dive into three basic algorithms for classification, regression, and clustering problems
# 
# ## Hello world - linear regression with `sklearn`
# We will begin first with linear regression, the simplest and foremost machine learning algorithm.
# 
# First, we need to generate random data. Please refer to [the NumPy lecture](https://drive.google.com/file/d/1gLaygnaxiPE5KssySFjvFLUuy8OwGhKm/view) of COTAI for random number generation.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# generate training data
# generate 2000 training samples from gaussian distribution as a 2000x1 vector
X_train = np.random.randn(2000, 1)
# generate gaussian noise for each samples (noise same shape with input)
noise = np.random.randn(2000, 1)
# create y = 4 + 3X and add noise
Y_train = 4+3*X_train+noise

# generate test data
# generate 10 test samples from gaussian distribution as a 10x1 vector
X_test = np.random.randn(10, 1)
# generate gaussian noise for each samples (noise same shape with input)
noise = np.random.randn(10, 1)
# create y = 4 + 3X and add noise
Y_test = 4+3*X_test+noise


# In[ ]:


# visualize data
# plot training samples as points ("o"), index 0 to get scalar values, transparency 0.2
plt.plot(X_train[:, 0], Y_train[:, 0], "o", alpha=0.2, label="train")
# plot test samples as points ("o")
plt.plot(X_test[:, 0], Y_test[:, 0], "o", label="test")
# set plot title 
plt.title("Data points")
# set axis name
plt.xlabel("X")
plt.ylabel("Y")
# create legend on top right
plt.legend()
# show plot
plt.show()


# Given the generated data (both training and testing), we create a `LinearRegression` object (from `sklearn.linear_model` package) and train it with the training data

# In[ ]:


from sklearn import linear_model

# create LinearRegression model
regressor = linear_model.LinearRegression()

# train the regression model using fit()
regressor.fit(X_train, Y_train)


# In[ ]:


# visualize the trained results
# learned regressor line will have W as .coef_ and b as .intercept_ of the regressor
print(f"Learned parameters- W: {regressor.coef_} - b:{regressor.intercept_}")
# plot training samples as points ("o"), index 0 to get scalar values, transparency 0.2
plt.plot(X_train[:, 0], Y_train[:, 0], "o", alpha=0.2, label="train")
# plot test samples as points ("o")
plt.plot(X_test[:, 0], Y_test[:, 0], "o", label="test")
# plot regression model as a line (X as x axis and projection on regressor line as y axis)
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, label="regression model")
# set plot title
plt.title("Data points")
# set axis name
plt.xlabel("X")
plt.ylabel("Y")
# create legend on top left
plt.legend()
# show plot
plt.show()


# Next, let's use the linear regression model to predict `Y_test` and tell how good our model is

# In[ ]:


# use the function predict() of regressor to predict with X_test
Y_preds = regressor.predict(X_test)
# sklearn LinearRegressor() provide score() function to estimate loss of learned regression model
print(f"Test score: {regressor.score(X_test, Y_test)}")


# In[ ]:


# visualize the trained results
# plot predicted samples as points ("o")
plt.plot(X_test[:, 0], Y_preds[:, 0], "o", label="prediction")
# plot generated test samples as points ("o")
plt.plot(X_test[:, 0], Y_test[:, 0], "o", label="test")
# plot regression model as a line (X as x axis and projection on regressor line as y axis), transparency 0.2
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, label="regression model", alpha=0.2)
# set plot title
plt.title("Data points")
# set axis name
plt.xlabel("X")
plt.ylabel("Y")
# create legend on top left
plt.legend()
# show plot
plt.show()


# ## Logistic regression - a simple linear classification algorithm
# In this section, we examine logistic regression, the simplest classification algorithm for two-class classification problems (i.e. binary classification)
# 
# First, let's generate the training data for each class and the testing data
# 
# We want the samples to be independant on each axis so we set the covariance matrix to $\begin{bmatrix} 1&0 \\ 0&1 \end{bmatrix}$.
# 
# We want the samples to be easily separatable (for illustration) so mean values of x axis and y axis of class 0 and class 1 should be far from each other. So we choose means $\begin{bmatrix} 2&2 \end{bmatrix}$ for class 0 and means $\begin{bmatrix} 5&7 \end{bmatrix}$ for class 1.
# 
# Each class will have 1000 samples.
# 
# Class 0 will have labels = 0 and class 1 will have labels = 1.

# In[ ]:


cov = [[1, 0], [0, 1]]
mean0 = [2, 2]
mean1 = [5, 7]
N = 1000

# generate train set
# class 0
# generate class 0 samples x axis and y axis values from gaussian distribution
# numpy provides multivariate_normal() function to generate multiple values at once
X0_train = np.random.multivariate_normal(mean0, cov, N)
# class 0 have labels 0
Y0_train = np.zeros(N)
# class 1
# generate class 1 samples x axis and y axis values from gaussian distribution
X1_train = np.random.multivariate_normal(mean1, cov, N)
# class 1 have labels 1
Y1_train = np.ones(N)
# concatenate the training data for each class to create training set with both class.
X_train = np.concatenate([X0_train, X1_train], axis=0)
Y_train = np.concatenate([Y0_train, Y1_train], axis=0)


# Create test set with similar statistic as the training set.
# 
# Test samples = 10.

# In[ ]:


# generate test set
M = 10

X0_test = np.random.multivariate_normal(mean0, cov, M)
Y0_test = np.zeros(M)
# class 1
X1_test = np.random.multivariate_normal(mean1, cov, M)
Y1_test = np.ones(M)
# concatenate the training data for each class to create test set with both class.
X_test = np.concatenate([X0_test, X1_test], axis=0)
Y_test = np.concatenate([Y0_test, Y1_test], axis=0)


# In[ ]:


# visualize generated numbers
# plot class 0 samples as red points ("ro"), transparency = 0.2
plt.plot(X0_train[:, 0], X0_train[:, 1], "ro", alpha=0.2, label="train 0")
# plot class 0 samples as blue points ("bo"), transparency = 0.2
plt.plot(X1_train[:, 0], X1_train[:, 1], "bo", alpha=0.2, label="train 1")
# plot test samples as black point ("o", color="black")
plt.plot(X_test[:, 0], X_test[:, 1], "o", color="black", label="test")
# set axis name
plt.xlabel("X")
plt.ylabel("Y")
# create legend on top left
plt.legend()
# show plot
plt.show()


# Given the generated data (both training and testing), we create a `LogisticRegression` object (from `sklearn.linear_model` package) and train it with the training data

# In[ ]:


from sklearn import linear_model

# create LogisticRegression model
classifier = linear_model.LogisticRegression()

# train the regression model with fit() function
classifier.fit(X_train, Y_train)


# We learned logistic classifier: $\begin{bmatrix} w_0&w_1&b \end{bmatrix}$
# 
# We one sample point: $\begin{bmatrix} x&y&1 \end{bmatrix}$
# 
# The classifier boundary will be:
# \begin{equation}
# \begin{bmatrix} w_0&w_1&b \end{bmatrix} \cdotp \begin{bmatrix} x&y&1 \end{bmatrix}^T = 0
# \end{equation}
# 
# In simple form:
# \begin{equation}
# x.w_0 + y.w_1 + b = 0
# \end{equation}
# 
# So we can plot a line with y as a function of x:
# \begin{equation}
# y = \frac{0 - w_0.x - b}{w_1}
# \end{equation}

# In[ ]:


# visualize the trained results
# learned regressor line will have W as .coef_ and b as .intercept_ of the regressor
print(f"Learned parameters- W: {classifier.coef_} - b:{classifier.intercept_}")
# plot class 0 samples as red points ("ro"), transparency = 0.2
plt.plot(X0_train[:, 0], X0_train[:, 1], "ro", alpha=0.2, label="train 0")
# plot class 0 samples as blue points ("bo"), transparency = 0.2
plt.plot(X1_train[:, 0], X1_train[:, 1], "bo", alpha=0.2, label="train 1")
# plot test samples as black point ("o", color="black")
plt.plot(X_test[:, 0], X_test[:, 1], "o", color="black", label="test")
# plot classification boundary line
plt.plot(X_train[:, 0], (0-classifier.intercept_ - classifier.coef_[0, 0]*X_train[:, 0])/classifier.coef_[0, 1], label="classification boundary")
# set axis name
plt.xlabel("X")
plt.ylabel("Y")
# create legend on top left
plt.legend()
# show plot
plt.show()


# Next, let's use the logistic regression model to predict `Y_test` and tell how good our model is

# In[ ]:


# create LogisticRegression model
Y_preds = classifier.predict(X_test)
print(f"Test score: {classifier.score(X_test, Y_test)}")


# In[ ]:


# visualize the trained results
# get test samples from each class
X_test_class0 = X_test[Y_preds == 0]
X_test_class1 = X_test[Y_preds == 1]
# plot class 0 test samples as red points ("ro")
plt.plot(X_test_class0[:, 0], X_test_class0[:, 1], "ro", label="test 0")
# plot class 1 test samples as blue points ("ro")
plt.plot(X_test_class1[:, 0], X_test_class1[:, 1], "bo", label="test 1")
# plot decision boundary
plt.plot(X_train[:, 0], (0-classifier.intercept_ - classifier.coef_[0, 0]*X_train[:, 0])/classifier.coef_[0, 1], label="classification boundary")
# set axis name
plt.xlabel("X")
plt.ylabel("Y")
# create legend on top left
plt.legend()
# show plot
plt.show()


# In[ ]:




