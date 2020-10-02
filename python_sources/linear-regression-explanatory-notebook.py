#!/usr/bin/env python
# coding: utf-8

# # Understanding Linear Regression
# 
# _Linear regression_ is perhaps one of the most well known and well understood algorithms in statistics and machine learning.
# 
# You do not need to know any statistics or linear algebra to understand linear regression. This is a gentle high-level introduction to the technique to give you enough background to be able to use it effectively on your own problems.
# 
# When you start looking into linear regression, things can get very confusing.
# 
# The reason is because linear regression has been around for so long (more than 200 years). It has been studied from every possible angle and often each angle has a new and different name.
# 
# Linear regression is a linear model, e.g. a model that assumes a linear relationship between the input variables (x) and the single output variable (y). More specifically, that y can be calculated from a linear combination of the input variables (x).
# 
# $y = Wx + b $

# Let's begin by importing all the required fundamental libraries beforehand 
# Here we will be using the _LinearRegression_ function from the _Sklearn_ module just to compare how well our model has been working. 
# 
# Apart from that, just to showcase the plots in a lucid representation, we are going to use the seaborn and the mpl_toolkits for 3d plots.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing linear regression module from sklearn
from sklearn.linear_model import LinearRegression

import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')
from mpl_toolkits.mplot3d import axes3d


# We will just be downloading the dataset of population datapoints.[](http://)

# In[ ]:


import urllib.request
local_filename, headers = urllib.request.urlretrieve("https://raw.githubusercontent.com/emilmont/Artificial-Intelligence-and-Machine-Learning/master/ML/ex1/ex1data1.txt","ex1data1.txt")
html = open(local_filename)
html.close()


# Once we get the dataset, we will be splitting the x and the y axis in the variable $X$ and $y$

# In[ ]:


data = np.loadtxt('ex1data1.txt', delimiter=',')

X = np.c_[np.ones(data.shape[0]),data[:,0]]
y = np.c_[data[:,1]]


# In[ ]:


plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s');


# What do we see? In the plot above, we can see that the points have been clustered on one side and somewhat linearly increases with the change in x. Can we find a line that will approximately fit the dataset? Lets find out.

# We will first begin with the cost function that will compute the cost difference between the actual and the predicted values. The cost function also called as the loss function used in the tutorial is given as $$ J = \frac{1}{2}m\sum{(y - \hat{y})^2} $$

# In[ ]:


def computeCost(X, y, w=[[0],[0]]):
    m = y.size
    J = 0
    h = X.dot(w)
    J = 1/(2*m)*np.sum(np.square(h-y))
    return(J)


# In[ ]:


computeCost(X,y)


# ## Gradient Descent
# Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost).
# Gradient descent is best used when the parameters cannot be calculated analytically (e.g. using linear algebra) and must be searched for by an optimization algorithm.
# 
# In gradient descent, the values of W and b optimize based in the following mathematical formulas
# 
# $$ \frac{\delta}{\delta{w}}J(w) = \Delta_w J $$
# 
# 
# $$ W = W - \alpha \Delta_w J(w)  $$
# 
# 
# $$ \frac{\delta}{\delta{b}}J(w) = \Delta_b J $$
# 
# 
# $$ b = b - \alpha \Delta_b J(w) $$
# 
# 3blue1Brown's video here explains very intutively the actual working of the gradient descent algorithm: 
# 
# https://www.youtube.com/watch?v=IHZwWFHWa-w

# In[ ]:


def gradientDescent(X, y, w=[[0],[0]], alpha=0.01, num_iters=1500):
    m = y.size
    J_history = np.zeros(num_iters)
    
    for iter in np.arange(num_iters):
        h = X.dot(w)
        w = w - alpha*(1/m)*(X.T.dot(h-y))
        J_history[iter] = computeCost(X, y, w)
    return(w, J_history)


# In[ ]:


# theta for minimized cost J
theta , Cost_J = gradientDescent(X, y)
print('theta: ',theta.ravel())

plt.plot(Cost_J)
plt.ylabel('Cost J')
plt.xlabel('Iterations');


# 

# In[ ]:


xx = np.arange(5,23)
# Corresponding with y = Wx + b
yy = ( theta[1]*xx )+ theta[0]

# Plot gradient descent
plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.plot(xx,yy, label='Linear regression (Gradient descent)')

# Compare with Scikit-learn Linear regression 
regr = LinearRegression()
regr.fit(X[:,1].reshape(-1,1), y.ravel())
plt.plot(xx, regr.intercept_+regr.coef_*xx, label='Linear regression (Scikit-learn GLM)')

plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend(loc=4);


# In[ ]:


# Predict profit for a city with population of 35000 and 70000
print(theta.T.dot([1, 3.5])*10000)
print(theta.T.dot([1, 7])*10000)


# In[ ]:


# Create grid coordinates for plotting
B0 = np.linspace(-10, 10, 50)
B1 = np.linspace(-1, 4, 50)
xx, yy = np.meshgrid(B0, B1, indexing='xy')
Z = np.zeros((B0.size,B1.size))

# Calculate Z-values (Cost) based on grid of coefficients
for (i,j),v in np.ndenumerate(Z):
    Z[i,j] = computeCost(X,y, w=[[xx[i,j]], [yy[i,j]]])

fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Left plot
CS = ax1.contour(xx, yy, Z, np.logspace(-2, 3, 20), cmap=plt.cm.jet)
ax1.scatter(theta[0],theta[1], c='r')

# Right plot
ax2.plot_surface(xx, yy, Z, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
ax2.set_zlabel('Cost')
ax2.set_zlim(Z.min(),Z.max())
ax2.view_init(elev=15, azim=230)

# settings common to both plots
for ax in fig.axes:
    ax.set_xlabel(r'$\theta_0$', fontsize=17)
    ax.set_ylabel(r'$\theta_1$', fontsize=17)

