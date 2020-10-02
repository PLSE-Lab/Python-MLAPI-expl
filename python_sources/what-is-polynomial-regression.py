#!/usr/bin/env python
# coding: utf-8

# # What is Polynomial Regression?
# Not all data is just a simple line. Most of the time it's a bit more complicated than that. That doesn't mean you can't use a linear model to predict a non linear dataset. Instead of only using one column of data we'll use 3 this time. We'll train our model on this data by giving each column a higher power. So column 1 will be to the power of 1, column 2 will be to the power of 2 and column 3 will be to the power of 3.
# 
# ## Creating our dataset
# Let's start by generating our data. This time we'll use the formula:
# 
# $ y = x^3 + 3x^2 + 2x $

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

m = 100
X = 4*np.random.rand(m, 1) - 2.5
y = X**3 + 3*X**2 + 2*X + np.random.randn(m, 1)
non_random_y = X**3 + 3*X**2 + 2*X


# In[ ]:


plt.plot(X, y, 'ro')
plt.plot(X, non_random_y, 'go')
plt.show()


# ## Creating our model
# The data we have created can not be predicted by a straight line. So we'll have to find a way to get 3 different $\thetha$'s to create a model that does work. Let's start by creating 3 $x$ columns instead of one. These will be for $x^1$, $x^2$ and $x^3$. To do this we are going to be using the _sklearn.preprocessing_ package. This package has a _PolynomialFeatures_ preprocessor that can help us with creating new columns.

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly_converter = PolynomialFeatures(degree=3, include_bias=False)

x_poly_features = poly_converter.fit_transform(X)


# In[ ]:


X[0]


# In[ ]:


x_poly_features[0]


# As shown above, it adds 2 more columns to our data. The second column looks like the squared version of $x$ and the third looks like the cubed version of $x$. Let's confirm this.

# In[ ]:


X[0]**2


# In[ ]:


X[0]**3


# Correct! This means that all PolynomialFeatures does is adding columns that are equal to $x^n$ where $n$ is the column number.
# 
# Now let's try creating a model that predicts our data. We are going to use the _LinearRegression_ from Scikit-Learn this time, as we have already built our own model once before in a previous kernel.

# In[ ]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_poly_features, y)
lin_reg.coef_


# In[ ]:


y_pred = lin_reg.predict(x_poly_features)


# In[ ]:


plt.plot(X, y, 'ro')
plt.plot(X, y_pred, 'bo')
plt.show()


# That looks quite accurate! But we can also check it against our y without randomness added.

# In[ ]:


plt.plot(X, non_random_y, 'go')
plt.plot(X, y_pred, 'bo')
plt.show()


# In[ ]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_true=non_random_y, y_pred=y_pred)


# That's a good score! A MSE of 0.05 means that it's near perfect at predicting our training data. 
# 
# ## Conclusion
# Polynomial Regression is not that different from regular Linear Regression, though it is a lot more versatile. Most of the time a linear model won't work on a real world dataset, but it's still a good model to use when analyzing numeric data that has some noise. 
# 
# ### Previous Kernel
# [What is Gradient Descent?](https://www.kaggle.com/veleon/what-is-gradient-descent)
# ### Next Kernel
# [What are Support Vector Machines?](https://www.kaggle.com/veleon/what-are-support-vector-machines)
