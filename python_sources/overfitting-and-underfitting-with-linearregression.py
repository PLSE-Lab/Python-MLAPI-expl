#!/usr/bin/env python
# coding: utf-8

# As usual, we will import stuffs we are going to use.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


# ### Firstly, we will need some data to start with.

# In[ ]:


# Generate training samples
x_train = np.random.rand(100,1)
y_train = - x_train + 3 * (x_train ** 2) - 2 * (x_train ** 3) + 2 + np.random.rand(100,1) * 0.1

# Generate some outlier points in the dataset 
x_train_noise = np.random.rand(10,1)
y_train_noise = - x_train_noise + 3 * (x_train_noise ** 2) - 2 * (x_train_noise ** 3) + 2                 + np.random.rand(10,1) * 0.5

# Combine 'normal' points and 'outlier' points to a single training set
x_train = np.concatenate((x_train, x_train_noise), axis=0)
y_train = np.concatenate((y_train, y_train_noise), axis=0)

# Generate test samples
x_test = np.random.rand(20,1)
y_test = - x_test + 3 * (x_test ** 2) - 2 * (x_test ** 3) + 2 + np.random.rand(20,1) * 0.1


# ### Let's see what we have here.

# In[ ]:


# Plot training samples
plt.scatter(x_train,y_train, label='Training samples')
plt.scatter(x_test,y_test, label='Test samples')
plt.xlabel('x', fontsize=18)
plt.ylabel('y', rotation=0, fontsize=18)
plt.legend()


# Okay so we can see we will need a polynomial line which describe:
# \begin{equation} y = \theta_0.x + \theta_1.x^2 + \theta_2.x^3 + \theta_3 \end{equation} 
# 
# Which is still a linear combination of *thetas* but use higher degree of *x* as features. So we can use the **PolynomialFeatures()** function from sklearn those features.

# ### Let's try some simple linear regression model without any high degree polynomial (or we can say degree = 1)

# In[ ]:


# Generate polynomial features
polynomial_features= PolynomialFeatures(degree=1)
x_train_poly = polynomial_features.fit_transform(x_train)[:,1:]
x_test_poly = polynomial_features.fit_transform(x_test)[:,1:]

# Create linear regression model
model = LinearRegression()

# fit model to polynomial data
model.fit(x_train_poly, y_train)

# print fitted model
print('Coef:', model.coef_, 'Intercept:', model.intercept_)


# In[ ]:


print('Train score:', mean_squared_error(model.predict(x_train_poly), y_train))
print('Test score:', mean_squared_error(model.predict(x_test_poly), y_test))


# ### Let's plot the fitting line we have just trained.

# In[ ]:


idx = np.argsort(x_train, axis=0)[:,0]
plt.plot(x_train[idx], model.predict(x_train_poly)[idx], 'r', label='Fitting line')
plt.scatter(x_train,y_train, label='Training samples')
plt.scatter(x_test,y_test, label='Test samples')
plt.legend()


# ### Okay-ish. But I bet the degree 3 polynomial will work better.

# In[ ]:


# Generate polynomial features
polynomial_features= PolynomialFeatures(degree=3)
x_train_poly = polynomial_features.fit_transform(x_train)[:,1:]
x_test_poly = polynomial_features.fit_transform(x_test)[:,1:]

# Create linear regression model
model = LinearRegression()

# fit model to polynomial data
model.fit(x_train_poly, y_train)

# print fitted model
print('Coef:', model.coef_, 'Intercept:', model.intercept_)


# In[ ]:


print('Train score:', mean_squared_error(model.predict(x_train_poly), y_train))
print('Test score:', mean_squared_error(model.predict(x_test_poly), y_test))


# In[ ]:


idx = np.argsort(x_train, axis=0)[:,0]
plt.plot(x_train[idx], model.predict(x_train_poly)[idx], 'r', label='Fitting line')
plt.scatter(x_train,y_train, label='Training samples')
plt.scatter(x_test,y_test, label='Test samples')
plt.legend()


# ### Or it's just higher degree is *always* better? Let's see what we got with degree 30 features.

# In[ ]:


# Generate polynomial features
polynomial_features= PolynomialFeatures(degree=30)
x_train_poly = polynomial_features.fit_transform(x_train)[:,1:]
x_test_poly = polynomial_features.fit_transform(x_test)[:,1:]

# Create linear regression model
model = LinearRegression()

# fit model to polynomial data
model.fit(x_train_poly, y_train)

# print fitted model
print('Coef:', model.coef_, 'Intercept:', model.intercept_)


# In[ ]:


print('Train score:', mean_squared_error(model.predict(x_train_poly), y_train))
print('Test score:', mean_squared_error(model.predict(x_test_poly), y_test))


# ### Okay it's not ;)

# In[ ]:


idx = np.argsort(x_train, axis=0)[:,0]
plt.plot(x_train[idx], model.predict(x_train_poly)[idx], 'r', label='Fitting line')
plt.scatter(x_train,y_train, label='Training samples')
plt.scatter(x_test,y_test, label='Test samples')
plt.legend()


# Err... What the !@#$ fitting line is this?

# In[ ]:




