#!/usr/bin/env python
# coding: utf-8

# # Lecture 2: Linear Regressions

# Here, we'll see examples of how to use the scikit-learn linear regression class, as well as the statsmodels OLS function, which is much more similar to R's lm function.

# [http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression_)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns


# Let's make a random dataset where X is uniformly distributed between 0 and 1, and y is a cosine function plus noise:

# In[ ]:


np.random.seed(10)

n_samples = 30

def true_fun(X):
    return np.cos(1.5 * np.pi * X)

X = np.sort(np.random.rand(n_samples))
noise_size = 0.1
y = true_fun(X) + np.random.randn(n_samples) * noise_size


# In[ ]:


np.random.rand(n_samples)


# In[ ]:


X.shape


# In[ ]:


plt.scatter(X, y)


# The scikit-learn linear regression class has the same programming interface we saw with k-NN:

# In[ ]:


linear_regression = LinearRegression()
linear_regression.fit(X.reshape((30, 1)), y)


# We can get the parameters of the fit:

# In[ ]:


print(linear_regression.intercept_)
print(linear_regression.coef_)


# And we can print the predictions as a line:

# In[ ]:


# equally spaced array of 100 values between 0 and 1, like the seq function in R
X_to_pred = np.linspace(0, 1, 100).reshape(100, 1)

preds = linear_regression.predict(X_to_pred)

plt.scatter(X, y)
plt.plot(X_to_pred, preds)
plt.show()


# Let's fit a model of the form $y \sim x + x^2$ to try and capture some of the non-linearity in the underlying cosine function.

# In[ ]:


X**2


# In[ ]:


X2 = np.column_stack((X, X**2))
X2


# In[ ]:


linear_regression.fit(X2, y)


# In[ ]:


print(linear_regression.intercept_)
print(linear_regression.coef_)


# In[ ]:


# equally spaced array of 100 values between 0 and 1, like the seq function in R
X_p = np.linspace(0, 1, 100).reshape(100, 1)
X_to_pred = np.column_stack((X_p, X_p**2))

preds = linear_regression.predict(X_to_pred)

plt.scatter(X, y)
plt.plot(X_p, preds)
plt.show()


# ## Statsmodels

# The `statsmodels` package provides statistical functionality a lot like R's for doing OLS.  It should already be available on your machine if you've setup Anaconda.  Otherwise, you can run ```conda install statsmodels``` at a terminal/prompt.

# [http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/ols.html](http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/ols.html)

# In[ ]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

np.random.seed(9876789)


# ### Using A Formula to Fit to a Pandas Dataframe

# [http://statsmodels.sourceforge.net/0.6.0/examples/notebooks/generated/formulas.html](http://statsmodels.sourceforge.net/0.6.0/examples/notebooks/generated/formulas.html)

# In[ ]:


# We load a datase compiled by A.M. Guerry in the 1830's looking at social factors like crime and literacy
# http://vincentarelbundock.github.io/Rdatasets/doc/HistData/Guerry.html
# In general, statsmodels can download any of the toy datasets provided in R, and provides
# the same documentation from within Python
dta = sm.datasets.get_rdataset("Guerry", "HistData", cache=True)
print(dta.__doc__)


# In[ ]:


original_df = dta.data
original_df.head()


# In[ ]:


# Now, let's select a subset of columns
subsetted_df = original_df[['Lottery', 'Literacy', 'Wealth', 'Region']]
subsetted_df.head(100)


# In[ ]:


df = dta.data[['Lottery', 'Literacy', 'Wealth', 'Region']].dropna()
df.head(100)


# In[ ]:


# Next, let's fit the model by using a formula, just as we can in R, then running .fit()
# We regress the amount of money bet on the lottery on literacy, wealth, region, and
# and interaction between literacy and wealth.
mod = smf.ols(formula='Lottery ~ Literacy + Wealth + Region + Literacy:Wealth', data=df)
res = mod.fit()
print(res.summary())


# In[ ]:


# Next, we add polynomial terms for wealth, i.e., wealth^2 and wealth^3
mod = smf.ols(formula='Lottery ~ Literacy + Wealth + I(Wealth ** 2.0) + I(Wealth ** 3.0) + Region + Literacy:Wealth', data=df)
res = mod.fit()
print(res.summary())


# If it were an integer code instead of a string, we could explicitly make `Region` categorical like this:

# In[ ]:


res = smf.ols(formula='Lottery ~ Literacy + Wealth + C(Region)', data=df).fit()
print(res.params) # Print estimated parameter values


# In[ ]:


print(res.bse) # Print standard errors for the estimated parameters


# In[ ]:


print(res.predict()) # Print fitted values


# In[ ]:




