#!/usr/bin/env python
# coding: utf-8

# **Bayesian Linear Regression in PYMC3 Tutorial**
# 
# This is a very simple tutorial on using PYMC3 for Bayesian linear regression, with an estimation of the posterior probability distributions.  It was adopted from the PYMC3 getting started documentation [https://docs.pymc.io/notebooks/getting_started.html].

# In[ ]:


import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')


# Let's create some data for our regression.  Our true values are:
# * $\alpha = 1$
# * $\sigma = 1$
# * $\beta = [1, 2.5]$
# 
# Our outcome variable is:
# $$ Y = \alpha + \beta_1 X_1 + \beta_2 X_2.$$

# In[ ]:


# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma


# Here is a quick plot of our data.

# In[ ]:


fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,4))
axes[0].scatter(X1, Y)
axes[1].scatter(X2, Y)
axes[0].set_ylabel('Y'); axes[0].set_xlabel('X1'); axes[1].set_xlabel('X2');


# Now we build our model using PYMC3.

# In[ ]:


basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)


# Now we take 5000 random MCMC samples.  The defualt PYMC3 sampler is the Hamiltonian MC No U-Turn Sampler (NUTS), which is almost always a good choice.

# In[ ]:


with basic_model:
    # draw 5000 posterior samples
    trace = pm.sample(5000)


# The traceplot is the standard good way to view the posterior probability distributions along with theMCMC samples.

# In[ ]:


pm.traceplot(trace);


# There is a built-in summary function as well.

# In[ ]:


pm.summary(trace).round(2)

