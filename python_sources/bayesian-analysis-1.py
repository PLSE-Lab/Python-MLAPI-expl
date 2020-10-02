#!/usr/bin/env python
# coding: utf-8

# # Bayesian data analysis

# In[ ]:


import numpy as np
import pandas as pd


# # Prior
# Without any knowledge we assume the rate is uniformly distributed

# In[ ]:


# number of draws from the prior
n_draws = 10000
prior = pd.Series(np.random.uniform(0.0, 1.0, n_draws))
prior.hist()


# # The generative model

# In[ ]:


# This is either 0, 1 (binary) rate so it should follow Bernoulli distribution
def generative_model(parameters):
    return np.random.binomial(16, parameters)


# In[ ]:


generative_model(0.1)


# # Simulate the data
# Simulate the data using the prior and the generative model

# In[ ]:


sim_data = list()
for p in prior:
    sim_data.append(generative_model(p))


# In[ ]:


prior[0]


# # Posterior
# Filter out all the draws that do not match the data

# In[ ]:


observed_data = 6
posterior = prior[list(map(lambda x: x == observed_data, sim_data))]


# In[ ]:


posterior.hist()


# In[ ]:


posterior


# In[ ]:


print("Number of draws left: %d, Posterior median: %.3f, Posterior quantile interval: %.3f-%.3f" % (len(posterior), posterior.median(), posterior.quantile(.025),posterior.quantile(.975)))


# In[ ]:


# How is the result comparing to 20%
sum(posterior>0.2)/len(posterior)


# In[ ]:


# Expected signups for 100 samples
signups = pd.Series([np.random.binomial(n=100, p=p) for p in posterior])


# In[ ]:


signups.hist()


# In[ ]:


print('Sign-up 95%% quantile interval %d-%d' % tuple(signups.quantile([0.025, 0.975]).values))


# In[ ]:




