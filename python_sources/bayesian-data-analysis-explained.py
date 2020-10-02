#!/usr/bin/env python
# coding: utf-8

# # Bayesian data analysis explained

# In[ ]:


import numpy as np
import pandas as pd


# # Problem definition
# Assume that we are experimenting a marketing strategy. And would like to predict whether customers will register for our product after we show the product to her. So we went out and did a sample of a few customers (say, 16 of them) and got back the registration rate (say 6 of them registered). Then with such registration rate, how can we infer the registration probability?

# In[ ]:


n_sample = 16
n_registered = 6


# # Prior
# Without any knowledge we assume that our estimating parameter (e.g., $\rho$ for a Bernoulli distribution, for success rate) is distributed from 0.0 to 1.0 uniformly. Therefore, we randomly draw 10,000 values for $\mu$ from the uniform distribution.

# In[ ]:



n_draws = 10000
# samples randomly for the prior
prior = pd.Series(np.random.uniform(0.0, 1.0, n_draws))
# draw the distributions of the prior
prior.hist()


# # Generative model
# The generative model is used to generate outcomes based on a selected prior. We are analysing either if a customer register for our product after an advertisement or not. Therefore, our generative mode folows Bernoulli distribution.

# In[ ]:


# This is either 0, 1 (binary) rate so it should follow Bernoulli distribution
# In this case parameter is the one of the randomly generated success rate based on prior in the previous step
def generative_model(parameters):
    return np.random.binomial(n_sample, parameters)


# In[ ]:


# Test
print(generative_model(0.1))
print(generative_model(0.9))


# # Simulate data based on the prior and the generative model
# We simulate the data using the prior and the generative model. Specifically, for every prior value (for $\rho$) we generate the number of registrations that might generate based on the selected prior and the generative model.

# In[ ]:


sim_data = list()
for p in prior:
    sim_data.append(generative_model(p))


# # Posterior
# Filter out all the draws that do not match the data. I.e., we only keep the samples that out of 16 customers, 6 got registered (or the generative model generates 6). Then observe the distribution of $\rho$ given this fact (data). The posterior will give the $\rho$ that gives the outcomes exactly as our outcome (facts). So we can get back and check for our prior (filter it out) and get the posterior.

# In[ ]:


posterior = prior[[x == n_registered for x in sim_data]]


# In[ ]:


posterior


# In[ ]:


# show the distribution of the posterior
posterior.hist()


# In[ ]:


print(f"Number of draws left: {len(posterior)}")


# In[ ]:


print(f"Posterior median: {'%.3f'%posterior.median()}")


# In[ ]:


print(f"Posterior interval: {'%.3f'%posterior.quantile(0.25)}, {'%.3f'%posterior.quantile(0.75)}")


# In[ ]:


# how is the result comparing to 20%, meaning the percentages of posteriors that we can have the registration rate as 20%
sum(posterior >0.2)/len(posterior)


#  # Model future event
#  Can we give th expected registrations for 100 samples

# In[ ]:


registrations = pd.Series([np.random.binomial(n=100, p=p) for p in posterior])


# In[ ]:


registrations.hist()


# In[ ]:


print('Registration 95%% quantile interval %d-%d'%tuple(registrations.quantile([0.025, 0.975]).values))


# In[ ]:




