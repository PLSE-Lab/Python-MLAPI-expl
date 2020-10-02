#!/usr/bin/env python
# coding: utf-8

# # Spike-and-Slab Model:  An attempt to imitate the winning stategy from Don't Overfit I
# 
# Bayesian tools have come a long way since Tim Salimans wrote his own Gibbs sampler for the first competition.  These days, it's extremely rare for people to write their own samplers.  Instead, higher level modeling languages like PyMC3 and Stan are used to specify the model with a back-end "inference engine" doing the sampling.  
# 
# However, learning how to specify models properly and using the high-level tools to fit them still requires overcoming a learning curve and thinking a bit differently than the traditional data science workflow, so here's a notebook to help get started.  
# 
# I actually prefer Stan, but the Spike-and-Slab model has discrete parameters that are not possible to fit.  
# 
# EDIT:  I've changed the prior on the coefficients to StudentT(3,0,1).  We don't want to over-regularize the included variables.  
# 
# EDIT (again):  A normal prior works even better.  For .86 score, use p=.05 for hyperprior on `xi` and N(0,.75) as a prior for `beta`.

# # Package Imports
# PyMC3 uses the now semi-orphaned theano for the backend.  PyMC4 will use tf-probability but that's not happening anytime soon.  

# In[ ]:


import pymc3 as pm
import numpy as np
import pandas as pd
import theano.tensor as tt
from scipy.special import expit
invlogit = lambda x: 1/(1 + tt.exp(-x))


# # Data Import
# You'll generally want to work with numpy arrays rather than pandas.  Even if your data is two-dimensional, don't make it a matrix.  It needs to be array. 

# In[ ]:


df = pd.read_csv('../input/train.csv')
y = np.asarray(df.target)
X = np.array(df.ix[:, 2:302])
df2 = pd.read_csv('../input/test.csv')
df2.head()
X2 = np.array(df2.ix[:, 1:301])


# # Model Specification
# This is the fun part.  You're basically designing how your data gets generated.  Learning the shape and properties of various probability distributions is an essential background step to specifying models, but it's also deeply gratifying.  
# 
# Here, we basically have an ordinary logistic regression model with a twist: every variable in the model also is multiplied by a binary indicator `xi` which acts to include or exclude that variable from the model.  
# 

# In[ ]:


def get_model(y, X):
    model = pm.Model()
    with model:
        xi = pm.Bernoulli('xi', .05, shape=X.shape[1]) #inclusion probability for each variable
        alpha = pm.Normal('alpha', mu = 0, sd = 5) # Intercept
        beta = pm.Normal('beta', mu = 0, sd = .75 , shape=X.shape[1]) #Prior for the non-zero coefficients
        p = pm.math.dot(X, xi * beta) #Deterministic function to map the stochastics to the output
        y_obs = pm.Bernoulli('y_obs', invlogit(p + alpha),  observed=y)  #Data likelihood
    return model


# In[ ]:


model1 = get_model(y, X)


# # Sampling
# This part is hard if you do it yourself.  There's a lot of room for mistakes, so this is generally not something you want to open the hood on if you can avoid it.  I won't go into sampling diagnostics here, but the Stan documentation has tons of good information.  
# 
# I recommend using more chains in your own work, but I used just one here as I was having some trouble with the kernel. 

# In[ ]:


with model1:
    trace = pm.sample(2000, random_seed = 4816, cores = 1, progressbar = True, chains = 1)


# # Results
# There's a bit of numpy bookkeeping to learn for working with posterior samples.  The `np.apply_along_axis` function comes in handy.

# In[ ]:


results = pd.DataFrame({'var': np.arange(300), 
                        'inclusion_probability':np.apply_along_axis(np.mean, 0, trace['xi']),
                       'beta':np.apply_along_axis(np.mean, 0, trace['beta']),
                       'beta_given_inclusion': np.apply_along_axis(np.sum, 0, trace['xi']*trace['beta'])
                            /np.apply_along_axis(np.sum, 0, trace['xi'])
                       })


# Now we can view how often the model included each variable and what the coefficients are.

# In[ ]:


results.sort_values('inclusion_probability', ascending = False).head(20)


# # Generate Predictions
# Now we'll use the parameters we estimated to predict on the new data.  First it's good to practice on a single draw from the posterior distribution to make sure our shapes and calculations are all right.  

# In[ ]:


#Scoring test.  Score new data from a single posterior sample
test_beta = trace['beta'][0]
test_inc = trace['xi'][0]
test_score = expit(trace['alpha'][0] + np.dot(X2, test_inc * test_beta))  
test_score


# Now generalize the code to generate a prediction for each data point from test on each draw of the posterior (the prediction is the average over the draws)

# In[ ]:


estimate = trace['beta'] * trace['xi'] 
y_hat = np.apply_along_axis(np.mean, 1, expit(trace['alpha'] + np.dot(X2, np.transpose(estimate) )) )


# Before we submit-- it's a good idea to do some sanity checks:
# * Does the posterior predictive mean roughly equal the data mean?
# * Are the number of included variables close to the `p` hyperprior set on `xi`?

# In[ ]:


#Sanity checks
np.mean(y_hat), np.sum(results.inclusion_probability/300)


# Generate the submission file and we're done

# In[ ]:


submission  = pd.DataFrame({'id':df2.id, 'target':y_hat})
submission.to_csv('submission.csv', index = False)

