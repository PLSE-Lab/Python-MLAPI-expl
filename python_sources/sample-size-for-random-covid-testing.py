#!/usr/bin/env python
# coding: utf-8

# # Sample size for stochastic Covid testing
# 
# The purpose of this notebook is to estimate how many people would need to be sampled to get an accurate picture of the true prevalence of SARS-COV2 infection.

# Let $p$ be the per capita infection rate.  The true value of $p$ is unknown but is lower bounded by the number of confirmed cases per capita. Then the probability $h$ of at least one hit, for $n$ samples is $h = 1 - (1-p)^n$. To find the number of samples needed for a 95% chance of finding a positive example, solve for $n$ to get $n = log(1-h) / log(1-p)$.

# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

h = 0.95 #we want at least this chance of finding a single infected person

def samples_needed(h,p):
    n = ceil(log(1-h) / log(1-p))
    print("%s tests needed for a %s%% chance to find at least one positive, if the true prevalence is %s" %(n, h*100,p))
    return n


# In[ ]:


print("New York State, assuming 10x more cases than reported")
p = 3066.2E-6 #confirmed cases per capita in new york state. source: http://www.91-divoc.com, 29 March 2020
p*=10
__ = samples_needed(h,p)


# In[ ]:


print("nationwide average, assuming 10x reported cases exist")
p = 427E-6 #confirmed cases per capita in US
p *=10 # assume 10x underreporting

__ = samples_needed(h,p)


# In[ ]:


print("Minnesota, assuming 10x more cases than reported")
p = 89.2E-6
p*=10

__ = samples_needed(h,p)


# In[ ]:




