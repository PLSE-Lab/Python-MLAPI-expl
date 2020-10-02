#!/usr/bin/env python
# coding: utf-8

# ### This is an attempt to create a model, using synthetic data, to show how the flow of waste through a pipe can be monitored to predict when it will clog.  We will only use 2 features:
# 
# ### Our X-axis will be the percentage of the amount of waste that can go through the pipe.  Each pipe will start at 100%, let's imagine that this particular size pipe has a maximum of 50 gallons per minute that can flow through it.  As we more forward in time, let's say 200 days, the pipe may only be at 25 gallons per minute, or 50%.  As a precaution, we'd like to focus on pipes that go under 20% of maximum allowable flow so these can be cleaned before they clog and cause an SSO.
# 
# ### Our Y-axis, and y-variable, will be the number of days.  We will predict the number of days that will occur before a particular pipe is expected to dip below 20% of its maximum flow.

# ### Let's first create a few dataframes.  Each will be different sizes to gauge how well the models can predict.

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import proof_of_concept_helpers
from proof_of_concept_helpers import create_pipe_data
from proof_of_concept_helpers import poly_regression

# Exploring
import scipy.stats as stats

# Modeling
import statsmodels.api as sm

from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error


# In[ ]:


pipe_1 = create_pipe_data(104)


# In[ ]:


pipe_1.head()


# In[ ]:


pipe_1.tail()


# In[ ]:


plt.scatter(pipe_1.days, pipe_1.percent_flow)


# In[ ]:


poly_regression(pipe_1, .8)


# In[ ]:


pipe_2 = create_pipe_data(76)


# In[ ]:


plt.scatter(pipe_2.days, pipe_2.percent_flow)


# In[ ]:


poly_regression(pipe_2, .8)


# In[ ]:


pipe_3 = create_pipe_data(142)


# In[ ]:


plt.scatter(pipe_3.days, pipe_3.percent_flow)


# In[ ]:


poly_regression(pipe_3, .8)


# In[ ]:


pipe_4 = create_pipe_data(211)


# In[ ]:


plt.scatter(pipe_4.days, pipe_4.percent_flow)


# In[ ]:


poly_regression(pipe_4, .8)


# In[ ]:


pipe_5 = create_pipe_data(21)


# In[ ]:


plt.scatter(pipe_5.days, pipe_5.percent_flow)


# In[ ]:


poly_regression(pipe_5, .8)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




