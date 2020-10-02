#!/usr/bin/env python
# coding: utf-8

# ## Restaurant Revenue Prediction
# 
# The data for this exercise is taken from the restaurant revenue prediction [competition](https://www.kaggle.com/c/restaurant-revenue-prediction/data). 
# 
# Perform data analysis on this dataset:
# * Look for anything out of the ordinary
# * Look for missing values
# * Look for outliers
# * Check duplicates
# * Propose any preprocessing steps

# In[ ]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import zipfile
with zipfile.ZipFile('/kaggle/input/restaurant-revenue-prediction/train.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/output/')


# In[ ]:


with zipfile.ZipFile('/kaggle/input/restaurant-revenue-prediction/test.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/output/')


# In[ ]:


restaurants = pd.read_csv('/kaggle/output/train.csv')


# In[ ]:


restaurants.shape


# In[ ]:




