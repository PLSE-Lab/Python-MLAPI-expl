#!/usr/bin/env python
# coding: utf-8

# <h1> Accelerated Data Analysis using Pandas-Profiling</h1>
# 
# # Introduction
# 
# We show here how we can accelerate data analysis using [Pandas-Profiling](https://pandas-profiling.github.io/pandas-profiling/docs/).
# 
# ## Dataset
# 
# As an application, we are using House Sales in King Country, USA dataset. We will apply Pandas-Profiling to this dataset.
# 
# ## Method
# 
# After loading the packages and the data, we just apply pandas_profiling.ProfileReport to the dataset. This will generate the profiling report (see below).

# # Prepare for analysis
# 
# ## Load packages

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import pandas_profiling


# ## Load data

# In[ ]:


data_df = pd.read_csv(os.path.join('../input/housesalesprediction/','kc_house_data.csv'))


# # Generate the profile report

# In[ ]:


pandas_profiling.ProfileReport(data_df)


# ## References
# 
# [1] Pandas-Profiling Documentation, https://pandas-profiling.github.io/pandas-profiling/docs/
