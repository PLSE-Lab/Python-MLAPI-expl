#!/usr/bin/env python
# coding: utf-8

# **Earthquakes - Exploratory data Analysis**
# 
# In this competition we want to predict timing between earthquakes at specific locations. But first let's understand the data better!
# 
# I highly recommend reading [the cited article](https://www.nature.com/articles/ncomms11104) to understand what we are talking here about at all.
# 
# Now, let's load libraries:

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Loading of data itself.

# In[ ]:


train = pd.read_csv("../input/train.csv", dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
print("train shape", train.shape)
pd.options.display.precision = 20
train.head()


# **Missing values**

# In[ ]:


train.isna().sum()


# There are no missing values - very good!
# 
# 
# **Basic descriptive statistics**

# In[ ]:


train["acoustic_data"].describe()


# First the most basic plot "time_to_failure" vs. "acoustic_data".

# In[ ]:


plt.figure(figsize=(12,6))
plt.title("time_to_failure histogram")
ax = plt.plot(train["time_to_failure"], train["acoustic_data"])


# Distribution of "acoustic_data".

# 

# In[ ]:


plt.figure(figsize=(12,6))
plt.title("Acoustic data histogram")
ax = sns.distplot(train["acoustic_data"], label='Acustic data')


# Most of data is gathered in a very narrow range. Therefore let's create a range = mean +/- 2 standard deviations (4 sigma) for better visualisation.

# In[ ]:


upper = train["acoustic_data"].mean()+2*train["acoustic_data"].std()
lower = train["acoustic_data"].mean()-2*train["acoustic_data"].std()

train_subset = train[(train["acoustic_data"]>lower) & (train["acoustic_data"]<upper)]


# In[ ]:


plt.figure(figsize=(12,6))
plt.title("Acoustic data histogram")
ax = sns.distplot(train_subset["acoustic_data"], label='Acustic data', kde=False)


# In[ ]:


plt.figure(figsize=(10,5))
plt.title("time_to_failure histogram")
ax = sns.distplot(train["time_to_failure"], label='time_to_failure')


# UNDER CONSTRUCTION
