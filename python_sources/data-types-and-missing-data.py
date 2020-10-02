#!/usr/bin/env python
# coding: utf-8

# **[Pandas Micro-Course Home Page](https://www.kaggle.com/learn/pandas)**
# 
# ---
# 

# ## Intro
# 
# This is your time to test your understanding of data types and missing value handling.
# 
# # Relevant Resources
# - [Data Types and Missing Data Reference](https://www.kaggle.com/residentmario/data-types-and-missing-data-reference)
# 
# # Set Up
# 
# Run the following cell to load your data and some utility functions.

# In[1]:


import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.data_types_and_missing_data import *
print("Setup complete.")


# # Exercises

# ## 1. 
# What is the data type of the `points` column in the dataset?

# In[3]:


# Your code here
dtype = reviews.points.dtype
print(dtype)
q1.check()


# In[5]:


# q1.hint()
# q1.solution()


# ## 2. 
# Create a `Series` from entries in the `points` column, but convert the entries to strings. Hint: strings are `str` in native Python.

# In[6]:


point_strings = pd.Series(data=reviews.points,dtype=str,)

q2.check()


# In[10]:


# q2.hint()
q2.solution()


# ## 3.
# Sometimes the price column is null. How many reviews in the dataset are missing a price?

# In[11]:


n_missing_prices = reviews.price.isnull().sum()
print(n_missing_prices)
q3.check()


# In[ ]:


#q3.hint()
#q3.solution()


# ## 4.
# What are the most common wine-producing regions? Create a `Series` counting the number of times each value occurs in the `region_1` field. This field is often missing data, so replace missing values with `Unknown`. Sort in descending order.  Your output should look something like this:
# 
# ```
# Unknown                    21247
# Napa Valley                 4480
#                            ...  
# Bardolino Superiore            1
# Primitivo del Tarantino        1
# Name: region_1, Length: 1230, dtype: int64
# ```

# In[13]:





# In[14]:


reviews.region_1.fillna('Unknown',inplace=True)
reviews_per_region = reviews.region_1.value_counts()

q4.check()


# In[15]:


#q4.hint()
# q4.solution()


# # Keep going
# Move on to learn about **[Renaming and combining](https://www.kaggle.com/residentmario/renaming-and-combining-reference)**.

# ---
# **[Pandas Micro-Course Home Page](https://www.kaggle.com/learn/pandas)**
# 
# 
