#!/usr/bin/env python
# coding: utf-8

# # Intro
# **This is your workspace for Kaggle's Machine Learning course**
# 
# You will build and continually improve a model to predict housing prices as you work through each tutorial.
# 
# The tutorials you read use data from Melbourne. The Melbourne data is not available in this workspace.  Instead, you will translate the concepts to work with the data in this notebook, the Iowa data.
# 
# # Write Your Code Below
# 

# In[1]:


import pandas as pd 

melbourne_data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')
print(melbourne_data.describe())


# In[2]:


print(melbourne_data.columns)


# In[5]:


melbourne_price_data = melbourne_data.Price
print(melbourne_price_data.head())


# In[9]:


my_column = ['Longtitude','Landsize']
column_data = melbourne_data[my_column]


# In[12]:


column_data.describe()


# 
# **If you have any questions or hit any problems, come to the [Learn Discussion](https://www.kaggle.com/learn-forum) for help. **
# 
# **Return to [ML Course Index](https://www.kaggle.com/learn/machine-learning)**

# 
