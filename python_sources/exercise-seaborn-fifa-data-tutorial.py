#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print("Setup Complete")


# In[32]:


# Path of the file to read
fifa_filepath = "../input/fifa.csv"


# In[33]:


# Read the file into a variable fifa_data
fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)


# In[34]:


# Read the file into a variable fifa_data
fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)


# In[35]:


# See data
print('Head Data')
print(fifa_data.head())
print('Describe :')
print(fifa_data.describe())
print('Mean Data : ')
print(fifa_data.mean())


# In[38]:


# Set the width and height of the figure
plt.figure(figsize=(16,6))
# Line chart showing how FIFA rankings evolved over time
sns.lineplot(data=fifa_data)

