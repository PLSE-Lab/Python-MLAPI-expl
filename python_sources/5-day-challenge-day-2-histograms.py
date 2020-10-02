#!/usr/bin/env python
# coding: utf-8

# From Rachel Tatman: https://www.kaggle.com/rtatman/the-5-day-data-challenge

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


# Load dataset
suicide_data = pd.read_csv('../input/master.csv')


# In[ ]:


# Describe dataset
suicide_data.describe()


# In[ ]:


# Get suicide rate column
suicide_rate_column = suicide_data['suicides/100k pop']
suicide_rate_column.sample(5)


# In[ ]:


# Plot histogram
suicide_rate_column.hist()
plt.title('Suicides per 100,000 people')
plt.ylabel('Frequency')


# In[ ]:




