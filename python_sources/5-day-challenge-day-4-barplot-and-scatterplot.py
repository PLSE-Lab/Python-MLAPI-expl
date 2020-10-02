#!/usr/bin/env python
# coding: utf-8

# From Rachel Tatman: https://www.kaggle.com/rtatman/the-5-day-data-challenge

# In[22]:


import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats


# In[2]:


# Load data
dataframe = pd.read_csv('../input/DigiDB_digimonlist.csv')


# In[3]:


# Show part of data 
dataframe.head()


# In[16]:


# Barplot 
sns.barplot(x = 'Type', y = 'Memory', data = dataframe).set_title('Memory vs. Digimon Type')


# In[21]:


# Scatterplot
sns.scatterplot(x = 'Memory', y = 'Lv 50 HP', data = dataframe).set_title('Memory vs. Lv 50 HP')


# In[26]:


# Pearson Correlation and P-value of Scatterplot
stats.pearsonr(x = dataframe['Memory'], y = dataframe['Lv 50 HP'])


# In[27]:


# Low P-value indications that low probability of null hypothesis being correct, meaning high chance that the correlation is significant. 


# In[ ]:




