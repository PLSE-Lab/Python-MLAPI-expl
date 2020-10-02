#!/usr/bin/env python
# coding: utf-8

# In[16]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))


# NOTES:
# * Basic Stats (Mean, Median, Mode, Standard Deviation, Variance, Quartile, Range)
# * Box and Whisker Plot
# * Histogram
# * Coorelation Vs Covariance
# 
# 

# In[17]:


import matplotlib.pyplot as plt


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


plt.plot([1,2,3,4]) #test code


# <H2> To Read the CSV file </H2>

# In[20]:


phony = pd.read_csv('../input/bigml_59c28831336c6604c800002a.csv')


# <H2> Get the Head of the data set </H2>

# In[21]:


phony.head()


# <H2> Get the column names </H2>

# In[22]:


phony.columns


# In[23]:


phony.describe()


# ## Box and Whisker Plot

# In[38]:


phony.boxplot(input(),rot=45, fontsize=15, grid = False)


# ## Correlation Table

# In[33]:


phony.corr()


# ## Histogram

# In[36]:


phony.hist('account length', grid = False)


# In[40]:


import seaborn as sns
# Density Plot and Histogram of all arrival delays
sns.distplot(phony['account length'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 6})


# In[41]:


phony.shape


# In[48]:


data = pd.read_csv(input())


# In[ ]:




