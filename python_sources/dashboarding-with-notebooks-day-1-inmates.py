#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('white')

import os
print(os.listdir("../input"))


# In[ ]:


# read in the data
inmates = pd.read_csv('../input/daily-inmates-in-custody.csv')
inmates.head()


# In[ ]:


# descriptive statistics
inmates.describe(include='all')


# In[ ]:


# info
inmates.info()


# In[ ]:


# let's drop DISCHARGED_DT, as it has no data and TOP_CHARGE, since it's missing a bunch of values
inmates = inmates.drop(['DISCHARGED_DT', 'TOP_CHARGE'], axis=1)
# let's drop the remaining rows missing values
inmates = inmates.dropna()


# In[ ]:


# updated info
inmates.info()


# Today's goal is to put together a couple visualizations of factors in which we are interested from this dataset.  It would be interesting to look at the distribution of ages across inmates.  It would also be interesting to look at the distribution of races across inmates.  Finally, it would also be interesting to look at how long it has been since inmates were admitted.

# In[ ]:


# visualization for age
plt.figure(figsize=(15,6))
sns.distplot(inmates['AGE'])
plt.show()


# In[ ]:


# visualization for race
plt.figure(figsize=(15,6))
sns.countplot(inmates['RACE'])
plt.show()


# In[ ]:


# convert to datetime
inmates['ADMITTED_DT'] = pd.to_datetime(inmates['ADMITTED_DT'])
# calculate difference betweeen admitted and today in years
inmates['YEARS_IN'] = (pd.to_datetime('today') - inmates['ADMITTED_DT']) / pd.Timedelta('365.25 days')
inmates[['ADMITTED_DT','YEARS_IN']].head()


# In[ ]:


# visualization for YEARS_IN
plt.figure(figsize=(15,6))
sns.distplot(inmates['YEARS_IN'], kde=False)
plt.show()


# In[ ]:




