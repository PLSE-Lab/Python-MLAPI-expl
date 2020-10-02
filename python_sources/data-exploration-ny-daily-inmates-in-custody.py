#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# In[ ]:


df = pd.read_csv('../input/daily-inmates-in-custody.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# Seems like DISCHARGED_DT has no useful data for us. Going to drop the entire column and remove the remaining rows that contain null values

# In[ ]:


df.drop('DISCHARGED_DT', axis = 1, inplace = True)
df.dropna(axis = 0, inplace = True)


# In[ ]:


df.info()


# In[ ]:


df.head()


# ### Inmate admission by year and month

# In[ ]:


from datetime import datetime


# In[ ]:


df['ADMITTED_DT'] = df['ADMITTED_DT'].apply(lambda x: datetime.strptime(x,"%Y-%m-%dT%H:%M:%S"))


# In[ ]:


df['admitted_year'] = df['ADMITTED_DT'].apply(lambda x: x.year)
df['admitted_month'] = df['ADMITTED_DT'].apply(lambda x: x.month)


# In[ ]:


df.head()


# In[ ]:


sns.countplot(x = 'admitted_year', data = df)


# In[ ]:


sns.catplot(x = 'admitted_year', kind = 'count', data = df, hue = 'admitted_month', height = 5, aspect = 1.4)


# Seems like there is no cyclical trend with respect to months. The admission rate based on what we have just keeps going up with time (pretty scary I think =O)

# #### Inmate age (at the point of admission) by year

# In[ ]:


current_year = datetime.now().year


# In[ ]:


# Get the age at the time when the inmates were admitted
df['admitted_age'] = df['AGE'] - (current_year - df['admitted_year'])


# In[ ]:


sns.catplot(x = 'admitted_year', y = 'admitted_age', kind = 'bar', data = df, height = 5, aspect = 1.4, ci = None)


# In[ ]:


df['admitted_year'].value_counts()


# Excluding the years where we can't get a good age estimate because our sample size is too small (i.e. before 2014), it seems like the average age (at the point of admission) of the inmates is increasing from 2015

# In[ ]:




