#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


print(df['country'])


# In[ ]:


plt.figure(figsize = (20,6))
sns.lineplot(x = 'country', y = 'suicides_no', data = df)


# In[ ]:


plt.figure (figsize = (20,6))
sns.barplot(x = 'country', y = 'suicides_no', data = df)


# In[ ]:


plt.figure(figsize = (20,6))
sns.lineplot(x = 'country', y = 'year', data = df)


# In[ ]:


plt.figure(figsize = (20,6))
sns.barplot(x = 'country', y = 'year', data = df)


# In[ ]:


plt.figure(figsize = (20,6))
sns.lineplot(x = 'sex', y = 'suicides_no', data = df)


# In[ ]:


plt.figure(figsize = (20,6))
sns.barplot(x = 'sex', y = 'suicides_no', data = df)


# In[ ]:


plt.figure(figsize = (20,6))
sns.barplot(x = 'country', y = 'gdp_per_capita ($)', data = df)


# In[ ]:


sns.pairplot(df)


# In[ ]:




