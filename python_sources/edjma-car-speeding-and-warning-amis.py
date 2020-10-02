#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('bmh')


# In[ ]:


df = pd.read_csv('../input/car-speeding-and-warning-signs/amis.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


print(df['speed'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['speed'], color='g', bins=100, hist_kws={'alpha': 0.4})


# In[ ]:


print(df['period'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['period'], color='g', bins=100, hist_kws={'alpha': 0.4})


# In[ ]:


list(set(df.dtypes.tolist()))


# In[ ]:


df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num.head()


# In[ ]:


df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)


# In[ ]:


for i in range(0, len(df_num.columns), 5):
    sns.pairplot(data=df_num,
                x_vars=df_num.columns[i:i+5],
                y_vars=['speed'])

