#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('../input/winemag-data_first150k.csv')


# In[ ]:


df.head()


# **1**

# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.tail()


# In[ ]:


df.describe()


# **2**

# In[ ]:


df.isnull().sum()


# In[ ]:


df.price.isnull().sum()


# In[ ]:


df[(df['price'] >45) & (df['price'] <100)].head()


# **3. Detele the points column**

# In[ ]:


df.drop('points', axis=1, inplace=True)


# In[ ]:


df.head()


# **4**

# In[ ]:


df.country


# In[ ]:


df.country.nunique()


# In[ ]:


df.points


# In[ ]:


df.points.nunique()


# **5. Where is the most expensive\the cheapest wine?**

# In[ ]:


df[df['price'] == df['price'].max()].head()


# In[ ]:


df[df['price'] == df['price'].min()].head()


# **6. Rename the variety column**

# In[ ]:


df.rename({'variety':'diversity'}, axis=1).head()


# **7**

# In[ ]:


df.loc[:, ['country', 'winery']]


# In[ ]:


df.loc[:,['country', 'region_1', 'region_2']].sort_values(['region_1', 'region_2'], ascending=[False,False])


# **8.Groupby**

# In[ ]:


df.groupby('country')['price'].agg(['max', 'min'])


# **9. First five country with highests wine review**

# In[ ]:


df['country'].value_counts().head(5)


# **10. Apply**
# *Capitalize strings*

# In[ ]:


df['high_points'] = df['points'].apply(
    lambda x: x > 80
)


# In[ ]:


df.head()


# **11**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
pr = df.price.plot.hist()
pr.set_title('Price Histogram', fontdict={'fontsize': 20})
pr.figure.savefig('gistogram.png')


# **12**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
po = df.points.plot.kde()
po.set_title('Points', fontdict={'fontsize': 20})
po.figure.savefig('points.png')


# In[ ]:




