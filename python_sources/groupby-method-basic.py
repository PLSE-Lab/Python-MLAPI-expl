#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv(r'../input/weather_place.data.csv')
df.head()


# In[ ]:


df.drop(columns = 'Unnamed: 0', inplace=True)
df.head()


# In[ ]:


df.columns = ['Temperature','Date','Parameters','Place']
df.head()


# In[ ]:


tf = df[:12]
tf


# In[ ]:


gp = tf.groupby('Temperature')


# In[ ]:


for Temp, Index in gp:
    print("Temp:", Temp)
    print("\n")
    print("Index:",Index)


# In[ ]:


gp.get_group(0.0)


# In[ ]:


gp.max()


# In[ ]:


gp['Temperature'].mean()


# In[ ]:


gp['Temperature'].min()


# In[ ]:


gp.describe()


# In[ ]:


gp.size()

