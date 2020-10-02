#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

df = pd.read_csv(r'../input/weather_place.data.csv')
df.head()


# In[ ]:


df.drop(columns='Unnamed: 0', inplace=True)
df.head()


# In[ ]:


df.columns = ['Temperature','Date','Parameters','Place']
df.head()


# In[ ]:


df.isnull().values.any()


# In[ ]:


tf = df[:10]


# In[ ]:


tf.head()


# In[ ]:


import numpy as np
tf.replace(0.00 , value=np.NaN)


# In[ ]:


tf.replace(0.00 , tf.mean())


# In[ ]:




