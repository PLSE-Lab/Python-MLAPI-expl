#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
df = pd.read_csv(r'../input/weather_place.data.csv' )
df.drop(columns = 'Unnamed: 0', inplace = True)
df.head()


# In[ ]:


df.tail()


# In[ ]:


df.columns = ['Temp','Date','Parameters','Place']
df.head(3)


# In[ ]:


tf = df[:10]
tf.head(10)


# In[ ]:


tf.pivot(index = 'Place', columns = 'Date')


# In[ ]:


tf.pivot_table(index='Place',columns ='Date',margins = True,aggfunc = np.sum)


# In[ ]:




