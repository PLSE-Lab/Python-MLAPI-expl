#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv(r'../input/weather_place.data.csv',parse_dates=['date'])
df.head()


# In[ ]:


df.drop(columns = 'Unnamed: 0' , inplace = True)
df.head()


# In[ ]:


df.isnull().values.any()


# In[ ]:


new_df = df.fillna(0)
new_df


# In[ ]:


new_df = df.fillna({
        'temperature': 0,
        'windspeed': 0,
        'event': 'no event'
    })
new_df


# In[ ]:


new_df = df.fillna(method="ffill")
new_df

