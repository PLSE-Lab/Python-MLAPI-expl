#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df_fitbit = pd.read_csv('/kaggle/input/one-year-of-fitbit-chargehr-data/One_Year_of_FitBitChargeHR_Data.csv')


# In[ ]:


df_fitbit.info()


# In[ ]:


df_fitbit.head(2)


# In[ ]:


df_fitbit['Date'] = pd.to_datetime(df_fitbit['Date'])


# In[ ]:


_=df_fitbit.set_index('Date').plot(subplots=True, figsize=(10, 50))


# In[ ]:


_=df_fitbit.set_index('Date').rolling(7).mean().plot(subplots=True, figsize=(10, 50))

