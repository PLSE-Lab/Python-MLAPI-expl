#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv("/kaggle/input/bikesfortutorial/Bikes.csv", sep=",", decimal=".")


# In[ ]:


df.head(7)


# In[ ]:


df.tail(10)


# In[ ]:


df.datetime


# In[ ]:


df["datetime"]


# In[ ]:


df[["datetime", "season"]]


# In[ ]:


df.iloc[0]


# In[ ]:


df.iloc[[0,3]]


# In[ ]:


df.loc[0, "datetime"]


# In[ ]:


df.loc[[1,3], ["datetime", "hum"]]


# In[ ]:


# https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
pd.to_datetime(df.datetime, format="%Y-%m-%d %H:%M:%S")


# In[ ]:


df.datetime = pd.to_datetime(df.datetime, format="%Y-%m-%d %H:%M:%S")


# In[ ]:


df.datetime


# In[ ]:


df.datetime.dt.hour


# In[ ]:


df["hour"] = df.datetime.dt.hour


# In[ ]:


df.head()


# In[ ]:


df["weekday"] = df.datetime.dt.weekday
df["day"] = df.datetime.dt.day
df["month"] = df.datetime.dt.month
df["year"] = df.datetime.dt.year


# In[ ]:


df.head()


# In[ ]:




