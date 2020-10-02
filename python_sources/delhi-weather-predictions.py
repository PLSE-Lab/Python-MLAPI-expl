#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv("../input/testset.csv",parse_dates=['datetime_utc'],skipinitialspace=True)


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.shape


# In[ ]:


plt.figure(figsize=(20, 10))
p = sns.heatmap(df.corr(), annot=True)


# In[ ]:


df['Date'] = pd.to_datetime(df['datetime_utc'])
df['Year'] = df['Date'].dt.year


# In[ ]:


p = sns.lineplot(x="Year", y="_dewptm", data=df)
_ = plt.ylabel("Dew")


# In[ ]:


p = sns.lineplot(x="Year", y="_fog", data=df)
_ = plt.ylabel("Fog")


# In[ ]:


p = sns.lineplot(x="Year", y="_hum", data=df)
_ = plt.ylabel("Humidity")


# In[ ]:


p = sns.lineplot(x="Year", y="_heatindexm", data=df)
_ = plt.ylabel("Heat")


# In[ ]:


p = sns.lineplot(x="Year", y="_rain", data=df)
_ = plt.ylabel("Rain")


# In[ ]:


plt.figure(figsize=(20, 10))
p = sns.countplot(x='_conds', data=df)
_ = plt.setp(p.get_xticklabels(), rotation=90)

