#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df =pd.read_csv("../input/flight-delay-prediction/Jan_2019_ontime.csv")


# In[ ]:


df.describe()
plt.hist(x=df.DAY_OF_WEEK)


# In[ ]:


plt.figure(figsize=(20,20))
sns.countplot(x=df.ORIGIN)


# In[ ]:


sns.distplot(df.DAY_OF_MONTH)


# In[ ]:


z=df[df.DAY_OF_MONTH==1]


# In[ ]:


# let see what happened on new year


# In[ ]:


sns.catplot(x="OP_CARRIER", y="DISTANCE", data=z)


# In[ ]:


plt.plot(df.groupby("OP_CARRIER")["DISTANCE"].sum())


# In[ ]:




