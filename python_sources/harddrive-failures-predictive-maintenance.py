#!/usr/bin/env python
# coding: utf-8

# # Filter data for Predictive maintenance = Time To Event Model
# * We'll filter for HDDs that failed in the given year (both for convenience and file size).
# * We will later use this data to create a regression model - time to failure. 

# In[30]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[31]:


df = pd.read_csv('../input/harddrive.csv')
print(df.shape)
df.head()


# In[32]:


# drop constant columns
df = df.loc[:, ~df.isnull().all()]
print(df.shape)


# In[33]:


# number of hdd
print("number of hdd:", df['serial_number'].value_counts().shape) 

# number of different types of harddrives
print("number of different harddrives", df['model'].value_counts().shape)


# ## Keep only HDDs that failed
# * Ideally we would use multiple years of data for more models and range. There's a huge bias here when we don't include HDDs that did not fail! 
#     * We also lose the ability to identify truly "healthy" HDDs.. 

# In[34]:


failed_hdds = df.loc[df.failure==1]["serial_number"]
len(failed_hdds)


# In[35]:


df = df.loc[df["serial_number"].isin(failed_hdds)]
df.shape


# In[36]:


df["end_date"] = df.groupby("serial_number")["date"].transform("max")


# In[37]:


df.tail()


# In[38]:


df["end_date"] = pd.to_datetime(df["end_date"])
df["date"] = pd.to_datetime(df["date"])


# In[39]:


df["date_diff"] = df["end_date"] - df["date"]
df["date_diff"].describe()


#  * It looks like the failures have an odd distribution. maybe they're clustered around the start/end of a year/quarter. 
#  
#  * NOTE that leaving in the date will result in a leak in our model if we're not careful (later in the [same] year = higher likelihood of failure!

# In[40]:


# replace string/object with number
df["date_diff"] = df["date_diff"].dt.days
df.drop(["end_date","failure"],axis=1,inplace=True)
print(df.shape)


# ## Save our data
# 

# In[41]:


df.to_csv("smartHDD_Failures_2016_survival.csv.gz",index=False,compression="gzip")
print("done")


# In[ ]:




