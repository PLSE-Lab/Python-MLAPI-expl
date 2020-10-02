#!/usr/bin/env python
# coding: utf-8

# ### remove bad rows, export for geospatial analysis
# 
# * Note that the rating will be a LEAK if left in naively!

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


df = pd.concat([pd.read_csv('../input/CallVoiceQuality_Data_2018_May.csv'),pd.read_csv('../input/CallVoiceQualityExperience-2018-April.csv')])
print(df.shape)


# In[ ]:


print(df.shape)
df.head()


# *We see that the colum nn maes changed : In Out Travelling	Indoor_Outdoor_Travelling
# * We can merge the data or drop it 

# In[ ]:


df['Indoor_Outdoor_Travelling'].fillna(df['In Out Travelling'],inplace=True)
df.drop('In Out Travelling',axis-1,inplace=True)


# In[ ]:


df = df.loc[(df.Latitude != -1) & (df.Longitude != -1)]
print("Cleaned DF shape",df.shape)


# In[ ]:


## There are many duplicates , but this is OK given the data
df.drop_duplicates().shape


# In[ ]:


df.to_csv("IndiaCallQuality.csv",index=False)


# In[ ]:


df.drop_duplicates().to_csv("IndiaCallQuality_Deduplicated.csv.gz",index=False,compression="gzip")

