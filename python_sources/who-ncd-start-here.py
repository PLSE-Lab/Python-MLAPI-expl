#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import os
path = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df1 = pd.read_csv('/kaggle/input/ncd-who-dataset/NCD_WHO_Data/Metadata_Indicator.csv')
df2 = pd.read_csv('/kaggle/input/ncd-who-dataset/NCD_WHO_Data/Metadata_Country.csv')
df3 = pd.read_csv('/kaggle/input/ncd-who-dataset/NCD_WHO_Data/WHO-cause-of-death-by-NCD.csv')


# # Data Wrangling
# * There's a lot of cleaning to do with these 3 dataframes. I will clean up each dataframe and extract the useful information, then merge them into one combined dataframe that is ready for EDA.

# In[ ]:


df1.head()


# ## df1
# * df1 provides some additional insight on the nature of the dataset. I will extract the relevant data then not use it any longer
# * 'INDICATOR_CODE' is the same in df3, so serves no additional purpose here
# * 'INDICATOR_NAME' is the same as 'Indicator Name' in df3, so also serves no additional purpose
# * 'SOURCE_NOTE' gives some useful insight, so I will print this below and take note of it
# * 'SOURCE_ORGANIZATION' tells us where the data is derived from: this is already noted on the Kaggle page for this dataset, so serves no more purpose

# In[ ]:


# SOURCE NOTE
print(df1['SOURCE_NOTE'].unique())


# ## Data description:
# * We found some useful insight on the nature of the data:
# >'Cause of death refers to the share of all deaths for all ages by underlying causes. Non-communicable diseases include cancer, diabetes mellitus, cardiovascular diseases, digestive diseases, skin diseases, musculoskeletal diseases, and congenital anomalies.'

# In[ ]:


df2.head()


# In[ ]:


df3.head()


# In[ ]:


df3.info()


# In[ ]:


# Drop years from 1960-1999 (no data)
df3 = df3.drop(df3.iloc[:,4:44],axis=1)


# In[ ]:


df3.info()


# In[ ]:


# Drop years from 2017 onwards (no data)
df3 = df3.drop(df3.iloc[:,21:25],axis=1)


# In[ ]:


df3.head()


# In[ ]:


print(df3['Indicator Name'].unique())


# In[ ]:


print(df3['Indicator Code'].unique())


# * 'Indicator Name' just gives us a brief description of the dataset. The data in the year columns (2000-2016) represent, as per the description:
# > 'Cause of death, by non-communicable diseases (% of total)'
# * 'Indicator Code' refers to the indicator of this particular dataset from the WHO website. It also refers to 'Metadata_Indicator.csv', which I have read in as df2<br/>
# **Now I drop both these columns since I have extracted the relevant information**

# In[ ]:


df3 = df3.drop(['Indicator Name','Indicator Code'],axis=1)


# In[ ]:


df3.head()


# In[ ]:




