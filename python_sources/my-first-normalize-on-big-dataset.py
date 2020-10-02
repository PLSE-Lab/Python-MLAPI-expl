#!/usr/bin/env python
# coding: utf-8
# *Normalize first big Dataset*
# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np


# Import data
# 

# In[ ]:



df = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')


# Show shape
# 

# In[ ]:


df.shape


# Controll if are present missing values.

# In[ ]:


df.isnull().sum()


# Missing Values are not existing...

# In[ ]:


for x in ["SO2","NO2","O3","CO","PM10","PM2.5"]:
    print(x+" : ")
    print(df[x].describe())


# I remove values content a illegal value(-1 and 0)

# In[ ]:


for x in ["SO2","NO2","O3","CO","PM10","PM2.5"]:
    df=df[df[x]>0]


# In[ ]:


df.describe()


# Now let's start see if address and Latitude/Longitude are correlated with Station code

# In[ ]:


df[df["Station code"]==101]


# Ok are all correlated. Now we start to clear address,latitude and longitude

# In[ ]:


del df["Latitude"]
del df["Longitude"]
del df["Address"]


# Now i'm start create a function for separe Measurement date in hour and date

# In[ ]:


def normalize_data(x):
    index=[]
    
    for date in x:
        normalized= date.split(' ')
        normalized= normalized[0].split('-')+normalized[1].split(':')
        index.append(normalized)
        
    
    return pd.DataFrame(index,columns=["Year","Month","Day","Hour","Min"])


# I apply this function

# In[ ]:


df_data_normalized=normalize_data(df["Measurement date"])
print(f'Normalized date shape {df_data_normalized.shape}.')
print(f'Original Df shape {df.shape}.')

df_normalized=pd.concat([df,df_data_normalized],axis=1)
print(f'Final shape {df_normalized.shape}')


# In[ ]:


df_normalized


# I dont know why create a final Df with new cases. Just i'll remove this Nan Values

# In[ ]:


df_final=df_normalized.dropna()


# In[ ]:


del df_final['Measurement date']


# In[ ]:


df_final


# I see if all "Min" values are 0

# In[ ]:


df_final['Min'].value_counts()


# In[ ]:


del df_final['Min']


# Now we have a good and normalized dataset for do on it Learning

# In[ ]:


df_final


# If u have any tip for me don't be timid!

# In[ ]:




