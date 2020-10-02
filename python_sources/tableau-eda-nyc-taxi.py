#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))

import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns 


# In[ ]:


train=pd.read_csv("../input/train.csv",nrows = 2_000_000)


# In[ ]:


train.shape


# In[ ]:


train.head()


# **Check datatypes of train dataframe**

# In[ ]:


train.dtypes


# **after checking datatypes we need to convert pickup_datetime from object to datetime**

# In[ ]:


train['pickup_datetime']=pd.to_datetime(train['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')


# **Check for missing value**

# In[ ]:


train.isna().sum()


# **only 14 null /2_000_000 so we can drop**

# In[ ]:


train = train.dropna(how = 'any', axis = 'rows')


# In[ ]:


train.describe().round()


# **statstical analysis of all features**
# 
# -fare_amount will be zero in case of free but never be minus. hence we drop it.  
# -in which max passanger count is 208 if we consider taxi as bus but it can not carry 208 passanger so we drop it 
# 

# In[ ]:


train = train[train.fare_amount>=0]


# In[ ]:


train=train[train.passenger_count!=208]


# ** linear graph of passangercount vs fare_amount is not linear in realtion so only consider passanger count as important feature is not good**

# In[ ]:


sns.lmplot(x="passenger_count", y="fare_amount", data=train)
plt.show()


# ** we filter latitude in range (-90,90) and longitude(-180,180) **

# In[ ]:


train = train.drop(((train[train['pickup_latitude']<-90])|(train[train['pickup_latitude']>90])).index, axis=0)
train = train.drop(((train[train['pickup_longitude']<-180])|(train[train['pickup_longitude']>180])).index, axis=0)
train = train.drop(((train[train['dropoff_latitude']<-90])|(train[train['dropoff_latitude']>90])).index, axis=0)
train[train['dropoff_latitude']<-180]|train[train['dropoff_latitude']>180]


# > ** Drop of location and pickup location of NYC taxi **

# In[ ]:


#https://public.tableau.com/views/DashBoardNYCTAXI/Dashboard1?:embed=y&:display_count=yes
            
from IPython.display import IFrame
IFrame('https://public.tableau.com/views/DashBoardNYCTAXI/Dashboard1?:embed=y&:showVizHome=no', width=1050, height=925)


# **Lets calculate distance between two gps points using haversine distance**  
# calculated distance is in miles

# In[ ]:


from haversine import haversine
train=train.reset_index(drop=True)


# In[ ]:


train['traveldistancemiles'] = train.apply(lambda row: haversine((row['pickup_longitude'], row['pickup_latitude']), (row['dropoff_longitude'], row['dropoff_latitude']),miles=True), axis=1)


# **traveldistancemiles is new generated feature and miles above 700.0 is some how unrealistic in taxi commute so we only take below 700.0 observations**

# In[ ]:


train=train[train['traveldistancemiles']< 700.0]


# **liner plot of traveldistancemiles vs fare_amount**  
# 
# - plot is not linear so need to analyse outliers

# In[ ]:


sns.lmplot(x="traveldistancemiles", y="fare_amount", data=train)
plt.show()


# ** Stay Connected for more insights**

# In[ ]:




