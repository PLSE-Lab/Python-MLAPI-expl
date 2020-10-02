#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Anaysis is Performed by Shubham S Kale as a Personal Project 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/data.csv',encoding = "ISO-8859-1")
data.head()


# In[ ]:


data.tail()


# In[ ]:


data.size


# In[ ]:


data.columns


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


# columns : ['stn_code', 'sampling_date','agency','location_monitoring_station'] seems to be useless , so lets drop them
data.drop(['stn_code', 'sampling_date','agency','location_monitoring_station'],axis=1,inplace=True)
data.head()


# In[ ]:


data.tail(10)


# In[ ]:


#dropping last three rows as most of the values are "NaN" & forming a new dataframe 'data_new'
data_new=data.drop(range(435739,435742))


# In[ ]:


data_new.tail(10)


# In[ ]:


data_new.columns


# In[ ]:


data_new.dtypes


# In[ ]:


data_new.describe()


# In[ ]:


#converting object types to string
data_new['state']=data_new.state.astype('str',inplace=True)
data_new['location']=data_new.location.astype('str',inplace=True)
data_new['type']=data_new.type.astype('str',inplace=True)
data_new['date']=data_new.date.astype('str',inplace=True)
data_new.info()


# Pandas only has a few specific data types that mainly handle numeric values. 
# For everything else it uses the object type. Strings fit into the latter category, and 
# so it is likely that your data is already strings, even though the data type is listed as object.

# In[ ]:


#replacing "NaN" values in above described columns with zero
import numpy as np
data_new.fillna(0.0,inplace=True)
data_new.head(10)


# In[ ]:


data_new.tail(10)


# In[ ]:


#Exploring relationship between proportion of Sulphur dioxide & Nitrogen dioxide
import seaborn as sns
sns.regplot(x=data_new['so2'],y=data_new['no2'],data=data_new)


# Above plot shows that over the years there has been continuos increase in proportion of Sulphur dioxide and Nitrogen dioxide

# In[ ]:


#Exploring air pollution state-wise
states=data_new.groupby(['state','location'],as_index=False).mean()
states


# In[ ]:


#location with highest Sulphur dioxide,Nitrogen dioxide, RSPM and SPM separately content in air
print("Location with highest SO2 content in air :-\n\n")
print(states[states['so2']==(states['so2'].max())])
print("Location with highest NO2 content in air :-\n\n")
print(states[states['no2']==(states['no2'].max())])
print("Location with highest RSPM content in air :-\n\n")
print(states[states['rspm']==(states['rspm'].max())])
print("Location with highest SPM content in air :-\n\n")
print(states[states['spm']==(states['spm'].max())])


# In[ ]:


state=states.groupby(['state'],as_index=False).mean()


# In[ ]:


#new dataframe with data related to states only
state


# In[ ]:


#adding a column of total to the 'state' dataframe
state['total']=state.sum(axis=1)
state.head()


# In[ ]:


print("The State with highest amount of air-pollution is :-\n\n")
print(state[state['total']==(state['total'].max())])


# 

# So in India Highest Air Pollution is in **New Delhi**

# In[ ]:


print("The State with lowest amount of air-pollution is :-\n\n")
print(state[state['total']==(state['total'].min())])


# So in India Least Air Polluted state  is  **Mizoram**

# In[ ]:


state=state.sort_values(['total'],ascending=False)
print("Top 5 Most Populated States are :-\n\n")
state.head()


# In[ ]:


print("Top 5 Least Populated States are :-\n\n")
state.tail().sort_values(['total'],ascending=True)


# In[ ]:


import folium


# In[ ]:


india_map=folium.Map(location=[20.5937,78.9629],zoom_start=5,tiles='Stamen Terrain')
airp = folium.map.FeatureGroup()

# Highest Pollution
airp.add_child(
    folium.features.CircleMarker(
            [28.6139, 77.2090],
            radius=5, # define how big you want the circle markers to be
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )
folium.Marker([28.6139, 77.2090], popup='Highly Poluted State').add_to(india_map)  
# Lowest Pollution
airp.add_child(
    folium.features.CircleMarker(
            [23.1645, 92.9376],
            radius=5, # define how big you want the circle markers to be
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )
folium.Marker([23.1645, 92.9376], popup='Least Poluted State').add_to(india_map)  
    
# add incidents to map
india_map.add_child(airp)


# Above Map shows locations of highly air polluted state and least air polluted state

# In[ ]:




