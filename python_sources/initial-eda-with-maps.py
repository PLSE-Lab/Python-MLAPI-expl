#!/usr/bin/env python
# coding: utf-8

# Hello everyone!!. In this following kernel I will do the EDA for this dataset. Please do upvote if you find it useful!!
# Please mention all the improvements or mistakes and new ideas as well in the comments below
# 
# Loading of the maps could be slower, but I will be improving them by aggregations

# # 1) Imorting Necessary modules

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import matplotlib.pyplot as plt
plt.style.use('ggplot')
pd.set_option('display.max_columns', 500)


# In[ ]:


px.set_mapbox_access_token('pk.eyJ1IjoiaGFyaXN5YW0iLCJhIjoiY2poZHRqMGV4MG93MDNkcXZqcmQ3b3RzcSJ9.V-QDWKoYu_6OqATbmH9ocw')


# In[ ]:


train_df = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv')
test_df = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/test.csv')


# In[ ]:


train_df.head(10)


# ***Observations:***
# * Every Intersection has an Id, and it is marked by Latitude and Longitude and labeled by City
# * Entry and Exit street were given along with directions
# * Temporal Information - Time, Boolean Weekend variable and Month
# * Path - is basically the string concat of the Spatial columns --> Entry/Exit Street Name + Entry/Exit Heading
# * Intermediate Target Variables - TimeFromFirstStop
# * Target Variables - TotalTimeStopped, DistanceToFirstStop -- 20th,40th ,50th, 60th & 80th percentiles

# In[ ]:


train_df.info()


# # 2) EDA

# In[ ]:


fig, axarr = plt.subplots(1,2, figsize=(18, 8))
sns.countplot(x='City',data=train_df,ax=axarr[0]);
axarr[0].set_title('measurements per city in Train Set')
axarr[0].set_ylabel('# of Observations in Train Set');
sns.countplot(x='City',data=test_df,ax=axarr[1]);
axarr[1].set_title('measurements per city  in Test Set')
axarr[1].set_ylabel('# of Observations in Test Set');


# ***Observation:*** Train set and test have similar %observations in the 4 cities

# In[ ]:


fig, axarr = plt.subplots(1, 2, figsize=(15, 8))
train_df.groupby(['City']).IntersectionId.nunique().sort_index().plot.bar(ax=axarr[0])
axarr[0].set_title('# of Intersections per city in Train Set')
axarr[0].set_ylabel('# of Intersections');
test_df.groupby(['City']).IntersectionId.nunique().sort_index().plot.bar(ax=axarr[1])
axarr[1].set_title('# of Intersections per city in Test Set')
axarr[1].set_ylabel('# of Intersections');


# **Observation: ** 
# * The train set have less number of Intersections per city when compared to test set. Need to confirm!!
# * In the train set and test set --> Chicago had higher number of Intersections

# In[ ]:


print('Number of Entry Headings in Train Set: ', len(train_df.EntryHeading.unique()))
print('Number of Exit Headings in Train Set: ', len(train_df.ExitHeading.unique()))
print('Number of Entry Street Names in Train Set: ', len(train_df.EntryStreetName.unique()))
print('Number of Exit Street Names in Train Set: ', len(train_df.ExitStreetName.unique()))

print('Number of Entry Headingds in Test Set: ', len(test_df.EntryHeading.unique()))
print('Number of Exit Headings in Test Set: ', len(test_df.ExitHeading.unique()))
print('Number of Entry Street Names in Test Set: ', len(test_df.EntryStreetName.unique()))
print('Number of Exit Street Names in Test Set: ', len(test_df.ExitStreetName.unique()))


# In[ ]:


train_intersections_count=train_df.groupby(['City','Latitude','Longitude']).IntersectionId.count().reset_index()
train_intersections_count.columns=['City','Latitude','Longitude','Count_Obs']


# In[ ]:


fig = px.scatter_mapbox(train_intersections_count[train_intersections_count.City=='Atlanta'], lat="Latitude", lon="Longitude",size="Count_Obs",color="Count_Obs",  
                        color_continuous_scale=px.colors.sequential.Viridis, size_max=15, zoom=10)
fig.update_layout(mapbox_style="open-street-map")
fig.show()


# In[ ]:


fig = px.scatter_mapbox(train_intersections_count[train_intersections_count.City=='Chicago'], lat="Latitude", lon="Longitude",size="Count_Obs",color="Count_Obs",  
                        color_continuous_scale=px.colors.sequential.Viridis, size_max=15, zoom=9)
fig.update_layout(mapbox_style="open-street-map")
fig.show()


# In[ ]:


fig = px.scatter_mapbox(train_intersections_count[train_intersections_count.City=='Philadelphia'], lat="Latitude", lon="Longitude",size="Count_Obs",color="Count_Obs",  
                        color_continuous_scale=px.colors.sequential.Viridis, size_max=15, zoom=10)
fig.update_layout(mapbox_style="open-street-map")
fig.show()


# In[ ]:


fig = px.scatter_mapbox(train_intersections_count[train_intersections_count.City=='Boston'], lat="Latitude", lon="Longitude",size="Count_Obs",color="Count_Obs",  
                        color_continuous_scale=px.colors.sequential.Viridis, size_max=15, zoom=10)
fig.update_layout(mapbox_style="open-street-map")
fig.show()


# **Observations:**
# *  In every city I see that more number of observations are recorded at the Intersections near to the Airports
# *  Temporal Information will help in more insights here

# # Topics still need to be answered/covered!!!
# * How to combine the percentile values for the Target Variables for visualization?
# * Would like to create a distance features from POI's like airports and shopping centres to Intersections...can be a very useful information
# * Plots using Temporal Information
# 

# **!!! WILL be updated!!!**
