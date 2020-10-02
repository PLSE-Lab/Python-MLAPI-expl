#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import folium

import os
print(os.listdir("../input"))


# Loading the film permits data

# In[ ]:


permit_data = pd.read_csv("../input/film-permits.csv")


# In[ ]:


permit_data.shape


# The datasets consists of 14 columns and 41191 rows

# In[ ]:


permit_data.head()


# The dataset consists of 14 columns such as:
# 
# 1. EventID - The unique ID for each events
# 2. EventType - The information of the event type
# 3. StartDateTime - The start scheduled time of the event
# 4. EndDateTime - The end scheduled time of the event
# 5. EnteredOn - The date when submitted requests to Mayor's Office of Media and Entertainment (MOME)
# 6. EventAgency - The agency of the scheduled event
# 7. ParkingHeld - The location of the parking hold for the events in advance
# 8. Borough - The city for the event
# 9. CommunityBoard(s) - The number of community boards for the event
# 10. PolicePrecinct(s) - The police for the precinct of activity for the day
# 11. Category - The category of the event medium
# 12. SubCategoryName - The sub category of the event medium
# 13. Country - The origin of the project
# 14. ZipCode(s) - The first zipcode of the activity

# ## Analysis of Event Type column

# Let's have a look of the event type and get the insights of the column by examing it.
# 
# We can get the number of event types.

# In[ ]:


unique_event_type = permit_data.EventType.unique()
print('The unique event types are ', str(unique_event_type))


# Let's do the visualisation to see which event type is occuring more in the film permits.

# In[ ]:


sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(20, 10)

sns.countplot(x="EventType", data=permit_data, ax=ax)
sns.despine()


# In[ ]:


pd.value_counts(permit_data.EventType)


# We can see that the Shooting Permit is the most event type that occurs and the count is around 36198 and the Least event type is DCAS Prep/Shoot/Wrap Permit
# 
# Let's see which Borough(city) is often used in all the types of events.

# In[ ]:


sns.set_style('ticks')
fig, ax = plt.subplots(ncols=4)
fig.set_size_inches(20, 10)

sns.countplot(x="Borough", data=permit_data[permit_data['EventType'] == 'Shooting Permit'], ax=ax[0])
ax[0].set(xlabel='Shooting Permit')
sns.countplot(x="Borough", data=permit_data[permit_data['EventType'] == 'Rigging Permit'], ax=ax[1])
ax[1].set(xlabel='Rigging Permit')
sns.countplot(x="Borough", data=permit_data[permit_data['EventType'] == 'Theater Load in and Load Outs'], ax=ax[2])
ax[2].set(xlabel='Theater Load in and Load Outs')
sns.countplot(x="Borough", data=permit_data[permit_data['EventType'] == 'DCAS Prep/Shoot/Wrap Permit'], ax=ax[3])
ax[3].set(xlabel='DCAS Prep/Shoot/Wrap Permit')
sns.despine()


# Based on the above visualisation, we can state that:
# 1. Manhattan is the top choice for all the event types
# 2. Brooklyn is the second choice except for the DCAS Prep/Shoot/Wrap Permit which is Bronx
# 3. Staten Island is the least choice for all the event types

# ## Analysis of the Time column

# The startDateTime and the endDateTime are the columns which gives the scheduled time for the event. The time is given in ISO format. We can calculate the time in minutes using the **time** package, which is imported at the top.

# In[ ]:


permit_data['EventTimeinHours'] = permit_data.apply(lambda x: abs(time.mktime(time.strptime(x['StartDateTime'], '%Y-%m-%dT%H:%M:%S.%f'))-time.mktime(time.strptime(x['EndDateTime'], '%Y-%m-%dT%H:%M:%S.%f')))/(60*60), axis=1)
permit_data.head()[['StartDateTime', 'EndDateTime', 'EventTimeinHours']]


# The **EventTimeinHours** consists of the total scheduled time in hours for each events. We can see that the maximum time schduled for an event is the event happened in Manhattan for a television Talk show which went nearly 1 year (358.7 days)

# In[ ]:


permit_data[permit_data['EventTimeinHours'] == max(permit_data['EventTimeinHours'])]['EventTimeinHours']/24


# The minimum event registered in the dataset is 0.016667 hours. Clearly we can see that the data is incorrect and there should be a wrangling process to fix the data

# In[ ]:


permit_data[permit_data['EventTimeinHours'] == min(permit_data['EventTimeinHours'])]['EventTimeinHours']


# ## Analysis of Category and Sub-category data

# The category of the events in the datasets are :

# In[ ]:


permit_data.Category.unique()


# Now, Let's plot the visualisation to find the top category of the event that happened

# In[ ]:


sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(20, 10)

sns.countplot(x="Category", data=permit_data, ax=ax, order = permit_data.Category.value_counts().index)
sns.despine()


# Television is the top category of the events that has been happened and the Least is the Music Video
# 
# The sub-category can be visualizated as follows:

# In[ ]:


print('The number of sub-category are ', len(permit_data.SubCategoryName.unique()))

sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(20, 10)

sns.countplot(y="SubCategoryName", data=permit_data, ax=ax, order = permit_data.SubCategoryName.value_counts().index)
sns.despine()


# The total sub category are 29 and most of the events are taken for the Episodic series
# 
# Let's have a insight for the top most category ('Television') using the visualisation

# In[ ]:


sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(20, 10)

sns.countplot(y="SubCategoryName", data=permit_data[permit_data.Category == 'Television'], ax=ax)
sns.despine()


# The Episodic Series is the top most sub-category, second is the Cable-episodic and the third is the Pilot

# ## Analysis of Country column
# The main origin countries for the events are :

# In[57]:


permit_data.Country.unique()


# Let's group the data by the country and the event type and visualize the data

# In[58]:


group_event_country = permit_data.groupby(['EventType', 'Country']).size()
group_by_country = pd.DataFrame(group_event_country).reset_index()
group_by_country


# This are the coordinates below of the countries in the dataset

# In[62]:


geo_corrs = {
    'Netherlands': (52.2379891, 5.53460738161551),
    'United States of America' : (39.7837304, -100.4458825),
    'Australia': (-24.7761086, 134.755),
    'Canada': (61.0666922, -107.9917071),
    'France': (46.603354, 1.8883335),
    'Germany': (51.0834196, 10.4234469),
    'Japan': (36.5748441, 139.2394179),
    'Panama': (8.3096067, -81.3066246),
    'United Kingdom': (54.7023545, -3.2765753)
}


# In[63]:


m = folium.Map(location=[20,0], tiles="Mapbox Bright", zoom_start=2)

for i in range(0,len(group_by_country)):
    corr = geo_corrs[group_by_country['Country'][i]]
    folium.Circle(
      location=[corr[0], corr[1]],
      popup=group_by_country['Country'][i],
      radius=int(group_by_country[0][i])*10
    ).add_to(m)

m


# In[ ]:




