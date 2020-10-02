#!/usr/bin/env python
# coding: utf-8

# ## UBER CASE STUDY
# 
# ### Business Objectives
# The aim of analysis is to identify the root cause of the problem (i.e. cancellation and non-availability of cars) and recommend ways to improve the situation. As a result of your analysis, you should be able to present to the client the root cause(s) and possible hypotheses of the problem(s) and recommend ways to improve them.  
# 
# **Note:** For this case study, only the trips to and from the airport are being considered.

# ### 1. Visually identify the most pressing problems for Uber. 
# 
# - Create plots to visualise the frequency of requests that get cancelled or show 'no cars available'; identify the most problematic types of requests (city to airport / airport to city etc.) and the time slots (early mornings, late evenings etc.) using plots
# 
# ### 2. Find out the gap between supply and demand and show the same using plots.
# - Find the time slots when the highest gap exists
# - Find the types of requests (city-airport or airport-city) for which the gap is the most severe in the identified time slots

# In[ ]:


#importing data from CSV file into pandas dataframe

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
 
df_raw = pd.read_csv('/kaggle/input/Uber Request Data.csv')
df_raw.head()


# In[ ]:


df_uber=df_raw.copy()

#addressing data quality issues and converting request and drop timestamp to datetime format
df_uber['Request timestamp']=pd.to_datetime(df_raw['Request timestamp'])
df_uber['Drop timestamp']=pd.to_datetime(df_raw['Drop timestamp'])

#replacing blanks in column name 
df_uber.columns = [col.replace(' ', '_') for col in df_uber.columns]

#drop unnecessary columns
df_uber=df_uber.drop(['Request_id','Driver_id','Drop_timestamp'], axis=1)
df_uber.head()


# In[ ]:


#checking unique values in status column
df_uber['Status'].unique()


# In[ ]:


#dividing the trips into 6 sessions based on dt.hour from Request Timestamp

session_labels=['Late Night','Early Morning','Late Morning','Afternoon','Evening','Night']
df_uber=df_uber.assign(session=pd.cut(df_uber.Request_timestamp.dt.hour,[-1,4,8,12,16,20,24],labels=session_labels))
df_uber.head()


# #### Session Details
# - Late Night: Midnight to 4AM
# - Early Morning: 4AM-8AM
# - Late Morning: 8AM-Noon
# - Afternoon: Noon-4PM
# - Evening: 4PM-8PM
# - Night: 8PM-Midnight

# In[ ]:


# plotting share/frequency of all "Trip Status" over the day to identify problem areas
plt.style.use('ggplot')
colors = ["#CC2529", "#8E8D8D","#008000"]
df_uber.groupby(['session','Status']).Status.count().unstack().plot.bar(legend=True, figsize=(15,10), color=colors)
plt.title('Total Count of all Trip Statuses')
plt.xlabel('Sessions')
plt.ylabel('Total Count of Trip Status')
plt.show()


# ### Observations: 
# 
# #### 1. Maximum number of "No Cars Available" status trips can be seen in Evening Session (Time Slot: 4PM to 8PM), followed by Night session (8PM to Midnight).
# 
# #### 2. Maximum number of "Cancelled" status trips can be seen in Early Morning Session (Time Slot: 4AM to 8AM), followed by Late Morning session (8AM to Noon).
# 

# ### Let's drill down further in each category to do a deeper analysis

# In[ ]:


# Filtering out only "Cancelled"  trips
df_tripscancelled=df_uber[df_uber["Status"].str.contains('Cancelled')==True]
df_tripscancelled=df_tripscancelled.reset_index(drop=True)
df_tripscancelled.head()


# In[ ]:


# plotting share/frequency of all Cancelled trips over the day to identify problem areas
plt.style.use('ggplot')
colors = ["#20B2AA", "#9400D3"]
df_tripscancelled.groupby(['session','Pickup_point']).Pickup_point.count().unstack().plot.bar(legend=True, figsize=(15,10), color=colors)
plt.title('Count and Distribution of all "Cancelled" Trips over the day')
plt.xlabel('Sessions')
plt.ylabel('Total Count of "Cancelled" Trips')
plt.show()


# ### Observation
#  1. Maximum Cancellations happen in "Early Morning" (4AM-8AM) session where the pick up is "City", followed by Late Morning session (8AM-Noon)
# 
# ### Inference:
# **Most number of cancellations** happen in the **Morning hours (4AM-8AM)** in the **City to Airport** route

# In[ ]:


# Filtering out only "Cancelled"  trips
df_nocar=df_uber[df_uber["Status"].str.contains('No Car')==True]
df_nocar=df_nocar.reset_index(drop=True)
df_nocar.head()


# In[ ]:


plt.style.use('ggplot')
colors = ["#20B2AA", "#9400D3"]
df_nocar.groupby(['session','Pickup_point']).Pickup_point.count().unstack().plot.bar(legend=True, figsize=(15,10), color=colors)
plt.title('Count and Distribution of all "No Car Available" Trips over the day')
plt.xlabel('Sessions')
plt.ylabel('Total Count of "No Car Availble" Trips')
plt.show()


# ### Observation
#  1. Maximum "No Cars Available" trips happen in "Evening" (4PM-8PM) session where the pick up is "Airport", followed by Night session (8PM-Midnight)
# 
# ### Inference
# Most number of **No Cars Available**  happen in the **Evening Hours (4PM-8PM)** in the **Airport to City** route

# ### Let's drill down further by Pick-Up Point

# In[ ]:


# Filtering out trips in the City to Airport route. Pick-up Point - City
df_citytoairport=df_uber[df_uber["Pickup_point"].str.contains('City')==True]
plt.style.use('ggplot')
colors = ["#CC2529", "#8E8D8D","#008000"]
df_citytoairport.groupby(['session','Status']).Status.count().unstack().plot.bar(legend=True, figsize=(15,10), color=colors)
plt.title('Total count of all Trip Statuses over the day for City to Airport route')
plt.xlabel('Sessions')
plt.ylabel('Total Count of Trips')
plt.show()


# ### We see that maximum trips are cancelled from City to airport in the Early Morning session followed by Late Morning Session

# In[ ]:


# Filtering out trips in the Airport to route. Pick-up Point - Airport
df_airporttocity=df_uber[df_uber["Pickup_point"].str.contains('Airport')==True]
plt.style.use('ggplot')
colors = ["#CC2529", "#8E8D8D","#008000"]
df_airporttocity.groupby(['session','Status']).Status.count().unstack().plot.bar(legend=True, figsize=(15,10), color=colors)
plt.title('Total count of all Trip Statuses over the day in the Airport to City route')
plt.xlabel('Sessions')
plt.ylabel('Total Count of Trips')
plt.show()


# ### We see that maximum number of "No Cars Available" from Airport to city in the Evening session followed by Night Session

# Lets Try to understand the **Supply** and **Demand** curve by considering all **Trips Completed** as **Supply** and **Cancelled** and **No Cars Available** as **Demand**

# In[ ]:


df_uber['supply_demand'] = ['Supply' if x == 'Trip Completed' else 'Demand' for x in df_uber['Status']]
df_uber.head()


# In[ ]:


#Plotting Supply and Demand on the City to Airport Route
df_citytoairport_supplydemand=df_uber[df_uber["Pickup_point"].str.contains('City')==True]
plt.style.use('ggplot')
df_citytoairport_supplydemand.groupby(['session','supply_demand']).supply_demand.count().unstack().plot.line(legend=True, figsize=(15,10))
plt.title('Supply-Demand curve for City to Airport Route')
plt.xlabel('Sessions')
plt.ylabel('Supply/Demand')
plt.show()


# We observe that the **Demand is very high in the morning from City to Airport route but the supply is very low**

# In[ ]:


#Plotting Supply and Demand on the Airport to City route
df_airporttocity_supplydemand=df_uber[df_uber["Pickup_point"].str.contains('Airport')==True]
plt.style.use('ggplot')
df_airporttocity_supplydemand.groupby(['session','supply_demand']).supply_demand.count().unstack().plot.line(legend=True, figsize=(15,10))
plt.title('Supply-Demand curve for Airport to City Route')
plt.xlabel('Sessions')
plt.ylabel('Supply/Demand')
plt.show()


# We observe that the **Demand is very high in the evening in the Airport to City route but the supply is very low**

# #### Correlating these two plots, a possible hypothesis could be:
# 1. The demand during morning and afternoon hours from the Airport to City is quite low, which means that a driver who completed a City to Airport route in the morning hours may have to wait a really long time to get a ride back to the City, or he may have to come back without a passenger. Both situations are not idle for drivers, which might be the reason for highest number of "Cancelled" trips in the City to Airport trip in the morning hours
# 
# 2. The vice versa happens in evening when there are too many flight coming in to the airport and hence there is a high demand for cabs and not enough supply. which is why most number of "No Cars Available" were observed in Evening in the Airport to City route.

# #### Possible Solution:
# 
# 1. Give incentives/surge pricing/bonus for trips from City to Airport during Morning hours.
# 2. Give incentives/surge pricing/bonus for trips from Airport to City during Evening hours.
# 3. Uber can give Gas expense to drivers qwhen they return from Airport without a ride or go to airpot for pick up without a ride.
# 4. Uber can increase market share by marketing campaigns and offers to customers when demand is low.

# In[ ]:




