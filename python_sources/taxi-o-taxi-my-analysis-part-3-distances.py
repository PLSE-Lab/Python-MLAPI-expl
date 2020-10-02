#!/usr/bin/env python
# coding: utf-8

# Hey there!!!
# 
# This is Part 3 of My Analysis on Taxi O Taxi
# 
# If you missed the first two parts here they are:
# 
# Part 1 : [Taxi O Taxi - My Analysis - Part 1][1]
# 
# Part 2 : [Taxi O Taxi - My Analysis - Part 2][2]
# 
# 
#   [1]: https://www.kaggle.com/jeru666/taxi-o-taxi-my-analysis-part-1/
#   [2]: https://www.kaggle.com/jeru666/taxi-o-taxi-my-analysis-part-2/

# In this series let's analyze the latitude and longitude variables. Both these variables are present for the pickup point and the dropoff point.

# In[8]:


import matplotlib.pyplot as plt    #--- for plotting ---
import numpy as np                 #--- linear algebra ---
import pandas as pd                #--- data processing, CSV file I/O (e.g. pd.read_csv) ---
import seaborn as sns              #--- for plotting and visualizations ---

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Input data files are available in the "../input/" directory.
path = 'D:/BACKUP/Kaggle/New York City Taxi/Data/'
train_df = pd.read_csv('../input/train.csv')

#--- Let's peek into the data
print (train_df.head())


# First and foremost we have to get the distance using the four variables(**pickup_longitude**,  **pickup_latitude**  , **dropoff_longitude** and **dropoff_latitude**). 
# 
# This is can be done using the Haversine formula. Where did I get it from? [HERE IS WHERE][1]
# 
# 
# 
# 
#   [1]: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points

# In[9]:


from math import radians, cos, sin, asin, sqrt   #--- for the mathematical operations involved in the function ---

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

train_df['Displacement (km)'] = train_df.apply(lambda x: haversine(x['pickup_longitude'], x['pickup_latitude'], x['dropoff_longitude'], x['dropoff_latitude']), axis=1)
#train_df['Haversine_dist'] = haversine(train_df['pickup_longitude'], train_df['pickup_latitude'],train_df['dropoff_longitude'], train_df['dropoff_latitude'])
print (train_df.head())


# As you can see an additional column **Haversine_dist** has been created containing the displacement between the two points.
# 
# 

# In[10]:


train_df = train_df.rename(columns = {'Displacement (km)' : 'Haversine_dist'})
#df=df.rename(columns = {'two':'new_name'})
print (train_df.head())


# Creating another column **Bearing distance** (adapted from [HERE][1])
# 
# 
#   [1]: https://www.kaggle.com/donniedarko/darktaxi-tripdurationprediction-lb-0-385/notebook

# In[11]:


def arrays_bearing(lats1, lngs1, lats2, lngs2, R=6371):
    lats1_rads = np.radians(lats1)
    lats2_rads = np.radians(lats2)
    lngs1_rads = np.radians(lngs1)
    lngs2_rads = np.radians(lngs2)
    lngs_delta_rads = np.radians(lngs2 - lngs1)
    
    y = np.sin(lngs_delta_rads) * np.cos(lats2_rads)
    x = np.cos(lats1_rads) * np.sin(lats2_rads) - np.sin(lats1_rads) * np.cos(lats2_rads) * np.cos(lngs_delta_rads)
    
    return np.degrees(np.arctan2(y, x))

train_df['bearing_dist'] = arrays_bearing(
train_df['pickup_latitude'], train_df['pickup_longitude'], 
train_df['dropoff_latitude'], train_df['dropoff_longitude'])

print (train_df.head())


# In[12]:


#--- Taken from Part 2 ---
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
train_df['dropoff_datetime'] = pd.to_datetime(train_df['dropoff_datetime'])

train_df['pickup_month'] = train_df.pickup_datetime.dt.month.astype(np.uint8)
train_df['pickup_day'] = train_df.pickup_datetime.dt.weekday.astype(np.uint8)
train_df['pickup_hour'] = train_df.pickup_datetime.dt.hour.astype(np.uint8)

train_df['dropoff_month'] = train_df.dropoff_datetime.dt.month.astype(np.uint8)
train_df['dropoff_day'] = train_df.dropoff_datetime.dt.weekday.astype(np.uint8)
train_df['dropoff_hour'] = train_df.dropoff_datetime.dt.hour.astype(np.uint8)
print (train_df.head())


# Adding another column called 'Manhattan_dist'

# In[13]:


train_df['Manhattan_dist'] =     (train_df['dropoff_longitude'] - train_df['pickup_longitude']).abs() +     (train_df['dropoff_latitude'] - train_df['pickup_latitude']).abs()
    
print(train_df.head())    


# What ranges do these distance metrics lie between?

# In[36]:



print('Range of Haversine_dist is {:f} to {:f}'.format(max(train_df['Haversine_dist']),min(train_df['Haversine_dist'])))
print('Range of Manhattan_dist is {:f} to {:f}'.format(max(train_df['Manhattan_dist']),min(train_df['Manhattan_dist'])))
print('Range of Bearing_dist is {:f} to {:f}'.format(max(train_df['bearing_dist']),min(train_df['bearing_dist'])))

  


# Is there any correlation between the three distance metric columns ?

# In[37]:


#--- get the distance metrics in a df ---
distance_df = train_df[['Haversine_dist','bearing_dist','Manhattan_dist']]
print (distance_df.corr())


# We see **high** correlation between **Haversine_dist** and **Manhattan_dist**.

# Let us see the relation that the Displacement(km) column has with respect to  **pickup_month**, **pickup_day**, **pickup_hour** and **dropoff_hour**. 
# 
# We are omitting **dropoff_month** and **dropoff_day** beacuse of the high correlation with  **pickup_month**and **pickup_day**.
# 
# ## Pick-up Month vs Displacement(km)
# 

# In[16]:


data = train_df.groupby('pickup_month').aggregate({'Haversine_dist':'sum'}).reset_index()
sns.barplot(x='pickup_month', y='Haversine_dist', data=data)
plt.title('Pick-up Month vs Haversine_dist')
plt.xlabel('Pick-up Month')
months = ['January', 'February', 'March', 'April', 'May', 'June']
plt.xticks(range(0,7), months, rotation='horizontal')
plt.ylabel('Displacement (km)')


# Months March, April and May appear to be most traveled (in terms of distance covered).
# 
# ## Pick-up Day vs Haversine_dist

# In[17]:


data = train_df.groupby('pickup_day').aggregate({'Haversine_dist':'sum'}).reset_index()
sns.barplot(x='pickup_day', y='Haversine_dist', data=data)
plt.title('Pick-up Day vs Haversine_dist')
plt.xlabel('Pick-up Month')
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.xticks(range(0,8), days, rotation='horizontal')
plt.ylabel('Displacement (km)')


# Days Thursday, Friday and Saturday seem to be the most travelled days (in terms of distance covered).
# 
# ## Pick-up Hour vs Haversine_dist

# In[18]:


data = train_df.groupby('pickup_hour').aggregate({'Haversine_dist':'sum'}).reset_index()
sns.barplot(x='pickup_hour', y='Haversine_dist', data=data)
plt.title('Pick-up Hour vs Haversine_dist')
plt.xlabel('Pick-up Hour')
#plt.xticks(range(0,8), days, rotation='horizontal')
plt.ylabel('Displacement (km)')


# Some observations:
# 
#  - Least distance covered between 2 - 5 AM 
#  - High distance covered between 6 pm - 12 AM 
#  - Moderate distance covered throughout the mornings and    early
#    evenings
# 
# ## Drop-off Hour vs Haversine_dist

# In[19]:


data = train_df.groupby('dropoff_hour').aggregate({'Haversine_dist':'sum'}).reset_index()
sns.barplot(x='dropoff_hour', y='Haversine_dist', data=data)
plt.title('Drop-off Hour vs Haversine_dist')
plt.xlabel('Drop-off Hour')
#plt.xticks(range(0,8), days, rotation='horizontal')
plt.ylabel('Displacement (km)')


# # STAY TUNED THIS PART IS YET TO BE UPDATED
# ## Please Upvote if you find it helpful and do share your comments and ways to help improve !!!!
# 
