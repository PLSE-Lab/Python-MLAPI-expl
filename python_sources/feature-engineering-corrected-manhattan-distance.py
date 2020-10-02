#!/usr/bin/env python
# coding: utf-8

# **Calculating " Corrected Manhattan Distance" (Approximating actual distance travelled)**
# 
# In a city block it may be more accurate to estimate total distance travelled by using the "Manhattan distance", rather than  the straight line distance between two points (Euclidean distance). 
# 
# https://en.wikipedia.org/wiki/Taxicab_geometry
# 
# The difficult part here is that although Manhattan island is arranged in blocks, it is tilted at approximately 29 degrees clockwise from north.  - Therefore we need to calculate a corrected distance!
# 
# <img src="https://i.imgur.com/ZPlvFzO.png" />

# **Overview of Steps**
# 
# The steps i'm going to take next are:
# 1. - Calculate the difference in degrees longitude and latitude between pickup & dropoff.
# 2. - Translate these values into distances (miles)
# 3. - Use trigonometry to calculate our corrected Manhattan distance

# In[ ]:


# Loading in our sample data.
import numpy as np # 
import pandas as pd # 
import os
train_df =  pd.read_csv('../input/train.csv', nrows = 1_000_000)


# **Step 1 - Calculate the difference in degrees longitude and latitude between pickup & dropoff.** 
# 
# 

# In[ ]:


# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the vector from
# the pickup location to the dropoff location.
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(train_df)


# Removing NaNs for good measure.

# In[ ]:


print('Old size: %d' % len(train_df))
train_df = train_df.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(train_df))


# **Step 2 - Translate these values into distances (miles)**
# 
# 
# 1 degree of Latitude is equal to 69 miles.
# 
# Longitude vs miles varies depending where we are in the world.
# 
# At the latitude of Manhattan, 1 degree of longitude is equal to approximately 50 miles.
# 

# In[ ]:


### Converting abs_diff_longitude & lattitude to miles...
# Since we are calculating this at New York, we can assign a constant, rather than using a formula
# longitude = degrees of latitude in radians * 69.172
#1 degree of longitude = 50 miles
def convert_miles(train_df):
    train_df['abs_diff_longitude'] = train_df.abs_diff_longitude*50
    train_df['abs_diff_latitude'] = train_df.abs_diff_latitude*69
convert_miles(train_df)


# **Step 3 - Use trigonometry to correct our distances**
# 
# 
# 
# 

# In[ ]:


### Angle difference between north, and manhattan roadways
meas_ang = 0.506 # 29 degrees = 0.506 radians
import math

##This could be dealt with via a bounding box...

## adding extra features
def add_extra_manh_features(df):
    df['Euclidean'] = (df.abs_diff_latitude**2 + df.abs_diff_longitude**2)**0.5 ### as the crow flies  
    df['delta_manh_long'] = (df.Euclidean*np.sin(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude)-meas_ang)).abs()
    df['delta_manh_lat'] = (df.Euclidean*np.cos(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude)-meas_ang)).abs()
    df['manh_length'] = df.delta_manh_long + df.delta_manh_lat
    df['Euc_error'] = (df['manh_length'] - df['Euclidean'])*100 /  df['Euclidean']

add_extra_manh_features(train_df)


# ** Step 4 (Next steps & Calculating the impact)**
# 
# Firstly, we want to limit the scope of these measurements to journeys staying within Manhattan.

# In[ ]:


# this is a rough way of achieving this, but isn't perfect due to the island's orientation
def manh_checker(x):
    if  40.7091 < x['dropoff_latitude'] < 40.8205 and     -74.0096 < x['dropoff_longitude'] < -73.9307 and     40.7091 < x['pickup_latitude'] < 40.8205 and     -74.0096 < x['pickup_longitude'] < -73.9307:
        return 1
    else:
        return 0
    
train_df['manh_island'] = train_df.apply(manh_checker, axis = 1) 


# In[ ]:


print ( train_df['manh_island'].sum()*100 / len(train_df))


# So 79% of all journeys start and end in (or slighlty around) Manhattan island..

# In[ ]:


# masking our dataframe to manhattan island

mask = train_df['manh_island'] == 1
train_df = train_df[mask]


# In[ ]:


### now plot ERROR vs Euclidean
import seaborn as sns; sns.set(style="white", color_codes=True)
import matplotlib.pyplot as plt

g = sns.jointplot(x= train_df['Euclidean'], y= train_df['Euc_error']).set_axis_labels("Euclidean distance (miles)", "Percentage error (vs Manhattan distance)")



# So from the plot above we can see the percentage error between our "true"/Manhattan distance, and the Euclidean distance.
# 
# Finally, lets have a look at how our corrected Manhattan distance compares to uncorrected.

# In[ ]:




train_df['old_manh'] = train_df['abs_diff_latitude'] + train_df['abs_diff_longitude']
train_df['old_manh_error'] = (train_df['manh_length'] - train_df['old_manh'])*100 /  train_df['old_manh']

g = sns.jointplot(x= train_df['old_manh'], y= train_df['old_manh_error']).set_axis_labels("Old Manhattan distance (miles)", "Percentage error (vs new Manhattan distance)")

