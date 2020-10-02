#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This is an excercise following Kaggle Dashboarding with Notebooks: [Day 1](https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-1/notebook).
# 
# The goal of this kernel is to take first steps towards replicating some of the functionality of the official NYPD Motor Vehicle Collisions dashboard ([link here](https://data.cityofnewyork.us/NYC-BigApps/NYPD-Motor-Vehicle-Collisions-Summary/m666-sf2m)), which includes collision information by location and date.
# 
# **__What information is changing relatively quickly (every day or hour)?__**  
# New collisions are added to the dataset every hour.
# 
# **__What information is the most important to your mission?__**  
# The mission in this case is provide the general public about the safety of a particular intersection. It imght be difficult to provide that a view at that fine granularity without using an interactive map (zooming in and out or typing in an address). Utlimately, that would be one feature a dashboard will need and therefore the exact LATITUDE and LONGITUDE of a collision will be important.
# 
# **__What will affect the choices you or others will need to make?__**  
# The total number of collision by location will inform how dangerous an interction is, as will a breakdown by the type of person either killed or injured. I also think that an understanding of factors contributing to the collision might inforn law makers and/or enforment officials of how to improve safety.
# 
# **_What changes have you made?_**  
# None

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


collisions = pd.read_csv('../input/nypd-motor-vehicle-collisions.csv')
collisions.head()


# In[ ]:


# Distribution of Collisions by Borough
collisions['BOROUGH'].value_counts().plot(kind='bar')


# In[ ]:


# Look at only Collisions where a person was either killed or injured
(collisions
    .loc[(collisions["NUMBER OF PERSONS INJURED"] > 0) | collisions["NUMBER OF PERSONS KILLED"] > 0]
    .loc[:,'BOROUGH']
    .value_counts()
    .plot(kind='bar')
)


# In[ ]:


# Deadliest collisions

(collisions
     .iloc[collisions.groupby("BOROUGH")["NUMBER OF PERSONS KILLED"].idxmax()]
     .loc[:,['DATE', 'BOROUGH']]
     .sort_values(by='DATE', ascending=False)
)


# In[ ]:


# Distribution of Collisions by Date
# Time of day not recorded

collisions['DATE'] = pd.to_datetime(collisions['DATE'], format="%Y-%m-%d")
collisions['DATE'].value_counts().resample('m').sum().plot.line()


# In[ ]:


# Causes

