#!/usr/bin/env python
# coding: utf-8

# ## Business Understanding
# 
# You may have some experience of travelling to and from the airport. Have you ever used Uber or any other cab service for this travel? Did you at any time face the problem of cancellation by the driver or non-availability of cars? 
# 
# Well, if these are the problems faced by customers, these very issues also impact the business of Uber. If drivers cancel the request of riders or if cars are unavailable, Uber loses out on its revenue. 
# 
# The aim of analysis is to identify the root cause of the problem (i.e. cancellation and non-availability of cars to and from the airport) and recommend ways to improve the situation. As a result of the analysis, we should be able to present to the client the root cause(s) and possible hypotheses of the problem(s) and recommend ways to improve them.  

# In[ ]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

# Import the numpy, pandas, matplotlib, seaborn packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


# ## Exploratory Data Analysis on Uber Request data

# ## Task 1: Data Cleaning
# 
# -  ### Subtask 1.1: Import and read
# 
#     - Load the Uber Request Data into a panda data frames and name it `uberReq`.
#     - Correct the Request Timestamp and Drop Timestamp datatype.

# In[ ]:


#Reading Uber Request data
uberReq = pd.read_csv('../input/Uber Request Data.csv',encoding = "ISO-8859-1")
uberReq.head()


# In[ ]:


#Correcting the data types
uberReq['Request timestamp'] = pd.to_datetime(uberReq['Request timestamp'])
uberReq['Drop timestamp'] = pd.to_datetime(uberReq['Drop timestamp'])
uberReq.head()


# In[ ]:


# Removing unnecessary columns
uberReq = uberReq.drop(['Driver id'], axis = 1)


# In[ ]:


uberReq.tail()


# -  ### Subtask 1.2: Understand the Dataset
# 
#     - How many unique pickup points are present in `uberReq`?
#     -  How many observations are present in `uberReq`?
#     - Number of null values?
#     - Inspecting the null values

# In[ ]:


#How many unique pickup points are present in uberReq?
print(uberReq['Pickup point'].unique())


# In[ ]:


#How many observations are present in uberReq?
uberReq.shape


# In[ ]:


uberReq.info()


# In[ ]:


#Inspecting the Null values , column-wise
uberReq.isnull().sum(axis=0)


# In[ ]:


uberReq[(uberReq['Drop timestamp'].isnull())].groupby('Status').size()


# #### NOTE:
# The cell above goes on to show that the `Drop timestamp` rows are empty when the `Status` is `No Cars Available` or `Cancelled`. Since the trips did not happen in those cases, the `Drop timestamp` can not be available, hence the null values here are valid.

# In[ ]:


print(len(uberReq['Request id'].unique()))
print(len(uberReq['Pickup point'].unique()))
print(len(uberReq['Status'].unique()))


# In[ ]:


#Checking if there are any duplicate values
len(uberReq[uberReq.duplicated()].index)


# ## Task 2: Univariate Analysis

# In[ ]:


#Univariate analysis on Status column 
status = pd.crosstab(index = uberReq["Status"], columns="count")     
status.plot.bar()


# #### Univariate Analysis conclusion of Status column:
# 
# `No cars available` is more than the number of trips `cancelled`. 

# In[ ]:


#Univariate analysis on Pickup Point column 
pick_point = pd.crosstab(index = uberReq["Pickup point"], columns="count")     
pick_point.plot.bar()


# #### Univariate Analysis conclusion of Pickup point column:
# 
# The pickup points `Airport` and `City` are almost equal times present in the dataset.

# ## Task 3: Bivariate Analysis

# In[ ]:


#grouping by Status and Pickup point.
uberReq.groupby(['Status', 'Pickup point']).size()


# In[ ]:


# Visualizing the count of Status and Pickup point bivariate analysis
sns.countplot(x=uberReq['Pickup point'],hue =uberReq['Status'] ,data = uberReq)


# #### Bivariate Analysis conclusion of Status and Pickup point columns:
# 
# - There are more `No cars available` from `Airport` to `City`.
# - There are more cars `Cancelled` from `City` to `Airport`.

# ## Task 4: Deriving new metrics

# In[ ]:


#Request and Drop hours
uberReq['Request Hour'] = uberReq['Request timestamp'].dt.hour


# In[ ]:


#Time Slots
uberReq['Request Time Slot'] = 'Early Morning'
uberReq.loc[uberReq['Request Hour'].between(5,8, inclusive=True),'Request Time Slot'] = 'Morning'
uberReq.loc[uberReq['Request Hour'].between(9,12, inclusive=True),'Request Time Slot'] = 'Late Morning'
uberReq.loc[uberReq['Request Hour'].between(13,16, inclusive=True),'Request Time Slot'] = 'Noon'
uberReq.loc[uberReq['Request Hour'].between(17,21, inclusive=True),'Request Time Slot'] = 'Evening'
uberReq.loc[uberReq['Request Hour'].between(21,24, inclusive=True),'Request Time Slot'] = 'Night'


# In[ ]:


#As Demand can include trips completed, cancelled or no cars available, we will create a column with 1 as a value
uberReq['Demand'] = 1


# In[ ]:


#As Supply can only be the trips completed, rest all are excluded, so we will create a column with 1 as a supply value trips completed and 0 otherwise.
uberReq['Supply'] = 0
uberReq.loc[(uberReq['Status'] == 'Trip Completed'),'Supply'] = 1


# In[ ]:


#Demand Supply Gap can be defined as a difference between Demand and Supply
uberReq['Gap'] = uberReq['Demand'] - uberReq['Supply']
uberReq.loc[uberReq['Gap']==0,'Gap'] = 'Trip Completed'
uberReq.loc[uberReq['Gap']==1,'Gap'] = 'Trip Not Completed'


# In[ ]:


#Removing unnecessary columns
uberReq = uberReq.drop(['Request Hour', 'Demand', 'Supply'], axis=1)


# In[ ]:


uberReq.head()


# ## Task 5: Graphical Analysis

# In[ ]:


# Plot to find the count of the three requests, according to the defined time slots
sns.countplot(x=uberReq['Request Time Slot'],hue =uberReq['Status'] ,data = uberReq)


# #### Conclusions from above plot :
# 
# - Most `No Cars Available` are in the `Evening`.
# - Most `Cancelled` trips are in the `Morning`.

# In[ ]:


# Plot to find the count of the status, according to both pickup point and the time slot
pickup_df = pd.DataFrame(uberReq.groupby(['Pickup point','Request Time Slot', 'Status'])['Request id'].count().unstack(fill_value=0))
pickup_df.plot.bar()


# #### Conclusions from above plot :
# 
# - Most `No Cars Available` are in the `Evening` from `Airport` to `City`.
# - Most `Cancelled` trips are in the `Morning` from `City` to `Airport`.

# In[ ]:


#Plot to count the number of requests that was completed and which was not
sns.countplot(x=uberReq['Gap'], data = uberReq)


# #### Conclusions from above plot :
# 
# More `Trip not completed` than `Trip Completed`.

# In[ ]:


##Plot to count the number of requests that was completed and which was not, against the time slot
gap_timeslot_df = pd.DataFrame(uberReq.groupby(['Request Time Slot','Gap'])['Request id'].count().unstack(fill_value=0))
gap_timeslot_df.plot.bar()


# In[ ]:


#Plot to count the number of requests that was completed and which was not, against pickup point
gap_pickup_df = pd.DataFrame(uberReq.groupby(['Pickup point','Gap'])['Request id'].count().unstack(fill_value=0))
gap_pickup_df.plot.bar()


# In[ ]:


#Plot to count the number of requests that was completed and which was not, for the final analysis
gap_main_df = pd.DataFrame(uberReq.groupby(['Request Time Slot','Pickup point','Gap'])['Request id'].count().unstack(fill_value=0))
gap_main_df.plot.bar()


# ### Hypothesis :
# 
# #### Pickup Point - City :
# 
# As per the analysis, the morning time slot is most problematic where the requests are being cancelled. Most probably the requests are being cancelled by the drivers due to the morning rush as it being the office hours and seeing the destination as airport which would be too far, the driver would think to earn more for the shorter trips within the city.
# 
# #### Pickup Point - Airport :
# 
# Upon analysis, the evening time slot seems to be most problematic for pickup points as airport where the requests being No Cars Available. The reason seems to be that not enough cars are available to service the requests as cars might not be available at the airport due to the cars serving inside the city.

# ### Conclusions :
# 
# - Based on the data analysis performed, following recommendation can be used by Uber to bridge the gap between supply and demand: -
# 
#     - For bridging the demand supply gap from airport to city, making a permanent stand in the airport itself where the cabs will be available at all times and the incomplete requests can come down significantly.
#     - Uber can provide some incentives to the driver who complete the trip from city to airport in the morning part. This might result the driver to not cancel the request from city to airport trips.
#     - Last but sure solution to bring down the gap is to increase the numbers of cab in its fleet.
