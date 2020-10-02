#!/usr/bin/env python
# coding: utf-8

# ## Uber Supply Demand Gap

# ![](https://gazettereview.com/wp-content/uploads/2015/10/Uber-Logo.jpg)* 

# ![](https://i2-prod.mylondon.news/article16106961.ece/ALTERNATES/s615/0_Uber-pink-cars.jpg)

# ## Business Understanding
# 
# We may have some experience of travelling to and from the airport. Have you ever used Uber or any other cab service for this travel? Did you at any time face the problem of cancellation by the driver or non-availability of cars?
# 
#  Well, if these are the problems faced by customers, these very issues also impact the business of Uber. If drivers cancel the request of riders or if cars are unavailable, Uber loses out on its revenue.

# ## Business Objectives
# 
# The aim of analysis is to identify the root cause of the problem (i.e. cancellation and non-availability of cars) and recommend ways to improve the situation. As a result of our analysis, we should be able to present to the client the root cause(s) and possible hypotheses of the problem(s) and recommend ways to improve them

# ## Data Understanding
# There are six attributes associated with each request made by a customer:
# 
# - Request id: A unique identifier of the request<br>
# - Time of request: The date and time at which the customer made the trip request<br>
# - Drop-off time: The drop-off date and time, in case the trip was completed<br>
# - Pick-up point: The point from which the request was made<br>
# - Driver id: The unique identification number of the driver<br>
# - Status of the request: The final status of the trip, that can be either completed, cancelled by the driver or no cars available<br>
# 
# **Note: In the analysis, only the trips to and from the airport are being considered.**

# #### This kernel is based on the assignment by IIITB collaborated with upgrad.

# #### If this Kernel helped you in any way, some <font color="red"><b>UPVOTES</b></font> would be very much appreciated

# ## Step 1: Reading and Understanding the Data

# In[ ]:


# import all libraries and dependencies for dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta

# import all libraries and dependencies for data visualization
pd.options.display.float_format='{:.4f}'.format
plt.rcParams['figure.figsize'] = [8,8]
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', -1) 
sns.set(style='darkgrid')
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker


# In[ ]:


# Reading the Uber file

path = '../input/uber-supplydemand-gap/'
file = path + 'Uber Request Data.csv'
uber = pd.read_csv(file)


# In[ ]:


uber.head()


# In[ ]:


# Dimensions of df

uber.shape


# In[ ]:


# Data description

uber.describe()


# In[ ]:


# Data info

uber.info()


# ## Step 2: Data Cleansing and Preparation

# In[ ]:


# check if any duplicates record exists

sum(uber.duplicated(subset = "Request id")) == 0


# In[ ]:


# Calculating the Missing Values % contribution in DF

df_null = uber.isna().mean().round(4)*100

df_null.sort_values(ascending=False)


# #### Inference:
# - More than 50% of `Drop time` is null because of Cancelled Trips which makes sense.
# - Around 40% of `Driver Id` is null because of Cancelled Trips which makes sense.

# In[ ]:


# Check datatypes of df

uber.dtypes


# - We need to change the datatype of `Request timestamp` and `Drop timestamp` to datetime

# In[ ]:


# Converting the datatype of Request timestamp and Drop timestamp

uber['Request timestamp'] = uber['Request timestamp'].astype(str)
uber['Request timestamp'] = uber['Request timestamp'].str.replace('/','-')
uber['Request timestamp'] = pd.to_datetime(uber['Request timestamp'], dayfirst=True)


# In[ ]:


uber['Drop timestamp'] = uber['Drop timestamp'].astype(str)
uber['Drop timestamp'] = uber['Drop timestamp'].str.replace('/','-')
uber['Drop timestamp'] = pd.to_datetime(uber['Drop timestamp'], dayfirst=True)


# In[ ]:


# Extract the hour from the request timestamp

req_hr = uber['Request timestamp'].dt.hour
req_hr.value_counts()
uber['Req hour'] = req_hr


# In[ ]:


# Extract the day from request timestamp

req_day = uber['Request timestamp'].dt.day
req_day.value_counts()
uber['Req day'] = req_day


# ## Step 3: Data Visualization

# In[ ]:


# Factor plot of hour and day with respect to Status

sns.factorplot(x = 'Req hour', hue = 'Status', row = 'Req day', data = uber, kind = 'count', size=5, aspect=3)


# #### Inference:
# - No Cars available sistuation occurs primarly at evening hours from 5PM to 10 PM.
# - Frequent Cancellations were encountered in morning hours.

# In[ ]:


# Factor plot of hour and day with respect to Pickup Point

sns.factorplot(x = 'Req hour', hue = 'Pickup point', row = 'Req day', data = uber, kind = 'count', size=5, aspect=3)


# #### Inference:
# - Most of the pickups encountered at daytime is from city suggesting more people travel to the airport in day hours.
# - The pickups from Airport at evening hours are more and it suggests most people land in evening hours. 

# In[ ]:


# Aggregate count plot for all days w.r.t. to Pickup point

sns.factorplot(x = 'Req hour', hue = 'Pickup point', data = uber, kind = 'count', size=5, aspect=3)


# In[ ]:


# Creating timeslots for various time period of the day

time_hour = [0,5,10,17,22,24]
time_slots =['Early Morning','Morning_Rush','Daytime','Evening_Rush','Late_Night']
uber['Time_slot'] = pd.cut(uber['Req hour'], bins = time_hour, labels = time_slots)


# In[ ]:


# Visualizing the different time slots wrt status

plt.rcParams['figure.figsize'] = [12,8]
sns.countplot(x = 'Time_slot', hue = 'Status', data = uber)
plt.xlabel("Time Slots",fontweight = 'bold')
plt.ylabel("Number of occurence ",fontweight = 'bold')


# #### Inference:
# - Cars not available situation arises mostly in evening hours.
# - Most of the Cancellation happens in morning hours.
# 

# In[ ]:


# as we can see in the above plot the higest number of cancellations are in the "Morning Rush" time slot
morning_rush = uber[uber['Time_slot'] == 'Morning_Rush']
sns.countplot(x = 'Pickup point', hue = 'Status', data = morning_rush)


# #### Inference:
# - The Cancellation situation is a problem for the trip from City to airport in  morning hours.

# In[ ]:


# as we can see in the above plot the higest number of no cars available are in the "Evening Rush" time slot
evening_rush = uber[uber['Time_slot'] == 'Evening_Rush']
sns.countplot(x = 'Pickup point', hue = 'Status', data = evening_rush)


# #### Inference:
# - The No cars available situation is a problem for the trip from airport to city in evening hours.

# In[ ]:


# Let's create pie charts instead of a count plots
def pie_chart(dataframe):
    
    labels = dataframe.index.values
    sizes = dataframe['Status'].values
        
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


# In[ ]:


# percentage breakup of status on the basis of pickup location
# Status of trips @ Morning Rush where pickup point is City
city = uber.loc[(uber["Pickup point"] == "City") & (uber.Time_slot == "Morning_Rush")]
city_count = pd.DataFrame(city.Status.value_counts())
pie_chart(city_count)


# In[ ]:


# percentage breakup of status on the basis of pickup location
# Status of trips @ Evening Rush where pickup point is City
city = uber.loc[(uber["Pickup point"] == "City") & (uber.Time_slot == "Evening_Rush")]
city_count = pd.DataFrame(city.Status.value_counts())
pie_chart(city_count)


# In[ ]:


# percentage breakup of status on the basis of pickup location
# Status of trips @ Morning Rush where pickup point is Airport
city = uber.loc[(uber["Pickup point"] == "Airport") & (uber.Time_slot == "Morning_Rush")]
city_count = pd.DataFrame(city.Status.value_counts())
pie_chart(city_count)


# In[ ]:


# percentage breakup of status on the basis of pickup location
# Status of trips @ Evening Rush where pickup point is Airport
city = uber.loc[(uber["Pickup point"] == "Airport") & (uber.Time_slot == "Evening_Rush")]
city_count = pd.DataFrame(city.Status.value_counts())
pie_chart(city_count)


# #### We have analyized and drawn various insights of the trips.

# ### If this Kernel helped you in any way, some <font color="red"><b>UPVOTES</b></font> would be very much appreciated
