#!/usr/bin/env python
# coding: utf-8

# ### Context
# 
# The aviation accident database throughout the world, from 1908-2019.
# * All civil and commercial aviation accidents of scheduled and non-scheduled passenger airliners worldwide, which resulted in a fatality (including all U.S. Part 121 and Part 135 fatal accidents)
# * All cargo, positioning, ferry and test flight fatal accidents.
# * All military transport accidents with 10 or more fatalities.
# * All commercial and military helicopter accidents with greater than 10 fatalities.
# * All civil and military airship accidents involving fatalities
# * Aviation accidents involving the death of famous people.
# * Aviation accidents or incidents of noteworthy interest.
# 
# ### Content (about the column)
# 
# * Date: Date of accident, in the format - January 01, 2001
# * Time: Local time, in 24 hr. format unless otherwise specified
# * Airline/Op: Airline or operator of the aircraft
# * Flight #: Flight number assigned by the aircraft operator
# * Route: Complete or partial route flown prior to the accident
# * AC Type: Aircraft type
# * Reg: ICAO registration of the aircraft
# * cn / ln: Construction or serial number / Line or fuselage number
# * Aboard: Total aboard (passengers / crew)
# * Fatalities: Total fatalities aboard (passengers / crew)
# * Ground: Total killed on the ground
# * Summary: Brief description of the accident and cause if known

# In[ ]:


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os, sys


# In[ ]:


dataset = pd.read_csv("../input/airplane-crash-data-since-1908/Airplane_Crashes_and_Fatalities_Since_1908_20190820105639.csv")
dataset.shape


# In[ ]:


dataset.head()


# In[ ]:


dataset.columns = dataset.columns.str.lower()


# In[ ]:


dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


operator = dataset['operator'].value_counts()
operator


# In[ ]:


locations = dataset['location'].value_counts()
locations


# In[ ]:


time = dataset['time'].value_counts()
print(time[time == 1].count())
print(time[time == 2].count())
print(time[time == 3].count())
print(time[time == 4].count())
print(time[time == 5].count())


# In[ ]:


time_count = time[time >4]

plt.figure(figsize=(10,8))
time_count.plot()
plt.ylabel("Frequency")
plt.xlabel("time count")
plt.grid(axis='y', alpha=0.75)
plt.show()


# In[ ]:


time15 = dataset[dataset['time'] == '15:00'][['time','location','operator','route', 'flight #']]
time15


# In[ ]:


time15['operator'].value_counts()


# In[ ]:


plt.figure(figsize=(12,8))
time15['operator'].value_counts().plot(kind='bar', title="Time15 Graph")
plt.show()


# In[ ]:


air_force = time15['operator'].value_counts().index.str.contains("Air Force",regex=True)
print("Air Force: {0}".format(time15['operator'].value_counts()[air_force].count()))
print("Private air company: {0}".format(time15['operator'].value_counts().count()-time15['operator'].value_counts()[air_force].count()))


# 

# ## Observation
# 
# From above we see that at 15:00 there are maximum number of plane accident which is 37 and muximum part of those accident were occured by `private air companay` and rest of different type of `air force`

# In[ ]:


time15.dropna(subset=['route'])


# In[ ]:


time15.groupby(['operator','location']).count()[['time','route']]


# In[ ]:


time17 = dataset[dataset['time'] == '17:00'][['time','location','operator','route']]
time17


# In[ ]:


time17['operator'].value_counts()
plt.figure(figsize=(12,8))
time17['operator'].value_counts().plot(kind='bar', title="Time17 Graph")
plt.show()


# In[ ]:


air_force = time17['operator'].value_counts().index.str.contains("Air Force",regex=True)
print("Air Force: {0}".format(time17['operator'].value_counts()[air_force].count()))
print("Private air company: {0}".format(time17['operator'].value_counts().count()-time17['operator'].value_counts()[air_force].count()))


# ### Some observation
# 
# I observed for time `15:00` and for `17:00`. For both time we see that maximum accident occur by private air company. So now let see for overall dataset. 

# In[ ]:


accident_data = []
accident_data_bool = []

for time in time_count.index:
    time_data = dataset[dataset['time'] == time][['time','location','operator','route']]
    time_data_freq = time_data['operator'].value_counts()
    air_force = time_data_freq.index.str.contains("Air Force",regex=True)
    air_force_count = time_data['operator'].value_counts()[air_force].count()
    private_air_count = time_data['operator'].value_counts().count()-time_data['operator'].value_counts()[air_force].count()
    acc_dic = {"time": time, "Air Force":air_force_count,"private air company": private_air_count}
    accident_data.append(acc_dic)
    if air_force_count > private_air_count:
        accident_data_bool.append("Air Force")
    else:
        accident_data_bool.append("Private Air") 


# In[ ]:


accident_data[0:10]


# In[ ]:


accident_data_bool_ser = pd.Series(accident_data_bool)
accident_data_bool_ser.value_counts().plot.bar(figsize=(10,8), rot=45)
plt.ylabel("Occurrence Count")
plt.xlabel("Air Force & Private air company")
plt.title("Frequency Graph for Private air company and Air Force")
plt.show()


# Here we see that only for one time air force accident was greater then private air company. But for maximum occurance private airplane accident was most.

# In[ ]:




