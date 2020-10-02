#!/usr/bin/env python
# coding: utf-8

# Let's observe this data

# In[9]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.lines as mlines
import pandas as pd
import seaborn as sns
import os


# In[10]:


train = pd.read_csv("../input/flight_delays_train.csv")


# In[11]:


def preprocess(X):
    X["Flight"] = X["Origin"] + "-" + X["Dest"]
    X["Hour"] = X["DepTime"] // 100
    X["Month"] = X["Month"].apply(lambda x: x.replace("c-", ""))
    X["DayOfMonth"] = X["DayofMonth"].apply(lambda x: x.replace("c-", ""))
    X = X.drop(["DayofMonth"], axis=1)
    X["DayOfWeek"] = X["DayOfWeek"].apply(lambda x: x.replace("c-", ""))

    if "dep_delayed_15min" in X.columns:
        X["dep_delayed_15min"] = X["dep_delayed_15min"].map({"Y": 1, "N":0})
    
    return X


# In[12]:


X = preprocess(train.copy())
X.head()


# # Graph charts of delays grouped by various features

# In[13]:


weekdays = [u"Mon", u"Tue", u"Wed", u"Thu", u"Fr", u"Sat", u"Sun"]
months = [u"Jan", u"Feb", u"Mar", u"Apr", u"May", u"Jun", u"Jul", u"Aug", u"Sep", u"Oct", u"Nov", u"Dec"]


# In[14]:


plt.figure(figsize=(16,8))
ax0 = plt.subplot(212)
ax1 = ax0.twinx()
line, = ax0.plot(range(1,8), X.groupby("DayOfWeek").agg({"dep_delayed_15min": np.mean}), 'k:', color='b')
line.set_label("average delayed by weekday")
ax0.legend(bbox_to_anchor=(.17, -0.13), frameon = False)

line2, = ax1.plot(range(1,8), list(X["DayOfWeek"].value_counts().sort_index()), color='m')
line2.set_label("total flights by weekday")
ax1.legend(bbox_to_anchor=(.14, -0.22), frameon = False)

ax0.text(4, 0.155, 'Day of week', ha='center')
ax0.text(0.3, 0.195, 'avg delay', va='center', rotation='vertical')
ax0.text(7.74, 0.195, 'total count of flight', va='center', rotation='vertical')
plt.title(u"Graph of delay grouped by weekdays")
plt.xticks(range(1,8), weekdays)
plt.show()


# In[15]:


plt.figure(figsize=(16,8))
ax0 = plt.subplot(212)
ax1 = ax0.twinx()
line, = ax0.plot(range(1,32), X.groupby("DayOfMonth").agg({"dep_delayed_15min": np.mean}), 'k:', color='b')
line.set_label("average delayed by day of month")
ax0.legend(bbox_to_anchor=(.17, -0.13), frameon = False)

line2, = ax1.plot(range(1,32), list(X["DayOfMonth"].value_counts().sort_index()), color='m')
line2.set_label("total flights by day of month")
ax1.legend(bbox_to_anchor=(.14, -0.22), frameon = False)

ax0.text(17, 0.15, 'Day of month', ha='center')
ax0.text(-2.5, 0.195, 'avg delay', va='center', rotation='vertical')
ax0.text(34, 0.195, 'total count of flight', va='center', rotation='vertical')
plt.title(u"Graph of delay grouped by day of month")
plt.show()


# In[16]:


plt.figure(figsize=(16,8))
ax0 = plt.subplot(212)
ax1 = ax0.twinx()
line, = ax0.plot(range(1,27), X.groupby("Hour").agg({"dep_delayed_15min": np.mean}), 'k:', color='b')
line.set_label("average delayed by hour")
ax0.legend(bbox_to_anchor=(.17, -0.13), frameon = False)

line2, = ax1.plot(range(1,27), list(X["Hour"].value_counts().sort_index()), color='m')
line2.set_label("total flights by hour")
ax1.legend(bbox_to_anchor=(.14, -0.22), frameon = False)

ax0.text(12, -0.2, 'hour', ha='center')
ax0.text(-1.5, 0.5, 'avg delay', va='center', rotation='vertical')
ax0.text(29, 0.5, 'total count of flight', va='center', rotation='vertical')
plt.title(u"Graph of delay grouped by hour")
plt.show()


# In[17]:


plt.figure(figsize=(16,8))
ax0 = plt.subplot(212)
ax1 = ax0.twinx()
line, = ax0.plot(range(1,13), X.groupby("Month").agg({"dep_delayed_15min": np.mean}), 'k:', color='b')
line.set_label("average delayed by month")
ax0.legend(bbox_to_anchor=(.17, -0.13), frameon = False)

line2, = ax1.plot(range(1,13), list(X["Month"].value_counts().sort_index()), color='m')
line2.set_label("total flights by month")
ax1.legend(bbox_to_anchor=(.14, -0.22), frameon = False)

ax0.text(7, 0.135, 'Month', ha='center')
ax0.text(-0.3, 0.195, 'avg delay', va='center', rotation='vertical')
ax0.text(13.2, 0.195, 'total count of flight', va='center', rotation='vertical')
plt.title(u"Graph of delay grouped by month")
plt.xticks(range(1,13), months)
plt.show()


# In[18]:


plt.figure(figsize=(16,4))
plt.hist(X['Distance'][X['dep_delayed_15min'] == 0], bins=100, label="Not delayed")
plt.hist(X['Distance'][X['dep_delayed_15min'] == 1], bins=100, label="Delayed")
plt.title(u"Dependency delay on flight distance")
plt.xlabel(u"Flight distance")
plt.ylabel(u"Flights count")
plt.legend()
plt.show()


# # Check data for correctness

# In[19]:


# Check distances between airports are the same for all flights
Z = X.groupby("Flight").agg({"Distance": [np.max, np.min]})
np.sum(Z["Distance"]["amax"] - Z["Distance"]["amin"]) == 0


# In[20]:


# Count of unique origin and destination airports
len(X["Dest"].unique()), len(X["Origin"].unique())


# In[21]:


# Count of unique flight and distances
len(np.unique(X['Flight'])), len(np.unique(X['Distance']))


# In[ ]:




