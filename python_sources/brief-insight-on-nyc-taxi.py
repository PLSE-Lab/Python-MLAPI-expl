#!/usr/bin/env python
# coding: utf-8

# # High level insight on NYC Taxi
# Note: Am open to suggestions. If this helped you, some up-votes would be very much appreciated.
# 
# ### Library and Settings
# Import required library and define constants

# In[ ]:


import os
import math
import numpy as np
import pandas as pd
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.pyplot as plt


# ### Files

# In[ ]:


for f in os.listdir('../input'):
    size_bytes = round(os.path.getsize('../input/' + f)/ 1000, 2)
    size_name = ["KB", "MB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    print(f.ljust(25) + str(s).ljust(7) + size_name[i])


# ###Sneak Peak of data
# Load training and testing data. Have a quick look at columns, its shape and values

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train Data".ljust(15), train_df.shape)
print("Test Data".ljust(15), train_df.shape)


# In[ ]:


print(train_df.head())


# In[ ]:


print(test_df.head())


# Let us plot correlation matrix 

# In[ ]:


sns.set(style="white")
corr = train_df.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# The number of passengers in the vehicle is a driver entered value. Plotting these numbers will give some info about how people generally prefer to travel.

# In[ ]:


print(train_df["passenger_count"].unique())


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="passenger_count", data=train_df)
#sns.swarmplot(x="passenger_count", y="trip_duration", data=train_df, color="w", alpha=.5);


# Passenger count generally lies in between 1 to 6. The passenger count of 0, 7 ,8 and 9 are negligible in number when compared to others. 

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="passenger_count", data=train_df[train_df["passenger_count"].isin([0,7,8,9])])


# I am not sure what 0 passenger means. This is strange, maybe this value is entered by mistake or some default value. 

# In[ ]:


some = train_df[train_df["passenger_count"].isin([7,8,9])]
print(some["vendor_id"].unique())


# Only vendor 2 provides seat for passenger more than 6

# Before making some decision on passenger_count relation with vendor. Lets see how is vendor distributed along the map and how many successful rides have vendor completed

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="vendor_id", data=train_df)


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="passenger_count", hue="vendor_id", data=train_df);


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="passenger_count", hue="vendor_id", data=train_df[train_df["passenger_count"]==0]);


# In[ ]:


plt.figure(figsize=(12,8))
sns.lmplot(x="pickup_longitude", y="pickup_latitude", hue="vendor_id", data=train_df, fit_reg=False);


# To be continued...
