#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#check_output(["ls", "../input"]).decode("utf8").split('\n')

# Any results you write to the current directory are saved as output.


# # Exploratory data analysis
# * Load data and describes it
# 
# **REF**
# 
# [1] https://www.kaggle.com/dvasyukova/talkingdata-mobile-user-demographics/brand-and-model-based-benchmarks "Dune Dwellers"

# In[ ]:


gatrain = pd.read_csv('../input/gender_age_train.csv')
gatest = pd.read_csv('../input/gender_age_test.csv')
phone = pd.read_csv('../input/phone_brand_device_model.csv',encoding='utf-8')
#Drop duplicas found by Dune Dwellers (check REF 1)
phone = phone.drop_duplicates('device_id', keep='first')


# In[ ]:


print("Percentage of Male vs. Female")
gatrain.gender.value_counts()/len(gatrain)


# In[ ]:


gatrain.gender.value_counts().plot(kind='bar')


# In[ ]:


print("Percentage distribution by group")
gatrain.group.value_counts().sort_values()/len(gatrain)


# In[ ]:


gatrain.group.value_counts().sort_values(ascending=False).plot('bar')


# # Explore: apps

# In[ ]:


apps = pd.read_csv('../input/app_labels.csv')
labels = pd.read_csv('../input/label_categories.csv')
print("Apps:", apps.shape, 'Labels:', labels.shape)


# In[ ]:


apps_extended = apps.merge(labels, how='left', on='label_id')
print("Shape after join:" + str(apps_extended.shape))
apps_extended.head(3)


# ## Apps Categories popularities
# Plot side by side the original categories and general_groups (hand curated)
# 
# * Total categories: 473
# * Total general categories: 24
# 
# **Note** only 24 categories this seems too general. Maybe the games should be splitted e.g. chess is played by a different audience

# In[ ]:


ac = apps_extended.category.value_counts()
#TODO fill the label_categories with general groups and replace the below via general_groups
ag = apps_extended.category.value_counts()

acs = ac.cumsum()/ac.sum()
ags = ag.cumsum()/ag.sum()

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))

ax1.plot(np.arange(acs.shape[0])+1, acs.values*100)
ax1.set_xlim(0,30)
ax1.set_xlabel('Category Popularity')
ax1.set_title('% of apps');

ax2.plot(np.arange(ags.shape[0])+1, ags.values*100)
ax2.set_xlim(0,30)
ax2.set_xlabel('Category Popularity')
ax2.set_title('% of apps');

plt.show()

print("Total categories:", apps_extended.category.nunique())
#TODO replace below
#print("Total general categories:", apps_extended.general_groups.nunique())


# ## Explore Events
# 
# * All app_events has the is_installed = 1
# * 60% are inactive events and 40% are active ==> Focus on active
# * 32% of devices has events => 60% of data should be matched using the brand/model only

# In[ ]:


events = pd.read_csv('../input/events.csv')
app_events = pd.read_csv('../input/app_events.csv')


# In[ ]:


print (events.shape, "app events:", app_events.shape)


# **All apps are installed**

# In[ ]:


print("All app events has is_installed = 1")
app_events.is_installed.value_counts()


# **60% of apps are inactive**

# In[ ]:


print ("Active apps")
app_events.is_active.value_counts()*1.0/len(app_events)


# In[ ]:


active_events = app_events[app_events.is_active==1]
active_apps = active_events.merge(events, how='inner', on='event_id')

device_with_event_count = active_apps.device_id.nunique()
print("Unique device IDs with events:", device_with_event_count, 'percent with events:', device_with_event_count*1.0/phone.device_id.nunique())


# ## Explore Apps
# ## Apps popularity
# 
# * Total active apps: 19237
# * Total active apps: 10582
# * Top 3000 apps are on 99+ 

# In[ ]:


ga = active_apps.groupby('app_id')
apps_popularity = ga.device_id.nunique().sort_values(ascending=False)


# In[ ]:


ga = active_apps.groupby('app_id')
apps_popularity = ga.device_id.nunique().sort_values(ascending=False)


# In[ ]:


aps = apps_popularity.cumsum()/apps_popularity.sum()

plt.plot(np.arange(aps.shape[0])+1, aps.values*100)
plt.xlabel('Apps count')
plt.ylabel('% of devices with this app')

plt.xlim(0, 3000)

print("\t\t Devices per app")
plt.show()

print("Total apps:", app_events.app_id.nunique(), "Total active apps:", active_apps.app_id.nunique())


# ## Apps per device
# 
# * 60669 active apps on devices
# * max app on a single device is 1342
# * 75% of devices contains at most 21 apps

# In[ ]:


gd = active_apps.groupby(['device_id'])
apps_per_device = gd.app_id.nunique()


# In[ ]:


apps_per_device.describe()


# In[ ]:


print("Apps with at most 21 apps:", round(len(apps_per_device[apps_per_device<=21])*100.0/len(apps_per_device), 1))


# **Plot the distribution of apps on the 75% of the devices**

# In[ ]:


apps_per_device[apps_per_device<=21].hist(bins=22)


# **Plot the last 25% of the devices**

# In[ ]:


apps_per_device[apps_per_device>21].hist(bins=250)


# ## Categories per devices
# 
# * 75% of devices have at most 45 categories or 16 general categories
# * Max categories per device (the outliers) has 332 categories or 22 general categories

# In[ ]:


print("Devices records with all events:", active_apps.shape)
#Get the first device app
cat_devices = active_apps.groupby(['device_id', 'app_id']).first().reset_index()
print("Devices with unique apps:", cat_devices.shape)
cat_devices = cat_devices.merge(apps, how='left', on='app_id').merge(labels, how='left', on='label_id')


# In[ ]:


cat_devices[['device_id', 'app_id', 'category']].head(3)


# In[ ]:


g_by_device = cat_devices.groupby(['device_id'])
device_categories = g_by_device.category.nunique().to_frame().reset_index()


# In[ ]:


device_categories.head()


# In[ ]:


device_categories.describe()


# In[ ]:




