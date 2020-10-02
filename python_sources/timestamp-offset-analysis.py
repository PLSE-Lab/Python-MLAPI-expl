#!/usr/bin/env python
# coding: utf-8

# # Timestamp analysis
# I also want to take a look at the timestamp. I especially want to focus on finding a good offset to organize the timestamp in days/weeks since there are already a few scripts analyzing the general structure of timestamps.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

fb_train = pd.read_csv('../input/train.csv')
fb_test = pd.read_csv('../input/test.csv')

# Any results you write to the current directory are saved as output.


# In[ ]:


fb_train.describe()


# In[ ]:


common_places = fb_train.place_id.value_counts()[0:30].index
train_common = fb_train[fb_train.place_id.isin(common_places)]
train_common.shape


# In[ ]:


train_common.describe()


# These are the offset values I empirically chose. I first examined the offset for each day and afterwards the weekly offset. I will explain later why I think that these offsets are reasonable.

# In[ ]:


day_offset = 9*60
week_offset = 60*24*3
complete_offset = day_offset + week_offset

number_of_places_to_display = 12


# Let's plot a daily aggregate of all check-ins for the most common places and see if my chosen offset is reasonable:

# In[ ]:


plt.figure(figsize=(18,21))
plt.title('Daily check-ins')
for i in range(0,number_of_places_to_display):
    ax = plt.subplot(number_of_places_to_display/3,3,i+1)
    ax.set_xlim([0, 60*24])
    sns.distplot((train_common[train_common.place_id==common_places[i]].time + complete_offset)%(60*24), bins=100)
    plt.xlabel("Time")
    plt.ylabel("Fraction")
    plt.xticks(np.arange(0, 60*24, 60))
    labels = [i for i in range(0,25)]
    ax = plt.gca()
    ax.set_xticklabels(labels)
    plt.title("pid: " + str(common_places[i]))


# Now do the same with a weekly aggregate for each place...

# In[ ]:


plt.figure(figsize=(18,21))
plt.title('Weekly check-ins')
for i in range(0,number_of_places_to_display):
    ax = plt.subplot(number_of_places_to_display/3,3,i+1)
    ax.set_xlim([0, 60*24*7])
    sns.distplot((train_common[train_common.place_id==common_places[i]].time + complete_offset)%(60*24*7), bins=100)
    plt.xlabel("Time")
    plt.ylabel("Fraction")
    plt.gca().get_xaxis().set_ticks([])
    plt.title("pid: " + str(common_places[i]))


# Now let's have a look at a daily and weekly aggregate of all common places and let's see if my chosen offset is reasonable:

# In[ ]:


plt.figure(figsize=(18,9))

plt.subplot(211)
sns.distplot((train_common.time + complete_offset)%(60*24), bins=100)
plt.xticks(np.arange(0, 60*24, 60))
labels = [i for i in range(0,25)]
ax = plt.gca()
ax.set_xticklabels(labels)

plt.xlabel("Time in h")
plt.ylabel("Fraction")
plt.title("Daily check-ins of all common places")

plt.subplot(212)
sns.distplot((train_common.time + complete_offset)%(60*24*7), bins=100)
plt.xticks(np.arange(0, 60*24*7, int(60*24/4)))
labels = [i*6%24 for i in range(0,int(7*24/4+1))]
ax = plt.gca()
ax.set_xticklabels(labels)

plt.xlabel("Time in h")
plt.ylabel("Fraction")
plt.title("Weekly check-ins of all common places")


# ## Daily aggregate
# There is a small peak at 8:00 for the daily aggregate which could indicate breakfast check-ins. There is also a small peak at 12:00-13:00 which could indicate lunch check-ins. The same appears to be true for dinner between 16:00 and 19:00. There are less check-ins at night between 21:00 and 7:00 because most people probably are asleep/at home then.
# 
# ## Weekly aggregate
# The plot for the daily aggregate starts at what I think is monday. The first four days are pretty similar. They have a small peak at breakfast and dinner hour each and they have less check-ins than the other days. On thursday is a high peak at dinner hour which I can't really explain. The same is the case on friday. I think the last two days are saturday and sunday because the check-ins are more uniformly distributed over the whole day and there are also more check-ins during these days. This could be the case because people don't have to work on a weekend. 

# Now let's have a look at a historgram with bins for each hour/day:

# In[ ]:


plt.figure(figsize=(18,9))

plt.subplot(211)
sns.distplot((train_common.time + complete_offset)%(60*24), bins=24)
plt.xlabel("Time in h")
plt.ylabel("Fraction")
plt.title("Daily check-ins of all common places")
plt.xticks(np.arange(0, 60*24, 60))
labels = [i for i in range(0,25)]
ax = plt.gca()
ax.set_xticklabels(labels)

plt.subplot(212)
sns.distplot((train_common.time + complete_offset)%(60*24*7), bins=7)
plt.xticks(np.arange(0, 60*24*7, 60*24))
labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
ax = plt.gca()
ax.set_xticklabels(labels)
plt.xlabel("Weekday")
plt.ylabel("Fraction")
plt.title("Weekly check-ins of all common places")


# The daily plot still shows peaks at breakfast, lunch and dinner. The weekly plot shows more check-ins at the weekend than during weekdays. 

# Finally, let's add new columns for the day of week and the hour of the day in our dataset:

# In[ ]:


fb_train['day'] = (fb_train.time + complete_offset)  % (60*24*7) / (60*24)
fb_train['day'] = fb_train['day'].apply(np.floor)

fb_train['hour'] = (fb_train.time + complete_offset)  % (60*24) / 60
fb_train['hour'] = fb_train['hour'].apply(np.floor)


# In[ ]:


fb_train.describe()


# Now let's plot our new classification for day/hour of each entry:

# In[ ]:


plt.figure(figsize=(18,4))
ax = plt.subplot()
ax.set_xlim([-1, 7])
sns.distplot(fb_train.day, kde=False)
plt.xticks(np.arange(0, 7, 1))
labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
ax = plt.gca()
ax.set_xticklabels(labels)
plt.xlabel("Weekday")
plt.ylabel("Fraction")
plt.title("Weekly check-ins of all places")


# The daily check-ins seem to be uniformly distributed with slightly more check-ins on Saturday and Sunday. 

# In[ ]:


plt.figure(figsize=(18,4))
ax = plt.subplot()
ax.set_xlim([-1, 24])
sns.distplot(fb_train.hour, kde=False)
plt.xlabel("Time in h")
plt.ylabel("Fraction")
plt.title("Daily check-ins of all places")
plt.xticks(np.arange(0, 25, 1))
labels = [i for i in range(0,25)]
ax = plt.gca()
ax.set_xticklabels(labels)


# The hourly check-ins of all places seem to be almost uniformly distributed.

# ## Conclusion
# I think I found a good offset for the timestamp and even if it's not 100% correct it shouldn't have a big impact on classification accuracy.
# 
# I hope that I could help you with my script. It's my first script on kaggle so please don't be too hard on me :-)
