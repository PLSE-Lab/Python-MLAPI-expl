#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# How many different people are active?

# In[ ]:


df = pd.read_csv('../input/act_train.csv')
peoples = df['people_id']
print('Number of active people: {}'.format(peoples.nunique()))


# Which are the most active people, i.e., people with an activity count more than a given threshold?

# In[ ]:


threshold = 400
people_counts = peoples.value_counts()
people_counts[people_counts > threshold].plot(kind='bar')
plt.title('People with more than {} activities'.format(threshold))
plt.ylabel('Activity count')
fig = plt.gcf()
fig.set_size_inches(16, 7)


# That's far from uniform. We can see that there are 4 very active people. Maybe these need special attention...
# 
# On the other side, how is the distribution of the less active people?

# In[ ]:


people_counts[people_counts <= threshold].hist(bins=int(threshold / 10))
plt.xlabel('Activity count')
plt.ylabel('People count')
plt.title('Distribution of peoples with less than {} activities'.format(threshold))
fig = plt.gcf()
fig.set_size_inches(16, 7)


# We see that the vast majority of the people (more than 100k out of the 151k) performed at most 10 activities.
