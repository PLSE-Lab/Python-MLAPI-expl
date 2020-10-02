#!/usr/bin/env python
# coding: utf-8

# **First look at the data**

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from collections import Counter

sns.set(style="whitegrid")
sns.set_color_codes("pastel")


# In[ ]:


columns = ['latitude', 'longitude', 'appearedLocalTime', 'appearedHour',
'appearedDay', 'city', 'temperature', 'population_density', 'class']

pcm = pd.read_csv('../input/300k.csv', low_memory=False, usecols=columns)


# 

# In[ ]:


pcm.head()


# In[ ]:


pcm.info()


# **Ok, let's see how much we have unique class of Pocemon**

# In[ ]:


uniq_cls = Counter(pcm['class'])

len(uniq_cls)


# **and here they are**

# **next, see what cities the highest activity**

# In[ ]:


cnt = pcm.groupby('city')['city'].size()

f, ax = plt.subplots(figsize=(10, 20))
sns.barplot(x=cnt.values, y=cnt.index, color='b', ax=ax)
plt.setp(ax.patches, linewidth=0)
texts = ax.set(title="Count activity by City")


# As we can see it mostly USA: 'Chicago', 'Los_Angeles', 'New_York',  and Europe: 'London',  'Rome', 'Prague'

# **ok, let's look at the activity of the cities by time**
# 
# Dataset contains information about local time in the column 'appearedLocalTime'. 
# Therefore, I first extract  local time in separate column

# In[ ]:


# extract feature local time
pcm['LocalTime'] = pcm.appearedLocalTime.apply(lambda x: x.split("T")[1])

# grouping data by time (hour)
loc_time = pcm.groupby(['city', pcm.LocalTime.map(lambda x: int(x.split(":")[0]))]).size()
loc_time = loc_time.unstack()
loc_time.fillna(0, inplace=True)

# plot grouping data
f, axes = plt.subplots(len(loc_time.columns), 1, figsize=(10, 40),sharex=True)
for i in range(len(loc_time.columns)):
	sns.barplot(x=loc_time.index, y=loc_time[i], ax=axes[i])
	axes[i].set(ylabel="Count", title="Pocemon in City activity at %2d:00 Local time" %(i))
	plt.setp(axes[i].patches, linewidth=0)
	plt.setp(axes[i].get_xticklabels(), rotation=90, fontsize=9)


# As we see in the USA ('Chicago', 'New_York',) the greatest activity at night and lowest during the period from 7-12 am. In Europe, on the contrary, the highest activity at day.

# In[ ]:


import plotly.plotly as py


# **strong text**

# In[ ]:




