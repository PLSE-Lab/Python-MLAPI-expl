#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})")


# In[ ]:


#this one is high def data
df = pd.read_csv("../input/train.csv", nrows=10000000)


# In[ ]:


train.rename({"acoustic_data": "acd", "time_to_failure": "ttf"}, axis="columns", inplace=True)


# Let's take a look at the general distribution of the data points, dropping 49 out of every 50 data points, 
# we can take a more general look (since my kernel keeps crashing on the complete dataset :') ). 
# 
# The quake takes place when the ttf (time to failure, green line) reaches 0. We can see these points correspond closely with the higher spikes in the acoustic data.

# In[ ]:


acd_small = train['acd'].values[::50]
ttf_small = train['ttf'].values[::50]

fig, ax1 = plt.subplots(figsize=(20, 8))

plt.plot(acd_small, color='pink')
ax2 = ax1.twinx()
plt.plot(ttf_small, color='g')


# Let's take a closer look and see how the ttf reaches zero and how this event corresponds with the acoustic data.

# In[ ]:


acd_small = train['acd'].values[::50]
ttf_small = train['ttf'].values[::50]

fig, ax1 = plt.subplots(figsize=(20, 8))

plt.plot(acd_small[:300000], color='pink')
ax2 = ax1.twinx()
plt.plot(ttf_small[:300000], color='g')


# As we can see the acoustic data peak is not where the earthquake occurs. The reason is that the acoustic** data of the moment of the earthquake is (apparently) erased from the data**, as I understand.
# To check if this holds, lets see if there are 0 ttfs in the dataset:
# 
# (Just to make sure let's check our df, which is the 1/60th of the data but ttf in float64. This range covers the first quake, but there is still no 0 val in ttf of the earthquake region. Meaning the data for the earthquake is deleted)

# In[ ]:


smallest_tvals=train[train["ttf"]==0]
print('t: ', (smallest_tvals.shape))

smallest_dvals=df[df["time_to_failure"]==0]
print('d: ', (smallest_dvals.shape))


# Number of tffs equal to 0 is 0. There is no such data. 
# 
# Let's look at the linearly decreasing ttf value closely:

# In[ ]:


acd_small = train['acd'].values[::50]
ttf_small = train['ttf'].values[::50]

fig, ax1 = plt.subplots(figsize=(20, 8))

plt.plot(acd_small[:3000], color='pink')
ax2 = ax1.twinx()
plt.plot(ttf_small[:3000], color='g')


# Closer look:

# In[ ]:


acd_small = train['acd'].values[::50]
ttf_small = train['ttf'].values[::50]

fig, ax1 = plt.subplots(figsize=(20, 8))

plt.plot(acd_small[:300], color='pink')
ax2 = ax1.twinx()
plt.plot(ttf_small[:300], color='g')


# We can see it is not linear but actually taking steps, this is caused by the recording device.
# [At the additional info given by Bertrand LD](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/77526) it is said that between every recorded sequence there is a gap much longer than the recording. These gaps correspond to the jumps we observe in the data. If it weren't for these gaps, the ttf data would be linear. See fig: (the jump is on i=4096)

# In[ ]:


df.head(4095).plot(y="time_to_failure",title="BEFORE JUMP SEGMENT VALS, 4095 DATA PTS", color = "green")
df.head(4096).tail(2).plot(y="time_to_failure",title="AFTER JUMP SEGMENT VALS, 2 DATA PTS")
print(df.head(1)["time_to_failure"])
print(df.head(4096).tail(3)["time_to_failure"])


# From the first figure we see that in one segment before jump (on the first 4095 indices [0-4094] (noting that this is not always the case since the segment length may also be 4096)), the rate at which the ttf decreases is approximately 0.000005/4095 = , while the jump is 0.000905, approximately 75420 times the normal rate. **The "silent" period is thus 75420/4095 == 18,x segments. So we do not truly have a continuous data.**

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




