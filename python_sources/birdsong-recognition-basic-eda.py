#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import plotly.express as px


# First let's look into the training metadata.

# In[ ]:


train = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')
train.head()


# The quality of a recording is depicted in a rating from 0 to 5.

# In[ ]:


ratings = train['rating'].value_counts().to_frame()
ratings.rename(columns={'rating':'Instances'}, inplace=True)
ratings['Rating'] = ratings.index
fig = px.bar(ratings, x='Rating', y='Instances',
            labels={'Instances':'Instances in train'} )

fig.show()


# The bulk of the data has a rating >3. It might be an idea to leave out recordings with rating <3 when training a model.

# In[ ]:


species = train['species'].value_counts().to_frame()
fig = px.bar(species, x=species.index, y='species',
            labels={'species:Species of birds'})
fig.show()


# The amount of recordings per bird species show a bias is present. Roughly a third of the data set has got significantly less recordings available in the training data set.

# In[ ]:


channels = train['channels'].value_counts().to_frame()
channels.rename(columns={'channels':'Instances'}, inplace=True)
channels['Channel'] = channels.index
channels.reset_index(drop=True, inplace=True)
channels.head()


# The recordings made with a mono or stereo channel are roughly 50/50. Maybe a different model for each of the channels is needed to make accurate predictions.

# In[ ]:


train['month'] = train.date.str[5:7].astype(int)
months = train.month.value_counts().to_frame()
months.rename(columns={'month':'Instances'}, inplace=True)
months['Month'] = months.index
months.reset_index(drop=True, inplace=True)
months.head()


# In[ ]:


fig = px.bar(months, x='Month', y='Instances')

fig.update_layout(
    title="Distribution of bird calls recorded over the months"
)
fig.show()


# In[ ]:


look_up = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
            6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec', 0 : '13th'}
months.Month = months.Month.apply(lambda x: look_up[x])
fig = px.bar(months, x='Month', y='Instances')
fig.update_layout(
    title="Distribution of bird calls recorded over the months ordered by instances"
)
fig.show()


# Apparently recording birds is more fun in summer. Also 34 birds have been recorded in the 13th month. In all seriousness the data might be biased to species of birds which are active in summer.

# In[ ]:


pitches = train['pitch'].value_counts().to_frame()
pitches.rename(columns={'pitch':'Instances'}, inplace=True)
pitches['pitch'] = pitches.index
pitches.reset_index(drop=True, inplace=True)
pitches.head()


# In[ ]:


fig = px.bar(pitches, x='pitch', y='Instances')
fig.update_layout(
    title="Distribution of bird call pitch",
    xaxis_title="Pitch"
)
fig.show()


# In[ ]:


fig = px.box(train, y='duration')
fig.update_layout(
    title="Duration of recordings",
    yaxis_title="Duration (seconds)"
)
fig.show()


# Why are some of these recording so long? 

# In[ ]:


times = train.time.value_counts().to_frame()
times.rename(columns={'time':'Instances'}, inplace=True)
times['time'] = times.index
times.reset_index(drop=True, inplace=True)


# In[ ]:


fig = px.bar(times, x='time', y='Instances')
fig.update_layout(
    title="Distribution of bird calls per time stamp",
    xaxis_title="Pitch"
)
fig.show()


# Time data needs some cleaning up... and it seems a lot of bird recordings are made either at 08:00 or 20:00.
