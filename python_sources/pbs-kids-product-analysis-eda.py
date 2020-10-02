#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import json
import seaborn as sns
import datetime
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
import warnings
import random
import plotly.express as px
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
my_pal = sns.color_palette(n_colors=10)


# In[ ]:


plt.figure(figsize=(13,13))
image = plt.imread("/kaggle/input/pbs-images/IMAGE 2019-10-25 160313.jpg");
plt.imshow(image);
plt.axis('off');


# Dataset is big, so import firstly from directory only 1M rows.

# In[ ]:


n = 11341042 #number of records in file
s = 1000000 #desired sample size
filename = '../input/data-science-bowl-2019/train.csv'
skip = sorted(random.sample(range(n),n-s))
train_sam = pd.read_csv(filename, skiprows=skip)
train_sam.columns = ['event_id','game_session','timestamp','event_data',
            'installation_id','event_count','event_code','game_time','title','type','world']
specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')


# In[ ]:


train_sam.head(3)


# - In this dataset user id is installation_id column. But we must remember that user can have several devices with different or same accounts, user can drop application and reinstall app and so on. I hope that dataset contains only real users without test developer accounts)
# - It's very important to understand "session" concept. Different analytical systems understand the concept of a session differently. Suppose classic, that session is end when user exited the application or was idle for 30 minutes (This is Google Analytics meaning). 
# 

# ## Sessions distribution
# Plot classic retention histogram:

# In[ ]:


tips = train_sam[['game_session','installation_id']].drop_duplicates().groupby(['installation_id']).count().reset_index()
fig = px.histogram(tips[tips['game_session'] < 100], x="game_session", title='Sessions by user')
fig.show()


# This histogram is very interesting:
# - Here we can see typical gap between histogram columns - approximately 2/3 of the users leave the application at the very beginning. 
# - We can see that after approximately 7 sessions users stop falling off the application - gap between columns noticeably reduced.

# Here we can see that user should make more than 2-3 events to get involved in the application. Interesting maximum in the region of 25-35 events. It may be necessary to complete some part of the game, and this requires 25-35 events.

# In[ ]:


tips = train_sam[['game_session','event_count']].groupby(['game_session']).max().reset_index()
fig = px.histogram(tips[tips['event_count'] < 100], x="event_count", title='Event count by session')
fig.show()

tips = train_sam[['game_session','event_count']].groupby(['game_session']).max().reset_index()
fig = px.histogram(tips[(tips['event_count'] > 1) & (tips['event_count'] < 200)], x="event_count", 
                   title='Event count by session', nbins=40)
fig.show()


# Transform columns timestamp to date:

# In[ ]:


train_sam['date'] = train_sam['timestamp'].apply(lambda x: x.split('T')[0])
train_sam['date'] = train_sam['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))


# I want to see apps activity plot by dates. The number of actions increases until the end of September, and then falls. Perhaps I have imported insufficient data. Perhaps this is due to the fact that the active using of applications is associated with the start of studies after the summer (even for kids).

# In[ ]:


ss = train_sam[['date','installation_id']].groupby(['date']).count().reset_index()
ss.columns = ['date', 'activity_count']
fig = px.line(ss, x='date', y='activity_count', title='Activity')
fig.show()


# This is installs plot. Here we can see cyclic structure - at least on weekends and maximum installations on Fridays. This application is educational, therefore such behavior is logical.

# In[ ]:


temp = train_sam[['date','installation_id']].groupby(['installation_id']).min().reset_index()
ss = temp.groupby(['date']).count().reset_index()
ss.columns = ['date', 'installs_count']
fig = px.line(ss, x='date', y='installs_count', title='Installs')
fig.show()


# In[ ]:


fig = px.bar(x=pd.value_counts(train_sam['world']).index, y=pd.value_counts(train_sam['world']).values)
fig.show()


# Facetted subplots about **event_count** with two parameters - type and world: <br>
# **type** - Media type of the game or video. Values: Game, Assessment, Activity, Clip.<br>
# * **world** - The section of the application the game or video belongs to. Helpful to identify the educational curriculum goals of the media. Values: NONE, TREETOPCITY(Length/Height), MAGMAPEAK (Capacity/Displacement), CRYSTALCAVES (Weight).

# In[ ]:


tips = train_sam[['game_session','type','event_count','world']].groupby(['type','world','game_session']).max().reset_index()
fig = px.histogram(tips[(tips['event_count']>1) & (tips['event_count']<100)], x="event_count", 
             facet_row="world", facet_col="type", nbins=40,
             category_orders={"world": ['CRYSTALCAVES', 'MAGMAPEAK', 'NONE', 'TREETOPCITY'],
                             'type': ['Activity', 'Game', 'Clip', 'Assessment']})
fig.show()


# ### Popular events

# In[ ]:


event = pd.value_counts(train_sam.event_id)
event[event>17000].plot('barh', title='Popular events');


# In[ ]:


popular = specs.merge(event[event>17000], how='inner', right_index=True, left_on='event_id')
popular = popular[['info','event_id_y']].sort_values(by='event_id_y', ascending=False)
for i in range(5):
    print(i, popular['info'].iloc[i])
    print('________________________________________')


# Let's visualize first events coordinates:

# In[ ]:


sec = train_sam[train_sam.event_id == '1325467d']
sec['xx'] = sec['event_data'].apply(lambda x: json.loads(x)['coordinates']['x'])
sec['yy'] = sec['event_data'].apply(lambda x: json.loads(x)['coordinates']['y'])


# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(sec['xx'], sec['yy']);


# - Firstly, we can see that several (57) values are outside the rectangle. It can be either a mistake or a specific group of users with another devices. Anyway, these event must be allocated to a separate group.
# - Secondary, we can see 4(5) clusters of clicks. It can be interesting feature - user that often click to the clusters, more distracted and spends more attempts to complete the level.
# - Same clusters can we see from another events. I think that these pictures can give us many usefull features.

# In[ ]:


fig = plt.figure(figsize=(20,15))
for i, event in enumerate(['1325467d','cf82af56','cfbd47c8','76babcde','6c517a88','884228c8']):
    fig.add_subplot(2,3,i+1)
    kk = train_sam[train_sam.event_id == event]
    kk['xx'] = kk['event_data'].apply(lambda x: json.loads(x)['coordinates']['x'])
    kk['yy'] = kk['event_data'].apply(lambda x: json.loads(x)['coordinates']['y'])
    plt.hist2d(kk['xx'],kk['yy'], bins=60, cmap=plt.cm.jet)


# This is only first version, to be continued...
