#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ![DSB](https://github.com/nadarsubash/articles/blob/master/dataScibowl.jpg?raw=true)

# ### About the Dataset
# 
# ##### This Dataset include game analytics for the PBS KIDS Measure Up! app. In this app, children navigate a map and complete various levels, which may be activities, video clips, games, or assessments. Each assessment is designed to test a child's comprehension of a certain set of measurement-related skills.  
# 
# These are the main data files which contain the gameplay events.<br>
# 
# **event_id** - Randomly generated unique identifier for the event type. Maps to event_id column in specs table.<br>
# **game_session** - Randomly generated unique identifier grouping events within a single game or video play session.<br>
# **timestamp** - Client-generated datetime<br>
# **event_data** - Semi-structured JSON formatted string containing the events parameters. Default fields are: event_count, event_code, and game_time; otherwise fields are determined by the event type.<br>
# **installation_id** - Randomly generated unique identifier grouping game sessions within a single installed application instance.<br>
# **event_count** - Incremental counter of events within a game session (offset at 1). Extracted from event_data.<br>
# **event_code** - Identifier of the event 'class'. Unique per game, but may be duplicated across games. E.g. event code '2000' always identifies the 'Start Game' event for all games. Extracted from event_data.<br>
# **game_time** - Time in milliseconds since the start of the game session. Extracted from event_data.<br>
# **title** - Title of the game or video.<br>
# **type** - Media type of the game or video. Possible values are: 'Game', 'Assessment', 'Activity', 'Clip'.<br>
# **world** - The section of the application the game or video belongs to. Helpful to identify the educational curriculum goals of the media. Possible values are: 'NONE' (at the app's start screen), TREETOPCITY' (Length/Height), 'MAGMAPEAK' (Capacity/Displacement), 'CRYSTALCAVES' (Weight).

# In[ ]:


data_train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
data_test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')


# ### View top 5 records in the data

# In[ ]:


data_train.head()


# ### View rows in the data with event code 4100 *(row with Assessment detail)*
# 

# In[ ]:


dt = data_train[data_train['event_code']==4100]
dt.head()


# In[ ]:


data_specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
data_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')


# ### Below is a view of *Assessment outcome* for each User (installation_id) per game_session

# In[ ]:


data_labels.head()


# ### Let's see data information

# In[ ]:


print('---------------Train Data---------------')
print(data_train.info())
print('---------------Test Data---------------')
print(data_test.info())
print('---------------Labels Data---------------')
print(data_labels.info())


# ### Let's check if any blank cells

# In[ ]:



print('---------------Train Data---------------')
print(data_train.isna().sum())
print('---------------Test Data---------------')
print(data_test.isna().sum())


# ### Dataset is clean. No missing data!!!

# In[ ]:


print('---------------Train Data---------------')
print(data_train.installation_id.nunique())
print('---------------Test Data---------------')
print(data_test.installation_id.nunique())


# ### There are 17000 unique users in the Training Data Set & 1000 in Test Data
# 

# ### Below is view of the 'world' wise split of game session, basis popularity

# In[ ]:


data_train.world.value_counts().plot(kind='bar')


# ### Magmapeak is by far the most played 'World'

# ### 'world' view Title wise

# In[ ]:


grp = data_train.groupby(['world','title'])


# In[ ]:


grp.size()


# ### Let's see the Data for all the assessment in the respective World

# In[ ]:


grp1 = data_train.query("event_code==4100 or event_code==4110").groupby(['world','title'])


# In[ ]:


grp1.size()


# In[ ]:


grp1.size().plot(kind='pie')


# ### 'world' view 'type' wise

# In[ ]:


grpt = data_train.groupby(['world','type'])
grpt.size()


# In[ ]:


grpt.size().plot(kind='pie')

