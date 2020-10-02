#!/usr/bin/env python
# coding: utf-8

# # 2019 Data Science Bowl

# One of the most awaited competitions is here!, let's do a quick EDA

# The dataset for this competitions comes from [PBS KIDS Measure Up! app](https://pbskids.org/apps/pbs-kids-measure-up.html), in this app children of ages 3 to 5 learn early math concepts focused on length, width, capacity, and weight. They have to navitage though maps and complete various levels, which may be activities, video clips, games, or **assessments**. Each **assessment** is designed to test a child's comprehension of a certain set of measurement-related skills. There are five assessments: Bird Measurer, Cart Balancer, Cauldron Filler, Chest Sorter, and Mushroom Sorter.
# The intent of the competition is to use the gameplay data to forecast how many attempts a child will take to pass a given **assessment**.

# In[ ]:


get_ipython().system('ls -lh ../input/data-science-bowl-2019/')


# In[ ]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook

import altair as alt
from altair.vega import v5
from IPython.display import HTML

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rc('figure', figsize=(15.0, 8.0))


# In[ ]:


root = '../input/data-science-bowl-2019/'
train = pd.read_csv(root + 'train.csv')
train_labels = pd.read_csv(root + 'train_labels.csv')
specs = pd.read_csv(root + 'specs.csv')
test = pd.read_csv(root + 'test.csv')
sample_submission = pd.read_csv(root + 'sample_submission.csv')


# In[ ]:


train.head()


# ### train.csv & test.csv
# 
# These are the main data files which contain the gameplay events.
# 
# * `event_id` - Randomly generated unique identifier for the event type. Maps to event_id column in specs table.
# * `game_session` - Randomly generated unique identifier grouping events within a single game or video play session.
# * `timestamp` - Client-generated datetime
# * `event_data` - Semi-structured JSON formatted string containing the events parameters. Default fields are: event_count, event_code, and game_time; otherwise fields are determined by the event type.
# * `installation_id` - Randomly generated unique identifier grouping game sessions within a single installed application instance.
# * `event_count` - Incremental counter of events within a game session (offset at 1). Extracted from event_data.
# * `event_code` - Identifier of the event 'class'. Unique per game, but may be duplicated across games. E.g. event code '2000' always identifies the 'Start Game' event for all games. Extracted from event_data.
# * `game_time` - Time in milliseconds since the start of the game session. Extracted from event_data.
# * `title` - Title of the game or video.
# * `type` - Media type of the game or video. Possible values are: 'Game', 'Assessment', 'Activity', 'Clip'.
# * `world` - The section of the application the game or video belongs to. Helpful to identify the educational curriculum goals of the media. Possible values are: 'NONE' (at the app's start screen), TREETOPCITY' (Length/Height), 'MAGMAPEAK' (Capacity/Displacement), 'CRYSTALCAVES' (Weight).
# 

# According to data page of the competition:
# 
# >Each application install is represented by an `installation_id`. This will typically correspond to one child, but you should expect noise from issues such as shared devices. In the training set, you are provided the full history of gameplay data. In the test set, we have truncated the history after the start event of a single assessment, chosen randomly, for which you must predict the number of attempts. Note that the training set contains many `installation_id`s which never took assessments, whereas every `installation_id` in the test set made an attempt on at least one assessment.

# In[ ]:


train.info()


# In[ ]:


train['installation_id'].unique().shape # total 17000 installations in train data


# In[ ]:


specs.head()


# ### specs.csv
# 
# This file gives the specification of the various event types.
# 
# * `event_id` - Global unique identifier for the event type. Joins to `event_id` column in events table.
# * `info` - Description of the event.
# * `args` - JSON formatted string of event arguments. Each argument contains:
#      - name - Argument name.
#      - type - Type of the argument (string, int, number, object, array).
#      - info - Description of the argument.
# 

# In[ ]:


specs.info()


# In[ ]:


train_labels.head()


# ### train_labels.csv
# This file demonstrates how to compute the ground truth for the assessments (`train.type == "Assessments"`) in the training set.
# 
# The outcomes in this competition are grouped into 4 groups (labeled `accuracy_group` in the data):
# 
#    * 3: the assessment was solved on the first attempt
#    * 2: the assessment was solved on the second attempt
#    * 1: the assessment was solved after 3 or more attempts
#    * 0: the assessment was never solved
# 
# 
# The file train_labels.csv has been provided to show how these groups would be computed on the assessments in the training set. Assessment attempts are captured in `event_code`, 4100 for all assessments except for Bird Measurer, which uses event_code 4110. If the attempt was correct, it contains `"correct":true`.

# In[ ]:


train_labels.info()


# In[ ]:


test.head()


# In[ ]:


test['installation_id'].unique().shape # total 1000 installations for which we have to predict


# In[ ]:


sample_submission.head()


# In[ ]:


sample_submission.shape # to predict for 1000 installations


# For each `installation_id` in test.csv we have to predict the accuracy group (based on the outcome of the assessment event)

# ## Let's do in-depth analysis

# Let's merge train, specs and train_labels dataframe

# In[ ]:


train = train.merge(specs, on='event_id')
train_labels = train.merge(train_labels, on=['game_session', 'installation_id']) # returns only type == Assessments


# Now:
# * `train` contains train + specs data for all event types.
# * `train_labels` contains train + specs + labels for all assessments

# In[ ]:


train.shape, train_labels.shape


# In[ ]:


train.head()


# In[ ]:


train_labels.head()


# `type`: Media type of the game or video, let's see `type`'s distribution in train/test set

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
plot = sns.countplot(y="type", data=train, palette=['navy', 'darkblue', 'blue', 'dodgerblue']).set_title('train type count', fontsize=16)
plt.yticks(fontsize=14)
plt.xlabel("Count", fontsize=15)
plt.ylabel("type", fontsize=15)
plt.show(plot)


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
plot = sns.countplot(y="type", data=test, palette=['navy', 'darkblue', 'blue', 'dodgerblue']).set_title('test type count', fontsize=16)
plt.yticks(fontsize=14)
plt.xlabel("Count", fontsize=15)
plt.ylabel("type", fontsize=15)
plt.show(plot)


# In[ ]:


train_by_type = train.groupby('type')
train_clip = train_by_type.get_group('Clip')
train_game = train_by_type.get_group('Game')
train_activity = train_by_type.get_group('Activity')
train_assessment = train_by_type.get_group('Assessment')


# In[ ]:


train_clip.head()


# In[ ]:


train_game.head()


# In[ ]:


train_activity.head()


# In[ ]:


train_assessment.head()


# To be updated!

# In[ ]:


sample_submission.to_csv('submission.csv', index=False)


# 

# In[ ]:




