#!/usr/bin/env python
# coding: utf-8

# <center> **GOAL OF THIS NOTEBOOK: Understand what is going on (assuming you've read the [description of the competition](https://www.kaggle.com/c/data-science-bowl-2019/overview))** </center>

# In[ ]:


import numpy as np
import pandas as pd
import os
import json

pd.set_option('max_rows', 500)


# # Get the data

# We start by looking at the `train` dataset. It takes some time to upload.

# In[ ]:


train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
train.head()


# > **Note:** The target variable is not given explicitly in this table, but rather given in `train_labels.csv`, which will be discussed later.

# First we modify the `timestamp` to be a datetime and sort by it. This will be very useful later when we will follow the progress of a specific user.

# In[ ]:


train.loc[:, 'timestamp'] = pd.to_datetime(train.timestamp)
train.sort_values('timestamp', inplace=True)


# For the analysis of the data it is important to understand what `game_session` means exactly. By the description page of the competition we know that `game_session` is a "Randomly generated unique identifier grouping events within a single game or video play **session**". The usage of the *Measure Up!* app is repetitive by nature (because kids watch the same clip many times and repeat the same activities many times), this identifier allows us to distinguish between otherwise identical sessions.
# 
# So to put it all together, we can describe the usage of the app like this: An installation is made of many sessions. Within a single `game_session`, the user selects a `type` of activity with a given `title` from the available options of the selected `world`. Then the session it is described as sequence of one or more events. To see the data more clearly we change the order of columns.

# In[ ]:


new_order = ['timestamp', 'installation_id', 'game_session', 'world', 'type', 'title', 'game_time', 'event_count' , 'event_code', 'event_id', 'event_data']
train = train.loc[:, new_order]
train.head()


# It is easy to conclude that `title` is the name of a specific type of activity (e.g. by `pd.crosstab(df_inst.title, df_inst.type)` and `pd.crosstab(train.world, train.title)`). Based on this understanding we can construct on auxiliary dictionary (manually). I don't know whether I will use it or not, but it helps me.

# In[ ]:


d_world_type_title = {'TREETOPCITY': {'Activity': ['Fireworks', 'Flower Waterer', 'Bug Measurer'], 
                                      'Assessment': ['Mushroom Sorter', 'Bird Measurer'], 
                                      'Clip': ['Tree Top City - Level 1', 'Ordering Spheres', 'Costume Box', '12 Monkeys', 
                                               'Tree Top City - Level 2', "Pirate's Tale", 'Treasure Map', 'Tree Top City - Level 3', 'Rulers'], 
                                      'Game': ['All Star Sorting', 'Air Show', 'Crystals Rule']}, 
                      'MAGMAPEAK': {'Activity': ['Sandcastle Builder', 'Watering Hole', 'Bottle Filler'], 
                                    'Assessment': ['Cauldron Filler'], 
                                    'Clip': ['Magma Peak - Level 1', 'Slop Problem', 'Magma Peak - Level 2'], 
                                    'Game': ['Scrub-A-Dub', 'Dino Drink', 'Bubble Bath', 'Dino Dive']}, 
                      'CRYSTALCAVES': {'Activity': ['Chicken Balancer', 'Egg Dropper'], 
                                       'Assessment': ['Cart Balancer', 'Chest Sorter'], 
                                       'Clip': ['Crystal Caves - Level 1', 'Balancing Act', 'Crystal Caves - Level 2', 
                                                'Crystal Caves - Level 3', 'Lifting Heavy Things', 'Honey Cake', 'Heavy, Heavier, Heaviest'], 
                                       'Game': ['Chow Time', 'Pan Balance', 'Happy Camel', 'Leaf Leader']}}


# # Basic statistics

# There are 17000 unique installations.

# In[ ]:


train.installation_id.nunique()


# We note that most of the users do not have assessments at all (in the test data the last record of every installation has `type=='Assessment'`).

# In[ ]:


train.groupby('installation_id')['type'].apply(lambda s: s.isin(['Assessment']).any()).value_counts()


# A normal user has 1-30 sessions.

# In[ ]:


sessions = train.groupby('installation_id')['game_session'].nunique()


# In[ ]:


ax = sessions.value_counts().iloc[:30].plot.bar(title='Sessions per installation (top 30)')


# # Follow an installation

# In[ ]:


my_inst = train.installation_id.iloc[12345]  # and others...
df_inst = train.loc[train.installation_id==my_inst]
print(len(df_inst))
df_inst.head()


# The installation is made of separate sessions, and it is easier to follow the progress of the user by following an aggregative overview of its session. 

# In[ ]:


df_sessions = df_inst.groupby('game_session')    .apply(lambda df_session: {'timestamp': df_session.timestamp.min(),
                               'world': df_session.world.iloc[0],
                               'title': df_session.title.iloc[0],
                               'type': df_session.type.iloc[0], 
                               'length': df_session.game_time.max(), 
                               'events': df_session.event_count.max()})\
    .apply(pd.Series)


# In[ ]:


df_sessions.head(30)


# > **Reference:** [An explanation about some in-app terms](https://www.kaggle.com/c/data-science-bowl-2019/discussion/115034)

# # Assessment session

# In[ ]:


my_session = df_sessions.loc[df_sessions.type=='Assessment'].index[0]
df_session = df_inst.loc[df_inst.game_session==my_session]
df_session.head()


# We can use the `event_code` (which will be explored later) as a discriminator for visualization (idea taken from [this kernel](https://www.kaggle.com/robikscube/2019-data-science-bowl-an-introduction)).

# In[ ]:


df_session.set_index('timestamp')['event_code']     .plot(style='.')


# ## Event data

# > **Note:** More information about the events can be found in the `specs.csv` file, but I will not use it in this notebook.
# 
# ```python
# specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
# specs.head()
# ```

# To illustrate the event data we follow the `event_code` 4100 or 4110, which are correlated with assessment attempts.

# In[ ]:


for idx, row in df_inst.iterrows():
    if row.type=='Assessment':
        if row.event_code in [4100, 4110]:
            event_data = json.loads(row.event_data)
            print(f"{row.game_session:20}, {row.title:30}, {event_data['correct']}")


# # The *train_labels* dataset

# As explained in the competition page, this dataset is not really necessary, as it can be derived from the `train` dataset (and the `event_data` column in particular).

# In[ ]:


train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
train_labels.head()


# As explained in the competition page, this is a classification problem, where the target is the `accuracy_group`. The `accuracy` is an auxiliary column, evaluated by the trials of the user, as derived from the event data (illustrated below).

# > **Note:** Many kernels consider the problem as a regression problem for the accuracy.

# Now if we consider our specific installation we will find the session(s) we've looked at ealier.

# In[ ]:


my_train_labels = train_labels.loc[train_labels.installation_id==my_inst]
my_train_labels


# > **Note:** The `accuracy_group` is evaluated per *session*, while the predicted `accuracy_group` is per *installation*. This is because we are asked to predict the `accuracy_group` of the last given session, from which we get only the opening event.

# > **Reference:** This is a good point to understand the evaluation metric defined in this competition, the [quadratic weighted kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa). A good explanation about this metric and its relevancy to this competition can be found in [this discussion](https://www.kaggle.com/c/data-science-bowl-2019/discussion/114539). You can calculate this metric in Scikit-learn using [`cohen_kappa_score(a, p, weights="quadratic")`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html), however [this kernel](https://www.kaggle.com/cpmpml/ultra-fast-qwk-calc-method) offers a 300x faster implementation.

# # Model 1

# In[ ]:


test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
test.loc[:, 'timestamp'] = pd.to_datetime(test.timestamp)
test.sort_values('timestamp', inplace=True)


# For our first model (taken from [this kernel](https://www.kaggle.com/mhviraf/a-baseline-for-dsb-2019)) we will ignore the specific installation and look at the entire data. For any assessment, regardless of the specific installation data, we predict the mode (most frequent value) of the `accuracy_group` for this assessment.

# In[ ]:


labels_map = dict(train_labels.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0])) # get the mode
labels_map


# In[ ]:


test_predictions = test.groupby('installation_id').last()['title'].map(labels_map).rename("accuracy_group")
test_predictions


# In[ ]:


submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
submission.head()


# In[ ]:


submission = submission    .join(test_predictions, on='installation_id', lsuffix='_orig')    .drop('accuracy_group_orig', axis=1)


# In[ ]:


submission.to_csv('submission.csv', index=None)


# # Analyzing the test dataset

# This section simply illustrates how each test installation ends with a single event of an Assessment session.

# In[ ]:


my_inst = test.installation_id.iloc[123]
df_inst = test.loc[test.installation_id==my_inst]


# In[ ]:


df_sessions = df_inst.groupby('game_session')    .apply(lambda df_session: {'timestamp': df_session.timestamp.min(),
                               'world': df_session.world.iloc[0],
                               'title': df_session.title.iloc[0],
                               'type': df_session.type.iloc[0], 
                               'length': df_session.game_time.max(), 
                               'events': df_session.event_count.max()})\
    .apply(pd.Series)


# In[ ]:


df_sessions.sort_values('timestamp')

