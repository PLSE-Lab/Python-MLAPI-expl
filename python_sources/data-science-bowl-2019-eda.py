#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import seaborn as sns
import matplotlib.style as style
style.use('fivethirtyeight')
import matplotlib.pylab as plt
import calendar
import warnings
warnings.filterwarnings("ignore")


import datetime
from time import time
from tqdm import tqdm_notebook as tqdm
from collections import Counter
from scipy import stats

from sklearn.model_selection import GroupKFold
from typing import Any
from numba import jit
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics
from itertools import product
import copy
import time

import random
seed = 1234
random.seed(seed)
np.random.seed(seed)


# In[ ]:


train = pd.read_csv('../input/data-science-bowl-2019/train.csv')
train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
test = pd.read_csv('../input/data-science-bowl-2019/test.csv')
specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


train.columns


# '''Kaggle provided the following note: Note that the training set contains many installation_ids which never took assessments, whereas every installation_id in the test set made an attempt on at least one assessment.'''
# 
# Deleting 'installation_id' which are not "Assessment"

# In[ ]:


# Select all Assessment from installation_id
keep_id = train[train["type"] == "Assessment"]["installation_id"].drop_duplicates()

train = pd.merge(train, keep_id, on="installation_id", how="inner")


# In[ ]:


train.shape


# In[ ]:


keep_id.shape


# In[ ]:


train.type.value_counts()


# In[ ]:


fig = plt.figure(figsize=(12, 10))

ax1 = fig.add_subplot(211)
ax1 = sns.countplot(y="type", data=train, order= train.type.value_counts().index)
plt.title("number of events by type")

ax2 = fig.add_subplot(212)
ax2 = sns.countplot(y="world", data=train, order = train.world.value_counts().index)
plt.title("number of events by world")
plt.tight_layout(pad=0)


# In[ ]:


fig = plt.figure(figsize=(12,10))

title_count = train.title.value_counts().sort_values(ascending=True)
title_count.plot.barh()
plt.title("Event counts by title")


# In[ ]:


train['timestamp'] = pd.to_datetime(train['timestamp'])
train['date'] = train['timestamp'].dt.date
train['month'] = train['timestamp'].dt.month
train['hour'] = train['timestamp'].dt.hour
train['dayofweek'] = train['timestamp'].dt.dayofweek


# In[ ]:


fig = plt.figure(figsize=(12,10))
date = train.groupby('date')['date'].count()
date.plot()
plt.xticks(rotation=90)
plt.title("Event counts by date")


# In[ ]:


fig = plt.figure(figsize=(12,10))
day_of_week = train.groupby('dayofweek')['dayofweek'].count()
# convert num -> category
day_of_week.index = list(calendar.day_abbr)
day_of_week.plot.bar()
plt.title("Event counts by day of week")
plt.xticks(rotation=0)


# In[ ]:


fig = plt.figure(figsize=(12,10))
hour = train.groupby('hour')['hour'].count()
hour.plot.bar()
plt.title("Event counts by hour of day")
plt.xticks(rotation=0)


# In[ ]:


test.head()


# In[ ]:


test.shape


# In[ ]:


test.installation_id.nunique()


# There are no installation_ids without assessment in the test set

# In[ ]:


sample_submission.shape


# Check if there is any overlap with regards to installation_id's in the train and test set

# In[ ]:


set(list(train.installation_id.unique())).intersection(list(test.installation_id.unique()))


# There are no installation_id's that appear in both train and test

# In[ ]:


test['timestamp'] = pd.to_datetime(test['timestamp'])

print(f'The range of date in train is: {train.date.min()} to {train.date.max()}')
print(f'The range of date in test is: {test.timestamp.dt.date.min()} to {test.timestamp.dt.date.max()}')


# train_labels dataset

# In[ ]:


train_labels.head()


# In[ ]:


train_labels.shape


# In[ ]:


pd.crosstab(train_labels.title, train_labels.accuracy_group)


# accuracy_group =>
# * 0: the assessment was never solved
# * 1: the assessment was solved after 3 or more attempts
# * 2: the assessment was solved on the second attempt
# * 3: the assessment was solved on the first attempt

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(y="title", data=train_labels, order = train_labels.title.value_counts().index)
plt.title("Counts of titles")


# In[ ]:


df = train_labels.groupby(['accuracy_group', 'title'])['accuracy_group']
df.count()


# In[ ]:


se = train_labels.groupby(['title', 'accuracy_group'])['accuracy_group'].count().unstack('title')
se.plot.bar(stacked=True, rot=0, figsize=(12,10))
plt.title("Counts of accuracy group")


# installation_id's who did assessments but without results in the train_labels (we have already taken out the ones who never took one)

# In[ ]:


train[~train.installation_id.isin(train_labels.installation_id.unique())].installation_id.nunique()


# Cannot train on installation_id's anyway, so taking them out of the train set. This reduces train set from 8.3 million rows to 7.7 million.

# In[ ]:


train = train[train.installation_id.isin(train_labels.installation_id.unique())]
train.shape


# Feature engineering

# In[ ]:


print(f'No. of rows in train_labels: {train_labels.shape[0]}')
print(f'Number of unique game_sessions in train_labels: {train_labels.game_session.nunique()}')


# In[ ]:


train = train.drop(['date', 'month', 'hour', 'dayofweek'], axis=1)


# In[ ]:


def encode_title(train, test, train_labels):
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    
    all_title_event_code = list(set(train['title_event_code'].unique()).union(test['title_event_code'].unique()))
    
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    
    activities_map = dict(zip(list_of_user_activities, np,arange(len(list_of_user_activities))))
    
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    
    # replace title with its number from the dictionary
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    
    train_labels['title'] = train_labels['title'].map(activities_map)
    
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    
    return train, test, train_labels, all_title_event_code, list_of_user_activities, list_of_event_code, list_of_event_id, activities_labels, assess_titles
    


# In[ ]:


train, test, train_labels, all_title_event_code, list_of_user_activities, list_of_event_code, list_of_event_id, activities_labels, assess_titles = encode_title(train, test, train_labels)

