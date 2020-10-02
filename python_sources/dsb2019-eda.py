#!/usr/bin/env python
# coding: utf-8

# ## [What to predict?](#h1)
# > For each installation_id represented in the test set, you must predict the accuracy_group of the last assessment for that installation_id. 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc, sys, math, random
import datetime
import re
import json
from pandas.io.json import json_normalize
from tqdm._tqdm_notebook import tqdm_notebook as tqdm

import warnings
warnings.filterwarnings('ignore')

sns.set_style('darkgrid')

pd.options.display.float_format = '{:,.3f}'.format

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# # Load data

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain = pd.read_csv('../input/data-science-bowl-2019/train.csv')\ntest = pd.read_csv('../input/data-science-bowl-2019/test.csv')\ntrain_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')\nspecs = pd.read_csv('../input/data-science-bowl-2019/specs.csv', converters={'args': json.loads})\n\ntrain = reduce_mem_usage(train)\ntest = reduce_mem_usage(test)")


# In[ ]:


train.info()


# In[ ]:


train.head()


# In[ ]:


specs.info()


# In[ ]:


specs.head()


# In[ ]:


train_labels.info()


# In[ ]:


train_labels.head()


# # Data analysis

# ## train.csv & test.csv
# 
# These are the main data files which contain the gameplay events.
# 
# - event_id - Randomly generated unique identifier for the event type. Maps to event_id column in specs table.
# - game_session - Randomly generated unique identifier grouping events within a single game or video play session.
# - timestamp - Client-generated datetime
# - event_data - Semi-structured JSON formatted string containing the events parameters. Default fields are: event_count, event_code, and game_time; otherwise fields are determined by the event type.
# - installation_id - Randomly generated unique identifier grouping game sessions within a single installed application instance.
# - event_count - Incremental counter of events within a game session (offset at 1). Extracted from event_data.
# - event_code - Identifier of the event 'class'. Unique per game, but may be duplicated across games. E.g. event code '2000' always identifies the 'Start Game' event for all games. Extracted from event_data.
# - game_time - Time in milliseconds since the start of the game session. Extracted from event_data.
# - title - Title of the game or video.
# - type - Media type of the game or video. Possible values are: 'Game', 'Assessment', 'Activity', 'Clip'.
# - world - The section of the application the game or video belongs to. Helpful to identify the educational curriculum goals of the media. Possible values are: 'NONE' (at the app's start screen), TREETOPCITY' (Length/Height), 'MAGMAPEAK' (Capacity/Displacement), 'CRYSTALCAVES' (Weight).

# ### event_id

# In[ ]:


train['event_id'].nunique(), test['event_id'].nunique()


# ### game_session

# In[ ]:


train['game_session'].nunique(), test['game_session'].nunique()


# In[ ]:


common_sesseion = set(train['game_session'].unique()) & set(test['game_session'].unique())
list(common_sesseion)


# ### timestamp

# In[ ]:


print('train: ', train['timestamp'].min(), 'to', train['timestamp'].max()) 
print('test:  ', test['timestamp'].min(), 'to', test['timestamp'].max())


# In[ ]:


_='''
START_DATE = '2019-07-23T00:00:00+0000'
startdate = pd.to_datetime(START_DATE)
train_day = train['timestamp'].apply(lambda x: (pd.to_datetime(x) - startdate).total_seconds() // (24 * 60 * 60))
train_day.hist()
'''


# In[ ]:


max_events_game_session = train[train['event_count'] == train['event_count'].max()]['game_session'].values[0]
subset = train[train['game_session'] == max_events_game_session][['game_session','event_count','timestamp','game_time']]
subset['timestamp'] = pd.to_datetime(subset['timestamp'])

min_ts = subset['timestamp'].min()
subset['timestamp_diff_sec'] = subset['timestamp'].apply(lambda x: int((x - min_ts).total_seconds() * 1000))
subset


# ### installation_id

# In[ ]:


train['installation_id'].nunique(), test['installation_id'].nunique()


# In[ ]:


common_installation = set(train['installation_id'].unique()) & set(test['installation_id'].unique())
list(common_installation)


# In[ ]:


game_session_count = train.groupby('installation_id')['game_session'].nunique()
game_session_count.describe()


# ### event_data

# In[ ]:


train_corrected = train['event_data'].str.contains('"correct":true')
test_corrected = test['event_data'].str.contains('"correct":true')

fig, ax = plt.subplots(1, 2, figsize=(12,3))

train_subset = train[train_corrected]
train_subset['event_code'].value_counts().plot.bar(ax=ax[0])
print('train:', train_subset['type'].unique(), train_subset['event_code'].unique())

test_subset = test[test_corrected]
test_subset['event_code'].value_counts().plot.bar(ax=ax[1], color='limegreen')
print('test:', test_subset['type'].unique(), test_subset['event_code'].unique())


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(12,3))

train_corrected.value_counts().plot.bar(ax=ax[0])
test_corrected.value_counts().plot.bar(ax=ax[1], color='limegreen')


# ### event_count

# In[ ]:


train['event_count'].hist(bins=100,figsize=(12,3))
train['event_count'].describe().to_frame()


# In[ ]:


test['event_count'].hist(bins=100,figsize=(12,3), color='limegreen')
test['event_count'].describe().to_frame()


# ### event_code

# In[ ]:


train['event_code'].nunique(), test['event_code'].nunique() 


# In[ ]:


fig, ax = plt.subplots(2, 1, figsize=(12,7))

train['event_code'].value_counts().plot.bar(ax=ax[0])
test['event_code'].value_counts().plot.bar(ax=ax[1], color='limegreen')


# ### title

# In[ ]:


train['title'].nunique(), test['title'].nunique()


# In[ ]:


np.sort(train['title'].unique())


# In[ ]:


train['title'].value_counts().plot.bar(figsize=(12,3))


# In[ ]:


test['title'].value_counts().plot.bar(figsize=(12,3), color='limegreen')


# In[ ]:


train_title = train['title'].unique()
test_title = train['title'].unique()
[n for n in train_title if n not in test_title] + [n for n in test_title if n not in train_title]


# In[ ]:


# type & world is unique in same title
print('[title + type] unique count:', (train['title'] + '-' +  train['type']).nunique())
print('[title + world] unique count:', (train['title'] + '-' +  train['world']).nunique())


# In[ ]:


cols = ['world','type','title']
train[cols].drop_duplicates().sort_values(by=cols).reset_index(drop=True)


# ### type

# In[ ]:


train['type'].unique()


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(12,3))

train['type'].value_counts().plot.bar(ax=ax[0])
test['type'].value_counts().plot.bar(ax=ax[1], color='limegreen')


# In[ ]:


for t in train['type'].unique():
    print(f'events of {t}:\n', np.sort(train[train['type'] == t]['event_code'].unique()))


# In[ ]:


train[(train['event_code'] == 4100) | (train['event_code'] == 4110)]['type'].value_counts().plot.bar()


# ### world

# In[ ]:


train['world'].unique()


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(12,3))

train['world'].value_counts().plot.bar(ax=ax[0])
test['world'].value_counts().plot.bar(ax=ax[1], color='limegreen')


# ### event_id & event_code

# In[ ]:


print('nunique event_id:', train['event_id'].nunique())
print('nunique event_code:', train['event_code'].nunique())
print('nunique event_id&event_code:', (train['event_id'] + '-' + train['event_code'].astype(str)).nunique())


# In[ ]:


train.groupby('event_code')['event_id'].unique().to_frame()


# In[ ]:


# start event
train[train['event_count'] == 1]['event_code'].value_counts()


# In[ ]:


subset = train[(train['event_code'] == 4100) | (train['event_code'] == 4110)]
subset


# ## specs.csv
# 
# This file gives the specification of the various event types.
# 
# - event_id - Global unique identifier for the event type. Joins to event_id column in events table.
# - info - Description of the event.
# - args - JSON formatted string of event arguments. Each argument contains:
# - name - Argument name.
# - type - Type of the argument (string, int, number, object, array).
# - info - Description of the argument.

# In[ ]:


specs['event_id'].nunique(), len(specs)


# In[ ]:


specs['info'][0]


# In[ ]:


json_normalize(specs['args'][0])


# In[ ]:


print('max(args array size):', max(specs['args'].apply(len)))


# In[ ]:


max_colwidth = pd.get_option('display.max_colwidth')
max_rows = pd.get_option("display.max_rows")


# In[ ]:


_='''
pd.set_option('display.max_colwidth', 1000)
pd.set_option("display.max_rows", 500)

subset = train[['event_code','event_id']]
subset = subset[~subset.duplicated()]
event_info = pd.merge(subset, specs[['event_id','info']]).sort_values(by=['event_code','info'])
event_info.to_csv("event_info.csv", index=False)
display(event_info)

pd.set_option('display.max_colwidth', max_colwidth)
pd.set_option("display.max_rows", max_rows)
'''


# ## train_labels.csv
# 
# This file demonstrates how to compute the ground truth for the assessments in the training set.
# 
# - game_session
# - installation_id
# - title
# - num_correct
# - num_incorrect
# - accuracy
# - accuracy_group
# 
# The outcomes in this competition are grouped into 4 groups (labeled accuracy_group in the data):
# 
# - 3: the assessment was solved on the first attempt
# - 2: the assessment was solved on the second attempt
# - 1: the assessment was solved after 3 or more attempts
# - 0: the assessment was never solved

# In[ ]:


train_labels.head()


# In[ ]:


max(train_labels['num_correct'] + train_labels['num_incorrect'])


# In[ ]:


print('[train] sessions:', train['game_session'].nunique(), ', installations:', train['installation_id'].nunique(), ', len:', len(train))
print('[test] sessions:', test['game_session'].nunique(), ', installations:', test['installation_id'].nunique(), ', len:', len(test))
print('[labels] sessions:', train_labels['game_session'].nunique(), ', installations:', train_labels['installation_id'].nunique(), ', len:', len(train_labels))


# In[ ]:


s = '77b8ee947eb84b4e'
subset = train[train['game_session'] == s]
subset['event_code'].value_counts().to_frame().T


# In[ ]:


s = '9501794defd84e4d'
subset = train[train['game_session'] == s]
subset['event_code'].value_counts().to_frame().T


# In[ ]:


'''
Assessment attempts are captured in event_code 4100 for all assessments except for Bird Measurer, 
which uses event_code 4110. If the attempt was correct, it contains "correct":true.
'''
# except for
train[(train['title'] == 'Bird Measurer (Assessment)') & ((train['event_code'] == '4100') | (train['event_code'] == '4110'))]


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntrain[\'event_data.correct\'] = \'\'\ntrain.loc[train[\'event_data\'].str.contains(\'"correct":true\'), \'event_data.correct\'] = \'true\'\ntrain.loc[train[\'event_data\'].str.contains(\'"correct":false\'), \'event_data.correct\'] = \'false\'\n\ntrain[\'event_code_correct\'] = train[\'event_code\'].astype(str) + train[\'event_data.correct\'].apply(lambda x: x if x == \'\' else f\'({x})\')\ntrain_events = train.groupby(\'game_session\').apply(lambda x: x[\'event_code_correct\'].value_counts().to_frame().T.reset_index(drop=True))\ntrain_events = train_events.reset_index().drop(\'level_1\', axis=1)\ntrain_events.head()')


# In[ ]:


train_events['_event_correct'] = train_events[['2010','2030','4100(true)','4110(true)']].sum(axis=1) # ,3021,3121
train_events['_event_incorrect'] = train_events[['4100(false)','4110(false)']].sum(axis=1) #'3020','3120',
train_events.fillna(0, inplace=True)

session_events = pd.merge(train_labels, train_events)

cols = ['num_correct','num_incorrect']
session_events[cols + list(train_events.columns[1:])].corr()[cols].sort_index()


# In[ ]:


session_events['_calc_accuracy_group'] = 0
session_events.loc[(session_events['_event_incorrect'] == 0) & (session_events['_event_correct'] > 0),'_calc_accuracy_group'] = 3
session_events.loc[(session_events['_event_incorrect'] == 1) & (session_events['_event_correct'] > 0),'_calc_accuracy_group'] = 2
session_events.loc[(session_events['_event_incorrect'] >= 2) & (session_events['_event_correct'] > 0),'_calc_accuracy_group'] = 1

session_events['_calc_accuracy_group'].value_counts().plot.bar()


# In[ ]:


session_events[['accuracy_group','_event_correct','_event_incorrect','_calc_accuracy_group']]


# <a id="h1"></a> <br>
# # What to predict?

# In[ ]:


ins_last_session = test.drop_duplicates(subset=['installation_id'], keep='last')
ins_last_session.head()


# In[ ]:


# last session contains only start-event
ins_last_session['event_count'].unique(), ins_last_session['event_code'].unique()


# In[ ]:


ins_last_session = ins_last_session[['installation_id','game_session']]
ins_last_session['last_session'] = 1

test = pd.merge(test, ins_last_session, how='left')
test['last_session'].fillna(0, inplace=True)

test['last_session'].value_counts()


# In[ ]:


print('last session events(mean): ', np.mean(test[test['last_session'] == 1].groupby('game_session')['event_id'].count()))


# <span style="font-size:large;color:blue">We have to predict to last sessions accuracy group.</span>

# In[ ]:




