#!/usr/bin/env python
# coding: utf-8

# # Sessions EDA
# 
# For this dataset, we have Installations (~Users), Game Sessions and Events. This seems like a good start, however a Game Session is not quite representative of learning patterns. It's well known that people should take breaks while studying because learning drastically drops off after 40 minutes or so. To get a better understanding of how much learning has taken place, rather than game sessions, we'll look at sessions. This approach is also commonly used in Ecommerce.
# 
# Sessions will be defined as consecutive events separated by no more than 15 minutes. This is motivated by the maximum length of clips (156s) as well as the recommended time for a break of 10 minutes.

# In[ ]:


import pandas as pd 
import numpy as np 
import os

import matplotlib.pyplot as plt
import seaborn as sns

from random import sample

np.random.seed(0)


# ## Read Data
# 
# Read in all the data and process to get some convenient lists.
# 
# Functions taken from [this kernel](https://www.kaggle.com/braquino/890-features).

# In[ ]:


def read_data():
    print('Reading train.csv file....')
    train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, train_labels, specs, sample_submission

def encode_title(train, test, train_labels):
    # encode title
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))
    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    
    
    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code

# read data
train, test, train_labels, specs, sample_submission = read_data()
# get usefull dict with maping encode
train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)


# ## Data Augmentation
# 
# Let's augment the data to provide some usefil timings.

# In[ ]:


if 'start_installation' in train.columns:
    train.drop(['start_installation', 'start_game_session'],
               axis=1, inplace=True)

# Add time since start
train = pd.merge(
    left=train,
    right=train \
        .groupby('installation_id') \
        .timestamp.min() \
        .to_frame() \
        .rename(columns={'timestamp': 'start_installation'}),
    how='inner',
    left_on='installation_id',
    right_on='installation_id'
) \
    .assign(seconds_since_installation =  \
                lambda x: (x.timestamp - x.start_installation) \
                .apply(lambda x: x.total_seconds()))

# Add time since session start
# Assuming game_session is unique
train = pd.merge(
    left=train,
    right=train \
        .groupby('game_session') \
        .timestamp.min() \
        .to_frame() \
        .rename(columns={'timestamp': 'start_game_session'}),
    how='inner',
    left_on='game_session',
    right_on='game_session'
) \
    .assign(seconds_since_game_session_start = \
                lambda x: (x.timestamp - x.start_game_session) \
                .apply(lambda x: x.total_seconds()))

# Add the mean timestamp and plot heatmap
installation_props = train     .query('event_count == 1')     .groupby('installation_id')     .seconds_since_installation.mean()     .to_frame()     .rename(columns={'seconds_since_installation':'seconds_since_installation_mean'})

train = pd.merge(
    train,
    installation_props,
    how='inner',
    left_on='installation_id',
    right_index=True)


# In[ ]:


train.head()


# ## Creating Sessions
# To create the sessions, we define the cutoff as mentioned above and then events together under each installation to find the '[islands](https://www.red-gate.com/simple-talk/sql/t-sql-programming/the-sql-of-gaps-and-islands-in-sequences/)' of events. These session ids are then merged back with the original data source.

# In[ ]:


SESSION_CUTOFF=15*60  # seconds

session_ids = train     .groupby('installation_id')     .timestamp     .apply(lambda x: x.sort_values()                .diff()                .apply(lambda y: y.total_seconds() > SESSION_CUTOFF)                .cumsum())     .reset_index()     .set_index('level_1')     .rename(columns={'timestamp': 'session_id'})     .session_id

# Add session_ids to train
train = pd.merge(train,
                 session_ids,
                 how='inner',
                 left_index=True,
                 right_index=True)


# In[ ]:


def duration(x):
    return x.max()-x.min()

sessions = train    .groupby(['installation_id' , 'session_id'])     .agg({'seconds_since_installation': ['min', 'max', 'count', duration],
          'game_session': 'nunique'})

sessions.head()


# ## Visualisation
# We can now visualise properties about these sessions.

# ### Session Counts
# How many sessions per installation?

# In[ ]:


train     .groupby(['installation_id'])     .session_id.nunique()     .value_counts(normalize=True, sort=True)     .head(10)     .apply(lambda x: x*100)     .plot(kind='bar')
plt.xlabel('Number of Sessions')
plt.ylabel('Percent of Installations')


# In[ ]:


print('Cummulative Installations with Session Count')
train     .groupby(['installation_id'])     .session_id.nunique()     .value_counts(normalize=True, sort=True)     .cumsum()     .head(10)


# Unsurprisingly, over 50% of the installations have just one session. The user installs the game and then quickly forgets about it - oh, poor developer! That said, there is still a decent number of installations with multiple sessions, which means the app is getting more regular usage. 

# ### Game Session Counts
# How many games being played in each session?

# In[ ]:


print('Game Session Counts')
sessions.game_session['nunique'].describe()


# In[ ]:


sessions.game_session['nunique']     .plot('box', vert=False, sym='')
plt.xlabel('Game Session Count')
plt.title('Boxplot of Game Session Counts')


# ### Session Durations
# How long is the average session?

# In[ ]:


print('Sessions Description')
sessions.seconds_since_installation.duration.describe()


# The biggest session is a whopping 4 hours, that's some addictive gameplay right there! Though the quality of the output may have dropped a little...
# 
# We can also visualise this if we drop the outliers.

# In[ ]:


sessions.seconds_since_installation.duration     .plot('box', vert=False, sym='')
plt.xlabel('Session Duration (seconds)')
plt.title('Boxplot of Session Durations')


# ### Sessions Over Time
# How do we sessions appearing over time? Are they all at once, or over multiple days? Are there big gaps?
# 
# Let's plot the sessions in time for a random sample of installations.

# In[ ]:


results ={}
num_samples = 50
num_days = 80
for installation in sample(list(train.installation_id.unique()), num_samples):
    count, division = np.histogram(train         .query('installation_id == @installation')         .drop_duplicates(subset=['installation_id', 'session_id'], keep='first')
        .seconds_since_installation \
        .apply(lambda x: x/60/60/24),
        bins = np.linspace(0, num_days, num_days + 1))
    results[installation] = count
division = list(map(int, division))
results = pd.DataFrame.from_dict(results,
                                 orient='index',
                                 columns=division[:-1])

sns.heatmap(
    pd.merge(
        results,
        installation_props,
        how='inner',
        left_index=True,
        right_index=True
    ) \
        .sort_values(by='seconds_since_installation_mean', ascending=True) \
        .drop(['seconds_since_installation_mean'], axis=1),
    yticklabels=False,
    vmax=3,
    xticklabels=5
)
fig = plt.gcf()
fig.set_size_inches(12,8)
plt.xlabel('Days since installation')
plt.ylabel('Installations')
plt.title('Daily Session Counts per Installation')


# Again, we see that the majority of users are with few sessions, and even those with multiple sessions are seen to have them on just one day. Looking past that we do see some interesting behaviour, like users combing back every few days or even those coming back 70 days after the installation with nothing in between.
# 
# We can get a closer look at those installations with sessions over many days by doing some filtering.

# In[ ]:


results ={}
num_samples = 50
num_days = 80
for installation in sample(list(train.query('seconds_since_installation_mean > 3*24*60*60').installation_id.unique()), num_samples):
    count, division = np.histogram(train         .query('installation_id == @installation')         .query('seconds_since_installation_mean > 1*24*60*60')         .drop_duplicates(subset=['installation_id', 'session_id'], keep='first')
        .seconds_since_installation \
        .apply(lambda x: x/60/60/24),
        bins = np.linspace(0, num_days, num_days + 1))
    results[installation] = count
division = list(map(int, division))
results = pd.DataFrame.from_dict(results,
                                 orient='index',
                                 columns=division[:-1])


# In[ ]:


sns.heatmap(
    pd.merge(
        results,
        installation_props,
        how='inner',
        left_index=True,
        right_index=True
    ) \
        .sort_values(by='seconds_since_installation_mean', ascending=True) \
        .drop(['seconds_since_installation_mean'], axis=1),
    yticklabels=False,
    vmax=3,
    xticklabels=5
)
fig = plt.gcf()
fig.set_size_inches(12,8)
plt.xlabel('Days since installation')
plt.ylabel('Installations')


# Now we can see some very addicted users accessing almost ever day without fail for weeks on end! This sort of behaviour seems like the kind of training that would lead to good performance, vs. those that are coming back a lot in one day or randomly. This is because spaced repetition is much more powerful than compressed learning. Also on the hourly scale, (e.g. no more than 1 hour of sessions or it becomes less valuable).

# The sort of graphics above leads us to ask questions like how often is a user coming back? For this, we can calculate the frequency.

# ### Frequency
# We will calculate frequency of an installation as the average time between the start of sessions. It is undefined for users with just one session.

# In[ ]:


sessions.head()


# In[ ]:


sessions.loc['0006a69f', :]['seconds_since_installation']['min'].diff()


# In[ ]:


results = {}
for i in list(set([a[0] for a in sessions.index]))[:20000]:
    results[i] = [sessions.loc[i, :]['seconds_since_installation']['min'].diff().mean(),
                  sessions.loc[i, :]['seconds_since_installation']['min'].diff().median()]
frequencies = pd.DataFrame.from_dict(results, orient='index', columns=['mean', 'median'])     .apply(lambda x: x/60/60/24)
frequencies[frequencies['mean'].notnull()].head()


# In[ ]:


frequencies.plot(kind='box', sym='')
fig = plt.gcf()
fig.set_size_inches(8, 5)
plt.ylabel('Number of Days between Sessions')
plt.title('Frequency of Installations')


# We see that ignoring the users that have just one session, the number of days between sessions is ~5, though the third quartile (50%-75%) extends up to ~10 days.
# 
# This sort of frequency feature might be useful as a feature to predict performance on the games as it describes what sort of habit the user has.
