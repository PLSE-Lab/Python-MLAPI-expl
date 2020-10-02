#!/usr/bin/env python
# coding: utf-8

# Parts of the code are taken and modified from the following notebooks:
# 
# https://www.kaggle.com/erikbruin/data-science-bowl-2019-eda-and-baseline
# 
# https://www.kaggle.com/robikscube/2019-data-science-bowl-an-introduction
# 
# https://www.kaggle.com/mhviraf/a-new-baseline-for-dsb-2019-catboost-model
# 
# 

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
#from numba import jit
import lightgbm as lgb
#import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics
from itertools import product
import copy
import time

import random
seed = 42
random.seed(seed)
np.random.seed(seed)


# In[ ]:


#load training data, training_labels, event-specifications, testdata and sample submisson into pandas dataframes
train = pd.read_csv('../input/data-science-bowl-2019/train.csv')
train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
test = pd.read_csv('../input/data-science-bowl-2019/test.csv')
specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')


# Applied metric is quadratic weightes kappa:
# 
# https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps

# In[ ]:


from sklearn.metrics import confusion_matrix
def qwk(act,pred,n=4,hist_range=(0,3)):
    
    O = confusion_matrix(act,pred)
    O = np.divide(O,np.sum(O))
    
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            W[i][j] = ((i-j)**2)/((n-1)**2)
            
    act_hist = np.histogram(act,bins=n,range=hist_range)[0]
    prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]
    
    E = np.outer(act_hist,prd_hist)
    E = np.divide(E,np.sum(E))
    
    num = np.sum(np.multiply(W,O))
    den = np.sum(np.multiply(W,E))
        
    return 1-np.divide(num,den)


# # The Training Set 

# In[ ]:


train.head()


# The data provided in these files are as follows:
# - `event_id` - unique identifier for a single event. Infos about event are found in specs table.
# - `game_session` - unique identifier grouping events within a single game or video play session.
# - `timestamp` - datetime (local time as it seems)
# - `event_data` - Semi-structured JSON formatted string containing the events parameters. Default fields are: event_count, event_code, and game_time; otherwise - fields are determined by the event type.
# - `installation_id` - Randomly generated unique identifier grouping game sessions within a single installed application instance.
# - `event_count` - Incremental counter of events within a game session (offset at 1). Extracted from event_data.
# - `event_code` - Identifier of the event 'class'. Unique per game, but may be duplicated across games. E.g. event code '2000' always identifies the 'Start Game' event for all games. Extracted from event_data.
# - `game_time` - Time in milliseconds since the start of the game session. Extracted from event_data.
# - `title` - Title of the game or video.
# - `type` - Media type of the game or video. Possible values are: 'Game', 'Assessment', 'Activity', 'Clip'.
# - `world` - The section of the application the game or video belongs to. Helpful to identify the educational curriculum goals of the media. Possible values are: 'NONE' (at the app's start screen), TREETOPCITY' (Length/Height), 'MAGMAPEAK' (Capacity/Displacement), 'CRYSTALCAVES' (Weight).

# "installation_id" groups multiple "game_session"
# "game_session" groups multiple "event_id"
# 
# 
# "event_data" details an event
# 
# "event_count", "event_code" and "game_time" are extracted from "event_data"
# 
# "game time" is 0 for every start-event with code 2000
# 
# "game time" reflects total time for games and assessments win_codes 4100 (4110 for bird measurer)
# 
# 
# "type" helps to concentrate on samples we need to train the algorithm on and that we need to predict (= assessments)
# 
# 
# "world" is redundant :
# 
# --> "Cart Balancer" and "Chest Sorter" are in CRYSTALCAVES
# 
# --> "Cauldron Filler" is in MAGMAPEAK
# 
# --> "Bird Measurer" and "Mushroom Sorter" are in TREETOPCITY

# In[ ]:


train.shape


# There are over 11M lines of training data.
# 
# There are provided certain gaming histories for a number of installation ids. We need to predict the number of attempts of a future assessment with the knowledge of the gaming history.
# 
# From Kaggle we know that there are installation_id that have not completet any assessments

# How many installation ids are in the training data set?
# 
# How many of them did never do any assessments?

# In[ ]:


ids_all = train["installation_id"].drop_duplicates()
ids_with_assessments = train[train.type == "Assessment"]["installation_id"].drop_duplicates()


# In[ ]:


print("There are {} IDs".format(len(ids_all)))
print("There are {} IDs with assessment(s)".format(len(ids_with_assessments)))


# In[ ]:


train_clean = pd.merge(train, ids_with_assessments, on="installation_id", how="inner")


# In[ ]:


train_clean.shape


# There are 8M remaining lines, only containing IDs that did at least 1 assessment

# In[ ]:


fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(311)
ax1 = sns.countplot(y="type", data=train_clean, color="blue")
plt.title("number of events by type")

ax2 = fig.add_subplot(312)
ax2 = sns.countplot(y="world", data=train_clean, color="blue")
plt.title("number of events by world")

ax3 = fig.add_subplot(313)
ax3 = sns.countplot(y="title", data=train_clean[train_clean["type"] == "Assessment"], color="blue")
plt.title("number of evetns by assessment")

plt.tight_layout(pad=2)
plt.show()


# "timestamp" is of string-type
# 
# extract "date", "month", "hour", "dayofweek" to get an overview of app-activity
# 
# maybe helpful maybe not, at least we get an overview

# In[ ]:


def add_time_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df
    
train_clean = add_time_features(train_clean)
test = add_time_features(test)


# In[ ]:


train_clean.head()


# In[ ]:


fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(311)
ax1 = sns.countplot(x="date", data=train_clean, color="blue")
plt.title("number of events by date")

ax2 = fig.add_subplot(312)
ax2 = sns.countplot(x="dayofweek", data=train_clean, color="blue")
plt.title("number of events by day of week")

ax3 = fig.add_subplot(313)
ax3 = sns.countplot(x="hour", data=train_clean, color="blue")
plt.title("number of events by local daytime")

plt.tight_layout(pad=2)
plt.show()


# # The Test-Set

# In[ ]:


print("There are {} lines in sample submission".format(sample_submission.shape[0]))


# In[ ]:


print("There are {} unique installation_ids in testset".format(test["installation_id"].unique().shape[0]))


# So far, so good, for every installation_id in the testset there is exactly one line in the sample_submission

# # The labels

# In[ ]:


train_labels.head()


# In[ ]:


train_labels["title"].unique()


# There are 4 classes - accuracy_group
# 
# 3: solved on 1st attempt
# 
# 2: solved on 2nd attempt
# 
# 1: solved after 3 or more attempts
# 
# 0: never solved

# In[ ]:


label_stats = train_labels.groupby(['title', 'accuracy_group'])['accuracy_group'].count().unstack("title")
label_stats.plot.bar(stacked=True, figsize=(10,10))


# In[ ]:


trainable_ids = train_labels["installation_id"].drop_duplicates()
trainable_ids.shape


# There are 3614 assessment results, but there are 4242 installation_ids in the training set.
# This means that there are 628 installation_ids that are not trainable because of missing training-labels.
# --> remove all events that belong to any of these untrainable installation_ids

# In[ ]:


train_clean = pd.merge(train_clean, trainable_ids, on="installation_id", how="inner")
train_clean.shape


# There are 7,7M remaining events

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


train_clean.groupby('installation_id').count()['event_id'].plot(kind='hist', bins=200, figsize=(15, 5),
         title='Num Events by installation_id')
plt.show()


# In[ ]:


sessions_per_id = train_clean[["installation_id", "game_session"]].drop_duplicates().groupby(["installation_id"])["game_session"].count().plot(kind='hist', bins=200, figsize=(15, 5),
         title='Num Sessions by installation_id')


# In[ ]:


train_clean


# In[ ]:


train_clean.groupby('event_code').count()['event_id'].sort_values().plot(kind='bar', figsize=(15, 5),
         title='event code count.')
plt.show()


# there are lot of events with code 4070, 4030, 3010, 3110, 4020
# 
# are they relevant, or is their count relevant?

# # Some Feature Engineering

# In[ ]:


train_labels[train_labels["title"].str.contains("Assessment")]


# In[ ]:


print("Number of rows in train_labels: {}".format(train_labels.shape[0]))
print("Number of unique game_sessions in train_labels: {}".format(train_labels["game_session"].nunique()))
print("Number of unique game_sessions that are Assessments in train_labels: {}".format(train_labels[train_labels["title"].str.contains("Assessment")]["game_session"].nunique()))
print("Number of unique installation_ids in train_labels: {}".format(train_labels["installation_id"].nunique()))


# What to do:
# 
# There are certain game_sessions that are of type "assessment" for which we know the correct label.
# 
# We need to collect all features for such assessments that help to predict the labels for the testset

# ## Basic Feature Engineering & Cleaning

# In[ ]:


# generate a list of all unique titles (=activities), enumerate them and replace occurances by numbers
list_of_user_activities = list(set(train_clean['title'].unique()).union(set(test['title'].unique())))
activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))

train_clean['title'] = train_clean['title'].map(activities_map)
test['title'] = test['title'].map(activities_map)
train_labels['title'] = train_labels['title'].map(activities_map)


# In[ ]:


# create a dictionary that lists winning codes for all titles (activities)
# every activity has a win_code 4100, only bird-measurer has 4110
attempt_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
attempt_code[activities_map['Bird Measurer (Assessment)']] = 4110


# In[ ]:


train_clean


# In[ ]:


# user_sample contains lines with a unique installation_id

def get_data(user_sample, test_set=False):
    last_activity = 0
    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
    accuracy_groups = {0:0, 1:0, 2:0, 3:0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy=0
    accumulated_correct_attempts = 0 
    accumulated_incorrect_attempts = 0 
    accumulated_events = 0
    counter = 0
    durations = []
    
    for i, session in user_sample.groupby('game_session', sort=False):
        #session type and title is unique within each session
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        
        #create feature vector for (session_type == Assessment) if (Testdata or (Trainingdata & len>1))
        if (session_type == 'Assessment') & ((test_set == True) | ((test_set == False) & (len(session)>1))):
            #add statistics of past events to feature vector and then update statistics
            
            #start feature vector with minimum content (Clip/Activity/Assessment/Game)
            features = user_activities_count.copy()
            
            # add session_title (as a number)
            features['session_title'] = session['title'].iloc[0] 
            
            # get all attempts in current game_session
            all_attempts = session.query(f'event_code == {attempt_code[session_title]}')
            # get all "true" attempts in current game_session
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            # get all "false" attempts in current game_session
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            
            #compute accuracy of current game_session --> this is the parameter we need to predict later on
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
            
            #add accumulated correct attempts that occured before current session
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            #update accumulated attempt values
            accumulated_correct_attempts += true_attempts 
            #same with incorrect attempts
            features['accumulated_incorrect_attempts'] = accumulated_incorrect_attempts
            #update accumulated attempt values
            accumulated_incorrect_attempts += false_attempts

            #add mean duration of previous sessions
            if durations == []: #first session, durations is still empty
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            #add current duration to list, therefore use the timestamp
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds) #last timestamp - first timestamp

            
            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0
            #update accumulated_accuracy with accuracy of current session
            accumulated_accuracy += accuracy
            
            # add accuracy_groups to features
            features.update(accuracy_groups)
            # update accuracy groups with current game_session
            accuracy_groups[features['accuracy_group']] += 1
            
            
            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            
            
            features['accumulated_actions'] = accumulated_events
            if test_set == True:
                all_assessments.append(features)
            else:
                if true_attempts+false_attempts > 0:
                    all_assessments.append(features)
                
            counter += 1

        accumulated_events += len(session)
        if last_activity != session_type: #cout number of different consecutive user activities
            user_activities_count[session_type] += 1
            last_activitiy = session_type

    if test_set:
        return all_assessments[-1] #for test data return last assessment 
    return all_assessments #for trainingdata return every assessment


# In[ ]:


from tqdm import tqdm

compiled_data = []
#for i, (ins_id, user_sample) in tqdm(enumerate(train_clean.groupby('installation_id', sort=False))):
for (ins_id, user_sample) in tqdm(train_clean.groupby('installation_id')):
    compiled_data += get_data(user_sample)


# In[ ]:


new_train = pd.DataFrame(compiled_data)
del compiled_data
new_train.shape


# In[ ]:


new_train.head()


# # Generate and train the model

# In[ ]:


all_features = [x for x in new_train.columns if x not in ['accuracy_group']]
cat_features = ['session_title']
X, y = new_train[all_features], new_train['accuracy_group']
del train


# In[ ]:


X.shape


# Catboost Classifier
# 
# https://catboost.ai/docs/concepts/about.html

# In[ ]:


def make_classifier():
    clf = CatBoostClassifier(
                               loss_function='MultiClass',
                               task_type="CPU",
                               learning_rate=0.01,
                               iterations=2000,
                               od_type="Iter",
                               early_stopping_rounds=500,
                               random_seed=42
                              )        
    return clf


# In[ ]:


from time import time
from sklearn.model_selection import KFold

oof = np.zeros(len(X))
n_folds = 5
classifiers = []

folds = KFold(n_splits=n_folds, shuffle=True, random_state=42)
training_start_time = time()
for fold, (train_idx, validate_idx) in enumerate(folds.split(X, y)):
    start_time = time()
    print(f'Training on fold {fold+1}')
    clf = make_classifier()
    clf.fit(X.loc[train_idx, all_features], y.loc[train_idx], 
            eval_set=(X.loc[validate_idx, all_features], y.loc[validate_idx]),
            use_best_model=True, verbose=500, cat_features=cat_features)
    classifiers.append(clf)
    oof[validate_idx] = clf.predict(X.loc[validate_idx, all_features]).squeeze()
    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))

print('-' * 30)
print('QWK Score on Training Set using K-Fold CV:', qwk(y, oof))
print('-' * 30)


# In[ ]:


# process test set
new_test = []
for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False)):
    a = get_data(user_sample, test_set=True)
    new_test.append(a)
    
X_test = pd.DataFrame(new_test)
#del test


# In[ ]:


# make predictions on test set once
preds_proba = np.zeros((X_test.shape[0], 4))

for classifier in classifiers:
    preds_proba += classifier.predict_proba(X_test)/len(classifiers)
preds = np.argmax(preds_proba, axis=1)
#del X_test


# # Make submission

# In[ ]:


sample_submission['accuracy_group'] = np.round(preds).astype('int')
sample_submission.to_csv('submission.csv', index=None)
sample_submission.head()


# In[ ]:


sample_submission['accuracy_group'].plot(kind='hist')


# In[ ]:


train_labels['accuracy_group'].plot(kind='hist')

