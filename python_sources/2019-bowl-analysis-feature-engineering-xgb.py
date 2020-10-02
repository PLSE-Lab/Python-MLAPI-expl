#!/usr/bin/env python
# coding: utf-8

# # Welcome to the Data Science Bowl Competition
# This notebook is a starter code for all beginners and easy to understand. <br>
# We focus on
# * a simple analysis of the data,
# * create new features,
# * encoding and 
# * scale data,
# * prepare data for train.
# 
# We use categorical feature encoding techniques, compare <br>
# https://www.kaggle.com/drcapa/categorical-feature-encoding-challenge-xgb
# 
# In this kernel we consider the train data. For prediction we must repeate all operations also for the test data. <br>
# After that we define X_train and y_train.
# The aim of the competition is to predict the target accuracy_group:
# * 3: the assessment was solved on the first attempt,
# * 2: the assessment was solved on the second attempt,
# * 1: the assessment was solved after 3 or more attempts,
# * 0: the assessment was never solved.
# 
# To predict the test data we use a XGB Classifier.

# # Load Libraries

# In[ ]:


import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
import os
import json


# In[ ]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


# In[ ]:


from xgboost import XGBClassifier


# # Show files in path

# In[ ]:


path_in = '../input/data-science-bowl-2019/'
os.listdir(path_in)


# # Load Train Data
# There is a column with a datetime information. So we can load as datetime type by parse_dates=['timestamp'].

# In[ ]:


train_data = pd.read_csv(path_in+'train.csv', parse_dates=['timestamp'])
train_labels = pd.read_csv(path_in+'train_labels.csv')
specs_data = pd.read_csv(path_in+'specs.csv')


# # Help Function

# In[ ]:


def plot_bar(data, name, width, lenght):
    fig = plt.figure(figsize=(width, lenght))
    ax = fig.add_subplot(111)
    data_label = data[name].value_counts()
    dict_train = dict(zip(data_label.keys(), ((data_label.sort_index())).tolist()))
    names = list(dict_train.keys())
    values = list(dict_train.values())
    plt.bar(names, values)
    ax.set_xticklabels(names, rotation=45)
    plt.grid()
    plt.show()


# # Analysis & Overview
# First we do a simple analysis and show important kpis.

# In[ ]:


print('# samples train_data:', len(train_data))
print('# samples train_labels:', len(train_labels))
print('# samples specs:', len(specs_data))


# In[ ]:


train_data.head()


# In[ ]:


train_labels.head()


# In[ ]:


specs_data.head()


# # Missing data
# 
# Fortunately there are no missing data we have to deal.

# In[ ]:


cols_with_missing_train_data = [col for col in train_data.columns if train_data[col].isnull().any()]
cols_with_missing_train_labels = [col for col in train_labels.columns if train_labels[col].isnull().any()]
cols_with_missing_specs_data = [col for col in specs_data.columns if specs_data[col].isnull().any()]


# In[ ]:


print(cols_with_missing_train_data)
print(cols_with_missing_train_labels)
print(cols_with_missing_specs_data)


# # Feature engineering
# There are 3 keys:
# 
# 1) game_session, installation_id: to merge train data and train labels
# 
# 2) event_id: to merge train data and specs 
# 
# For the idea of the feature engineering we reduce the train data and use a subset. 

# In[ ]:


#train_data = train_data.loc[0: len(train_data.index)/233]


# ## Train data

# In[ ]:


train_data.columns


# In[ ]:


train_data.dtypes


# ### Feature event_id
# Randomly generated unique identifier for the event type. <br>
# There are dublictated codes.

# In[ ]:


train_data['event_id'].value_counts()


# ### Feature game_session
# Randomly generated unique identifier grouping events within a single game or video play session. <br>
# There are dublictated codes.

# In[ ]:


train_data['game_session'].value_counts()


# ### Feature timestamp
# Is the client-generated datetime. You can extract new features like month, day or hour. These are cyclic features which can be encoded. Additionally we create the feature weekend: 5 = saturday and 6 = sunday.

# In[ ]:


train_data['month'] = train_data['timestamp'].dt.month
train_data['day'] = train_data['timestamp'].dt.weekday
train_data['hour'] = train_data['timestamp'].dt.hour
train_data['weekend'] = np.where((train_data['day'] == 5) | (train_data['day'] == 6), 1, 0)


# In[ ]:


features_cyc = {'month' : 12, 'day' : 7, 'hour' : 24}
for feature in features_cyc.keys():
    train_data[feature+'_sin'] = np.sin((2*np.pi*train_data[feature])/features_cyc[feature])
    train_data[feature+'_cos'] = np.cos((2*np.pi*train_data[feature])/features_cyc[feature])
train_data = train_data.drop(features_cyc.keys(), axis=1)


# ### Feature event_data
# Semi-structured JSON formatted string containing the events parameters. Default fields are: event_count, event_code, and game_time; otherwise fields are determined by the event type. <br>
# Next we show how to encode the features of a dictionary. 
# Without loss of generality we consider only the feature *description*.
# If you don't want to extract new features you can delete the column.

# In[ ]:


encode_fields = ['description']
# steps = 233
# for i in range(steps):
#     print('work on step: ', i+1)
#     for encode_field in encode_fields:
#         slice_from = i*len(train_data.index)/steps
#         slice_to = (i+1)*len(train_data.index)/steps-1
#         train_data.loc[slice_from:slice_to, encode_field] = train_data.loc[slice_from:slice_to, 'event_data'].apply(json.loads).apply(pd.Series)[encode_field]
del train_data['event_data']


# ### Feature installation_id
# The installation_id will typically correspond to one child.
# It is dandomly generated unique identifier grouping game sessions within a single installed application instance.

# ### Feature event_count
#  Incremental counter of events within a game session (offset at 1). Extracted from event_data.

# ### Feature event_code
# Identifier of the event 'class'. Unique per game, but may be duplicated across games. E.g. event code '2000' always identifies the 'Start Game' event for all games. Extracted from event_data.

# ### Feature game_time
# Time in milliseconds since the start of the game session. Extracted from event_data.

# ### Feature title
# Title of the game or video. The feature title is a categorical feature with lot of categories. For the first we use a simple mapping.

# In[ ]:


plot_bar(train_data, 'title', 30, 5)


# In[ ]:


map_train_title = dict(zip(train_data['title'].value_counts().sort_index().keys(),
                     range(1, len(train_data['title'].value_counts())+1)))


# In[ ]:


train_data['title'] = train_data['title'].replace(map_train_title)


# ### Feature type
# Media type of the game or video. Possible values are: 'Game', 'Assessment', 'Activity', 'Clip'. This is a categorical feature which we encode by one hot encoding technique.

# In[ ]:


plot_bar(train_data, 'type', 9, 5)


# In[ ]:


train_data = pd.get_dummies(train_data, columns=['type'])


# ### Feature world
# The section of the application the game or video belongs to. Helpful to identify the educational curriculum goals of the media. Possible values are: 'NONE' (at the app's start screen), TREETOPCITY' (Length/Height), 'MAGMAPEAK' (Capacity/Displacement), 'CRYSTALCAVES' (Weight). This is a categorical feature which we encode by one hot encoding technique.

# In[ ]:


plot_bar(train_data, 'world', 9, 5)


# In[ ]:


train_data = pd.get_dummies(train_data, columns=['world'])


# ## Train labels
# This dataset demonstrates how to compute the ground truth for the assessments in the training set.

# In[ ]:


train_labels.columns


# ### Feature game_session
#  Randomly generated unique identifier grouping events within a single game or video play session.

# ### Feature installation_id
# Randomly generated unique identifier grouping game sessions within a single installed application instance.

# ### Feature title
# The feature title is a categorical feature. For the first we use a simple mapping.

# In[ ]:


plot_bar(train_labels, 'title', 9, 5)


# In[ ]:


map_label_title = dict(zip(train_labels['title'].value_counts().sort_index().keys(),
                     range(1, len(train_labels['title'].value_counts())+1)))


# In[ ]:


train_labels['title'] = train_labels['title'].replace(map_label_title)


# ### Feature num_correct
# This is a binary feature we can use without modification. 

# In[ ]:


train_labels['num_correct'].value_counts()


# ### Feature num_incorrect
# This is a numerical feature.

# In[ ]:


#train_labels['num_incorrect'].value_counts()


# ### Feature accuracy
# This is a float fearure.

# In[ ]:


train_labels['accuracy'].describe()


# ### Feature accuracy_group
# This is the target we have to predict.

# In[ ]:


plot_bar(train_labels, 'accuracy_group', 8, 4)


# In[ ]:


train_labels['accuracy_group'].value_counts().sort_index()


# ## Specs
# This file gives the specification of the various event types.

# In[ ]:


specs_data.columns


# ### Feature event_id
# Global unique identifier for the event type.

# In[ ]:


specs_data['event_id']


# ### Feature info
# Description of the event. There are 168 different types of informations.

# In[ ]:


specs_data['info'].value_counts()


# ### Feature args
# JSON formatted string of event arguments. Each argument contains:
# * name - Argument name.
# * type - Type of the argument (string, int, number, object, array).
# * info - Description of the argument.
# 
# So what can we do with the information?

# In[ ]:


specs_data.loc[0, 'args']


# # Merge data
# For the first step we only merge the train_data with the train_label by the key . 

# In[ ]:


train_data = pd.merge(train_data, train_labels,  how='right', on=['game_session','installation_id'])


# # Define X_train and y_train
# The featrue accuracy_group is the target which is to predict.

# In[ ]:


no_features = ['accuracy_group', 'event_id', 'game_session', 'timestamp','installation_id',
              'accuracy', 'num_correct', 'num_incorrect']
X_train = train_data[train_data.columns.difference(no_features)].copy(deep=False)
y_train = train_data['accuracy_group']

del X_train['title_y']
X_train = X_train.rename(columns = {'title_x': 'title'})


# In[ ]:


len(X_train.index), len(train_data.index)


# In[ ]:


X_train.head()


# In[ ]:


del train_data


# # Define XGB Classifier
# For the first step we use the XGB Classifier.

# In[ ]:


model = XGBClassifier(objective ='multi:softmax',
                      learning_rate = 0.2,
                      max_depth = 16,
                      n_estimators = 350,
                      random_state=2019,
                      num_class = 4)
model.fit(X_train,y_train)


# In[ ]:


del X_train, y_train


# # Load Test Data

# In[ ]:


test_data = pd.read_csv(path_in+'test.csv', parse_dates=['timestamp'])
samp_subm = pd.read_csv(path_in+'sample_submission.csv')


# # Prepare Test Data
# We repeat the data preparation of the train set.

# In[ ]:


""" Extract new features from timestamp """
test_data['month'] = test_data['timestamp'].dt.month
test_data['day'] = test_data['timestamp'].dt.weekday
test_data['hour'] = test_data['timestamp'].dt.hour
test_data['weekend'] = np.where((test_data['day'] == 5) | (test_data['day'] == 6), 1, 0)

""" Encode cyclic features """
features_cyc = {'month' : 12, 'day' : 7, 'hour' : 24}
for feature in features_cyc.keys():
    test_data[feature+'_sin'] = np.sin((2*np.pi*test_data[feature])/features_cyc[feature])
    test_data[feature+'_cos'] = np.cos((2*np.pi*test_data[feature])/features_cyc[feature])
test_data = test_data.drop(features_cyc.keys(), axis=1)

""" Encode feature title """
test_data['title'] = test_data['title'].replace(map_train_title)

""" Encode feature type """
test_data = pd.get_dummies(test_data, columns=['type'])

""" Encode feature world """
test_data = pd.get_dummies(test_data, columns=['world'])

""" Delete feature event_data """
del test_data['event_data']


# # Define X_test

# In[ ]:


X_test = test_data[test_data.columns.difference(no_features)].copy(deep=False)


# # Predict y_test

# In[ ]:


y_test = model.predict(X_test)


# # Prepare y_test for output
# We group the results by the installation_id and take the most frequent accuracy_group.

# In[ ]:


y_temp = pd.DataFrame(y_test, index=test_data['installation_id'], columns=['accuracy_group'])


# In[ ]:


y_temp_grouped = y_temp.groupby(y_temp.index).agg(lambda x:x.value_counts().index[0])


# # Write output

# In[ ]:


output = pd.DataFrame({'installation_id': y_temp_grouped.index,
                       'accuracy_group': y_temp_grouped['accuracy_group']})
output.index = samp_subm.index
output.to_csv('submission.csv', index=False)


# In[ ]:


output.head()


# In[ ]:


output['accuracy_group'].value_counts().sort_index()

