#!/usr/bin/env python
# coding: utf-8

# **[Feature Engineering Home Page](https://www.kaggle.com/learn/feature-engineering)**
# 
# ---
# 

# # Introduction
# 
# In this set of exercises, you'll create new features from the existing data. Again you'll compare the score lift for each new feature compared to a baseline model. First off, run the cells below to set up a baseline dataset and model.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
import lightgbm as lgb

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.feature_engineering.ex3 import *

# Create features from   timestamps
click_data = pd.read_csv('../input/feature-engineering-data/train_sample.csv', 
                         parse_dates=['click_time'])
click_times = click_data['click_time']
clicks = click_data.assign(day=click_times.dt.day.astype('uint8'),
                           hour=click_times.dt.hour.astype('uint8'),
                           minute=click_times.dt.minute.astype('uint8'),
                           second=click_times.dt.second.astype('uint8'))

# Label encoding for categorical features
cat_features = ['ip', 'app', 'device', 'os', 'channel']
for feature in cat_features:
    label_encoder = preprocessing.LabelEncoder()
    clicks[feature] = label_encoder.fit_transform(clicks[feature])
    
def get_data_splits(dataframe, valid_fraction=0.1):

    dataframe = dataframe.sort_values('click_time')
    valid_rows = int(len(dataframe) * valid_fraction)
    train = dataframe[:-valid_rows * 2]
    # valid size == test size, last two sections of the data
    valid = dataframe[-valid_rows * 2:-valid_rows]
    test = dataframe[-valid_rows:]
    
    return train, valid, test

def train_model(train, valid, test=None, feature_cols=None):
    if feature_cols is None:
        feature_cols = train.columns.drop(['click_time', 'attributed_time',
                                           'is_attributed'])
    dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])
    
    param = {'num_leaves': 64, 'objective': 'binary', 
             'metric': 'auc', 'seed': 7}
    num_round = 1000
    print("Training model. Hold on a minute to see the validation score")
    bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], 
                    early_stopping_rounds=20, verbose_eval=False)
    
    valid_pred = bst.predict(valid[feature_cols])
    valid_score = metrics.roc_auc_score(valid['is_attributed'], valid_pred)
    print(f"Validation AUC score: {valid_score}")
    
    if test is not None: 
        test_pred = bst.predict(test[feature_cols])
        test_score = metrics.roc_auc_score(test['is_attributed'], test_pred)
        return bst, valid_score, test_score
    else:
        return bst, valid_score

print("Baseline model score")
train, valid, test = get_data_splits(clicks)
_ = train_model(train, valid)


# ### 1) Add interaction features
# 
# Here you'll add interaction features for each pair of categorical features (ip, app, device, os, channel). The easiest way to iterate through the pairs of features is with `itertools.combinations`. For each new column, join the values as strings with an underscore, so 13 and 47 would become `"13_47"`. As you add the new columns to the dataset, be sure to label encode the values.

# In[ ]:


import itertools

cat_features = ['ip', 'app', 'device', 'os', 'channel']
interactions = pd.DataFrame(index=clicks.index)

# Iterate through each pair of features, combine them into interaction features
for col1, col2 in itertools.combinations(cat_features, 2):
        new_col_name = '_'.join([col1, col2])
        
        # Convert to strings and combine
        new_values = clicks[col1].map(str) + "_" + clicks[col2].map(str)

        encoder = preprocessing.LabelEncoder()
        interactions[new_col_name] = encoder.fit_transform(new_values)

# Check your answer
q_1.check()


# In[ ]:


# Uncomment if you need some guidance
#q_1.hint()
#q_1.solution()


# In[ ]:


clicks = clicks.join(interactions)
print("Score with interactions")
train, valid, test = get_data_splits(clicks)
_ = train_model(train, valid)


# # Generating numerical features
# 
# Adding interactions is a quick way to create more categorical features from the data. It's also effective to create new numerical features, you'll typically get a lot of improvement in the model. This takes a bit of brainstorming and experimentation to find features that work well.
# 
# For these exercises I'm going to have you implement functions that operate on Pandas Series. It can take multiple minutes to run these functions on the entire data set so instead I'll provide feedback by running your function on a smaller dataset.

# ### 2) Number of events in the past six hours
# 
# The first feature you'll be creating is the number of events from the same IP in the last six hours. It's likely that someone who is visiting often will download the app.
# 
# Implement a function `count_past_events` that takes a Series of click times (timestamps) and returns another Series with the number of events in the last six hours. **Tip:** The `rolling` method is useful for this.

# In[ ]:


def count_past_events(series, time_window='6H'):
        series = pd.Series(series.index, index=series)
        # Subtract 1 so the current event isn't counted
        past_events = series.rolling(time_window).count() - 1
        return past_events

# Check your answer
q_2.check()


# In[ ]:


# Uncomment if you need some guidance
#q_2.hint()
#q_2.solution()


# Because this can take a while to calculate on the full data, we'll load pre-calculated versions in the cell below to test model performance.

# In[ ]:


# Loading in from saved Parquet file
past_events = pd.read_parquet('../input/feature-engineering-data/past_6hr_events.pqt')
clicks['ip_past_6hr_counts'] = past_events

train, valid, test = get_data_splits(clicks)
_ = train_model(train, valid)


# ### 3) Features from future information
# 
# In the last exercise you created a feature that looked at past events. You could also make features that use information from events in the future. Should you use future events or not? 
# 
# Run the following line after you've decided your answer.

# In[ ]:


# Check your answer (Run this code cell to receive credit!)
q_3.solution()


# ### 4) Time since last event
# 
# Implement a function `time_diff` that calculates the time since the last event in seconds from a Series of timestamps. This will be ran like so:
# 
# ```python
# timedeltas = clicks.groupby('ip')['click_time'].transform(time_diff)
# ```

# In[ ]:


def time_diff(series):
            return series.diff().dt.total_seconds()

# Check your answer
q_4.check()


# In[ ]:


# Uncomment if you need some guidance
#q_4.hint()
#q_4.solution()


# We'll again load pre-computed versions of the data, which match what your function would return

# In[ ]:


# Loading in from saved Parquet file
past_events = pd.read_parquet('../input/feature-engineering-data/time_deltas.pqt')
clicks['past_events_6hr'] = past_events

train, valid, test = get_data_splits(clicks.join(past_events))
_ = train_model(train, valid)


# ### 5) Number of previous app downloads
# 
# It's likely that if a visitor downloaded an app previously, it'll affect the likelihood they'll download one again. Implement a function `previous_attributions` that returns a Series with the number of times an app has been downloaded (`'is_attributed' == 1`) before the current event.

# In[ ]:


def previous_attributions(series):
    # Subtracting raw values so I don't count the current event
    sums = series.expanding(min_periods=2).sum() - series
    return sums

# Check your answer
q_5.check()


# In[ ]:


#q_5.hint()
#q_5.solution()


# Again loading pre-computed data.

# In[ ]:


# Loading in from saved Parquet file
past_events = pd.read_parquet('../input/feature-engineering-data/downloads.pqt')
clicks['ip_past_6hr_counts'] = past_events

train, valid, test = get_data_splits(clicks)
_ = train_model(train, valid)


# ### 6) Tree-based vs Neural Network Models
# 
# So far we've been using LightGBM, a tree-based model. Would these features we've generated work well for neural networks as well as tree-based models?
# 
# Run the following line after you've decided your answer.

# In[ ]:


# Check your answer (Run this code cell to receive credit!)
q_6.solution()


# Now that you've generated a bunch of different features, you'll typically want to remove some of them to reduce the size of the model and potentially improve the performance. Next, I'll show you how to do feature selection using a few different methods such as L1 regression and Boruta.

# # Keep Going
# 
# You know how to generate a lot of features. In practice, you'll frequently want to pare them down for modeling. Learn to do that in the **[Feature Selection lesson](https://www.kaggle.com/matleonard/feature-selection)**.

# ---
# **[Feature Engineering Home Page](https://www.kaggle.com/learn/feature-engineering)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum/161443) to chat with other Learners.*
