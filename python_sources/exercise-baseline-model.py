#!/usr/bin/env python
# coding: utf-8

# **[Feature Engineering Home Page](https://www.kaggle.com/learn/feature-engineering)**
# 
# ---
# 

# # Introduction
# 
# In this exercise, you will develop a baseline model for predicting if a customer will buy an app after clicking on an ad. With this baseline model, you'll be able to see how your feature engineering and selection efforts improve the model's performance.

# In[ ]:


# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.feature_engineering.ex1 import *

import pandas as pd

click_data = pd.read_csv('../input/feature-engineering-data/train_sample.csv',
                         parse_dates=['click_time'])
click_data.head(10)


# ## Baseline Model
# 
# The first thing you need to do is construct a baseline model. All new features, processing, encodings, and feature selection should improve upon this baseline model. First you need to do a bit of feature engineering before training the model itself.
# 
# ### 1) Features from timestamps
# From the timestamps, create features for the day, hour, minute and second. Store these as new integer columns `day`, `hour`, `minute`, and `second` in a new DataFrame `clicks`.

# In[ ]:


# Add new columns for timestamp features day, hour, minute, and second
clicks = click_data.copy()
clicks['day'] = clicks['click_time'].dt.day.astype('uint8')
# Fill in the rest
clicks['hour'] = clicks['click_time'].dt.hour.astype('uint8')
clicks['minute'] = clicks['click_time'].dt.minute.astype('uint8')
clicks['second'] = clicks['click_time'].dt.second.astype('uint8')

q_1.check()


# In[ ]:


# Uncomment these if you need guidance
#q_1.hint()
#q_1.solution()


# ### 2) Label Encoding
# For each of the categorical features `['ip', 'app', 'device', 'os', 'channel']`, use scikit-learn's `LabelEncoder` to create new features in the `clicks` DataFrame. The new column names should be the original column name with `'_labels'` appended, like `ip_labels`.

# In[ ]:


from sklearn import preprocessing

cat_features = ['ip', 'app', 'device', 'os', 'channel']

# Create new columns in clicks using preprocessing.LabelEncoder()
encoder=preprocessing.LabelEncoder()
for feature in cat_features:
    encoded=encoder.fit_transform(clicks[feature])
    clicks[feature +'_labels']=encoded

q_2.check()


# In[ ]:


clicks.head()


# In[ ]:


# Uncomment these if you need guidance
# q_2.hint()
 #q_2.solution()


# In[ ]:


clicks.head()


# ### 3) One-hot Encoding
# 
# Now you have label encoded features, does it make sense to use one-hot encoding for the categorical variables ip, app, device, os, or channel?
# 
# Uncomment the following line after you've decided your answer.

# No it is not required since we have already label encoded them.

# In[ ]:


q_3.solution()


# ## Train, validation, and test sets
# With our baseline features ready, we need to split our data into training and validation sets. We should also hold out a test set to measure the final accuracy of the model.
# 
# ### 4) Train/test splits with time series data
# This is time series data. Are they any special considerations when creating train/test splits for time series? If so, what and why?
# 
# Uncomment the following line after you've decided your answer.

# Yes.The train and test set split for a time series data is different from the normal dataset split.The split should be in such a way that the data is trained for one month and tested for the next month.

# In[ ]:


q_4.solution()


# ### Create train/validation/test splits
# 
# Here we'll create training, validation, and test splits. First, `clicks` DataFrame is sorted in order of increasing time. The first 80% of the rows are the train set, the next 10% are the validation set, and the last 10% are the test set.

# In[ ]:


feature_cols = ['day', 'hour', 'minute', 'second', 
                'ip_labels', 'app_labels', 'device_labels',
                'os_labels', 'channel_labels']

valid_fraction = 0.1
clicks_srt = clicks.sort_values('click_time')
valid_rows = int(len(clicks_srt) * valid_fraction)
train = clicks_srt[:-valid_rows * 2]
# valid size == test size, last two sections of the data
valid = clicks_srt[-valid_rows * 2:-valid_rows]
test = clicks_srt[-valid_rows:]


# In[ ]:


train.shape,valid.shape,test.shape


# ### Train with LightGBM
# 
# Now we can create LightGBM dataset objects for each of the smaller datasets and train the baseline model.

# In[ ]:


import lightgbm as lgb

dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])
dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])
dtest = lgb.Dataset(test[feature_cols], label=test['is_attributed'])

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10)


# ## Evaluate the model
# Finally, with the model trained, I'll evaluate it's performance on the test set. 

# In[ ]:


from sklearn import metrics

ypred = bst.predict(test[feature_cols])
score = metrics.roc_auc_score(test['is_attributed'], ypred)
print(f"Test score: {score}")


# This will be our baseline score for the model. When we transform features, add new ones, or perform feature selection, we should be improving on this score. However, since this is the test set, we only want to look at it at the end of all our manipulations. At the very end of this course you'll look at the test score again to see if you improved on the baseline model.
# 
# # Keep Going
# Now that you have a baseline model, you are ready to learn **[Categorical Encoding Techniques](https://www.kaggle.com/matleonard/categorical-encodings)** to improve it.

# ---
# **[Feature Engineering Home Page](https://www.kaggle.com/learn/feature-engineering)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum) to chat with other Learners.*
