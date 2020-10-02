#!/usr/bin/env python
# coding: utf-8

# **[Feature Engineering Home Page](https://www.kaggle.com/learn/feature-engineering)**
# 
# ---
# 

# # Introduction
# 
# In this exercise you'll apply more advanced encodings to encode the categorical variables ito improve your classifier model. The encodings you will implement are:
# 
# - Count Encoding
# - Target Encoding
# - CatBoost Encoding
# 
# You'll refit the classifier after each encoding to check its performance on hold-out data. 
# 
# Begin by running the next code cell to set up the notebook.

# In[ ]:


# Set up code checking
# This can take a few seconds
from learntools.core import binder
binder.bind(globals())
from learntools.feature_engineering.ex2 import *


# The next code cell repeats the work that you did in the previous exercise.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
import lightgbm as lgb

clicks = pd.read_parquet('../input/feature-engineering-data/baseline_data.pqt')


# Next, we define a couple functions that you'll use to test the encodings that you implement in this exercise.

# In[ ]:


def get_data_splits(dataframe, valid_fraction=0.1):
    """Splits a dataframe into train, validation, and test sets.

    First, orders by the column 'click_time'. Set the size of the 
    validation and test sets with the valid_fraction keyword argument.
    """

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


# Run this cell to get a baseline score. 

# In[ ]:


print("Baseline model")
train, valid, test = get_data_splits(clicks)
_ = train_model(train, valid)


# ### 1) Categorical encodings and leakage
# 
# These encodings are all based on statistics calculated from the dataset like counts and means. 
# 
# Considering this, what data should you be using to calculate the encodings?  Specifically, can you use the validation data?  Can you use the test data?
# 
# Run the following line after you've decided your answer.

# In[ ]:


# Check your answer (Run this code cell to receive credit!)
q_1.solution()


# ### 2) Count encodings
# 
# Begin by running the next code cell to get started.

# In[ ]:


import category_encoders as ce

cat_features = ['ip', 'app', 'device', 'os', 'channel']
train, valid, test = get_data_splits(clicks)


# Next, encode the categorical features `['ip', 'app', 'device', 'os', 'channel']` using the count of each value in the data set. 
# - Using `CountEncoder` from the `category_encoders` library, fit the encoding using the categorical feature columns defined in `cat_features`. 
# - Then apply the encodings to the train and validation sets, adding them as new columns with names suffixed `"_count"`.

# In[ ]:


# Create the count encoder
import category_encoders as ce
count_enc = ce.CountEncoder(cols=cat_features)

    # Learn encoding from the training set
count_enc.fit(train[cat_features])

    # Apply encoding to the train and validation sets
train_encoded = train.join(count_enc.transform(train[cat_features]).add_suffix('_count'))
valid_encoded = valid.join(count_enc.transform(valid[cat_features]).add_suffix('_count'))

# Check your answer
q_2.check()


# In[ ]:


# Uncomment if you need some guidance
# q_2.hint()
q_2.solution()


# Run the next code cell to see how count encoding changes the results.

# In[ ]:


# Train the model on the encoded datasets
# This can take around 30 seconds to complete
_ = train_model(train_encoded, valid_encoded)


# Count encoding improved our model's score!

# ### 3) Why is count encoding effective?
# At first glance, it could be surprising that count encoding helps make accurate models. 
# Why do you think is count encoding is a good idea, or how does it improve the model score?
# 
# Run the following line after you've decided your answer.

# In[ ]:


# Check your answer (Run this code cell to receive credit!)
q_3.solution()


# ### 4) Target encoding
# 
# Here you'll try some supervised encodings that use the labels (the targets) to transform categorical features. The first one is target encoding. 
# - Create the target encoder from the `category_encoders` library. 
# - Then, learn the encodings from the training dataset, apply the encodings to all the datasets, and retrain the model.

# In[ ]:


# Create the target encoder. You can find this easily by using tab completion.
# Start typing ce. the press Tab to bring up a list of classes and functions.
target_enc = ce.TargetEncoder(cols=cat_features)

    # Learn encoding from the training set
target_enc.fit(train[cat_features], train['is_attributed'])

    # Apply encoding to the train and validation sets
train_encoded = train.join(target_enc.transform(train[cat_features]).add_suffix('_target'))
valid_encoded = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_target'))

# Check your answer
q_4.check()


# In[ ]:


# Uncomment these if you need some guidance
#q_4.hint()
q_4.solution()


# Run the next cell to see how target encoding affects your results.

# In[ ]:


_ = train_model(train_encoded, valid_encoded)


# ### 5) Try removing IP encoding
# 
# If you leave `ip` out of the encoded features and retrain the model with target encoding, you should find that the score increases and is above the baseline score! Why do you think the score is below baseline when we encode the IP address but above baseline when we don't?
# 
# Run the following line after you've decided your answer.

# In[ ]:


# Check your answer (Run this code cell to receive credit!)
q_5.solution()


# ### 6) CatBoost Encoding
# 
# The CatBoost encoder is supposed to work well with the LightGBM model. Encode the categorical features with `CatBoostEncoder` and train the model on the encoded data again.

# In[ ]:


# Remove IP from the encoded features
cat_features = ['app', 'device', 'os', 'channel']

# Create the CatBoost encoder
    # Have to tell it which features are categorical when they aren't strings
cb_enc = ce.CatBoostEncoder(cols=cat_features, random_state=7)

    # Learn encoding from the training set
cb_enc.fit(train[cat_features], train['is_attributed'])

    # Apply encoding to the train and validation sets
train_encoded = train.join(cb_enc.transform(train[cat_features]).add_suffix('_cb'))
valid_encoded = valid.join(cb_enc.transform(valid[cat_features]).add_suffix('_cb'))
# Check your answer
q_6.check()


# In[ ]:


# Uncomment these if you need some guidance
#q_6.hint()
q_6.solution()


# Run the next code cell to see how the CatBoost encoder changes your results.

# In[ ]:


_ = train_model(train_encoded, valid_encoded)


# # Keep Going
# 
# Now you are ready to **[generate completely new features](https://www.kaggle.com/matleonard/feature-generation)** from the data.

# ---
# **[Feature Engineering Home Page](https://www.kaggle.com/learn/feature-engineering)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum/161443) to chat with other Learners.*
