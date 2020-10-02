#!/usr/bin/env python
# coding: utf-8

# **[Feature Engineering Home Page](https://www.kaggle.com/learn/feature-engineering)**
# 
# ---
# 

# # Introduction
# 
# In this exercise you'll use some feature selection algorithms to improve your model. Some methods take a while to run, so you'll write functions and verify they work on small samples.
# 
# To begin, run the code cell below to set up the exercise.

# In[ ]:


# Set up code checking
get_ipython().system('pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git')
from learntools.core import binder
binder.bind(globals())
from learntools.feature_engineering.ex4 import *


# Then run the following cell. It takes a minute or so to run.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
import lightgbm as lgb

import os

clicks = pd.read_parquet('../input/feature-engineering-data/baseline_data.pqt')
data_files = ['count_encodings.pqt',
              'catboost_encodings.pqt',
              'interactions.pqt',
              'past_6hr_events.pqt',
              'downloads.pqt',
              'time_deltas.pqt',
              'svd_encodings.pqt']
data_root = '../input/feature-engineering-data'
for file in data_files:
    features = pd.read_parquet(os.path.join(data_root, file))
    clicks = clicks.join(features)

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
    print("Training model!")
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


# ## Baseline Score
# 
# Let's look at the baseline score for all the features we've made so far.

# In[ ]:


train, valid, test = get_data_splits(clicks)
_, baseline_score = train_model(train, valid)


# ### 1) Which data to use for feature selection?
# 
# Since many feature selection methods require calculating statistics from the dataset, should you use all the data for feature selection?
# 
# Run the following line after you've decided your answer.

# In[ ]:


# Check your answer (Run this code cell to receive credit!)
q_1.solution()


# Now we have 91 features we're using for predictions. With all these features, there is a good chance the model is overfitting the data. We might be able to reduce the overfitting by removing some features. Of course, the model's performance might decrease. But at least we'd be making the model smaller and faster without losing much performance.

# ### 2) Univariate Feature Selection
# 
# Below, use `SelectKBest` with the `f_classif` scoring function to choose 40 features from the 91 features in the data. 

# In[ ]:


from sklearn.feature_selection import SelectKBest, f_classif
feature_cols = clicks.columns.drop(['click_time', 'attributed_time', 'is_attributed'])
train, valid, test = get_data_splits(clicks)

# Do feature extraction on the training data only!
selector = SelectKBest(f_classif, k=40)
X_new = selector.fit_transform(train[feature_cols], train['is_attributed'])

# Get back the features we've kept, zero out all other features
selected_features = pd.DataFrame(selector.inverse_transform(X_new),
                                    index=train.index,
                                    columns=feature_cols)

# Dropped columns have values of all 0s, so var is 0, drop them
dropped_columns = selected_features.columns[selected_features.var() == 0]


# Check your answer
q_2.check()


# In[ ]:


# q_2.hint()
# q_2.solution()


# In[ ]:


_ = train_model(train.drop(dropped_columns, axis=1), 
                valid.drop(dropped_columns, axis=1))


# ### 3) The best value of K
# 
# With this method we can choose the best K features, but we still have to choose K ourselves. How would you find the "best" value of K? That is, you want it to be small so you're keeping the best features, but not so small that it's degrading the model's performance.
# 
# Run the following line after you've decided your answer.

# In[ ]:


# Check your answer (Run this code cell to receive credit!)
q_3.solution()


# ### 4) Use L1 regularization for feature selection
# 
# Now try a more powerful approach using L1 regularization. Implement a function `select_features_l1` that returns a list of features to keep.
# 
# Use a `LogisticRegression` classifier model with an L1 penalty to select the features. For the model, set:
# - the random state to 7,
# - the regularization parameter to 0.1,
# - and the solver to `'liblinear'`.
# 
# Fit the model then use `SelectFromModel` to return a model with the selected features.
# 
# The checking code will run your function on a sample from the dataset to provide more immediate feedback.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

def select_features_l1(X, y):
        logistic = LogisticRegression(C=0.1, penalty="l1", random_state=7, solver='liblinear').fit(X, y)
        model = SelectFromModel(logistic, prefit=True)

        X_new = model.transform(X)

        # Get back the kept features as a DataFrame with dropped columns as all 0s
        selected_features = pd.DataFrame(model.inverse_transform(X_new),
                                        index=X.index,
                                        columns=X.columns)

        # Dropped columns have values of all 0s, keep other columns
        cols_to_keep = selected_features.columns[selected_features.var() != 0]

        return cols_to_keep

# Check your answer
q_4.check()


# In[ ]:


# Uncomment these if you're feeling stuck
#q_4.hint()
#q_4.solution()


# In[ ]:


n_samples = 10000
X, y = train[feature_cols][:n_samples], train['is_attributed'][:n_samples]
selected = select_features_l1(X, y)

dropped_columns = feature_cols.drop(selected)
_ = train_model(train.drop(dropped_columns, axis=1), 
                valid.drop(dropped_columns, axis=1))


# ### 5) Feature Selection with Trees
# 
# Since we're using a tree-based model, using another tree-based model for feature selection might produce better results. What would you do different to select the features using a trees classifier?
# 
# Run the following line after you've decided your answer.

# In[ ]:


# Check your answer (Run this code cell to receive credit!)
q_5.solution()


# ### 6) Top K features with L1 regularization
# 
# Here you've set the regularization parameter `C=0.1` which led to some number of features being dropped. However, by setting `C` you aren't able to choose a certain number of features to keep. What would you do to keep the top K important features using L1 regularization?
# 
# Run the following line after you've decided your answer.

# In[ ]:


# Check your answer (Run this code cell to receive credit!)
q_6.solution()


# Congratulations on finishing this course! To keep learning, check out the rest of [our courses](https://www.kaggle.com/learn/overview). The machine learning explainability and deep learning courses are great next skills to learn!

# ---
# **[Feature Engineering Home Page](https://www.kaggle.com/learn/feature-engineering)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum/161443) to chat with other Learners.*
