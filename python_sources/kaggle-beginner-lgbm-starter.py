#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This note book should serve as a guide for setting up a basic pipeline for a kaggle competition. It walks through the basic steps of simple EDA, feature engineering, CV setup, model setup, submissions, and working in the kaggle environment.

# ## Setup
# Adding competition datasources is easy! On the right toolbar, click the add data button, navigate to the 'competitions' tab, and select the relevant competition. You can also use this feature to add data from other public kernels. These files can be accessed via a relative path from your notebook/script in kaggle.

# In[ ]:


import pandas as pd
import numpy as np
import time
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns


import os
print(os.listdir("../input")) # Any results you write to the current directory are saved as output.


# ## Data Exploration
# 
# For the purpose of this kernel, let's just look at the core data set in train.csv and test.csv

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
submit_df = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


print(train_df.shape)
train_df.head()


# In[ ]:


print(test_df.shape)
test_df.head()


# ### Some information to check
# 
# * are card_ids unique?
# * what do the 'feature' distributions look like?
# * How does train differ from test?

# In[ ]:


# it'll be useful to combine the train and test data into a single dataframe to see how they differ

df = pd.concat([train_df, test_df], sort=False)
df['is_train'] = df['target'].notnull() 
df['first_active_month'] = df['first_active_month'].apply(pd.to_datetime)
df.sample(4)


# In[ ]:


# card ids are indeed unique identifiers
print(df.card_id.nunique() / df.shape[0])


# In[ ]:


# lets look at the first_active_month column. I suspect that train and test differ greatly
def df_pivot_distribution(df, index, split='is_train', count_on='card_id'):
    return df.groupby([index, split]).count()[count_on].reset_index().pivot(index=index, columns=split, values=count_on)#.plot.bar(figsize=(20,6), stacked=True)

df_pivot_distribution(df, 'first_active_month').plot.bar(stacked=True, figsize=(15, 4))


# Interestingly, the train and test set actually seem to split evenly around the dates. The majority of these dates occur in 2017.

# So, what do the mysterious "features" looks like?

# In[ ]:


df_pivot_distribution(df, 'feature_1').plot.bar()
df_pivot_distribution(df, 'feature_2').plot.bar()
df_pivot_distribution(df, 'feature_3').plot.bar()


# There isn't too much to infer from this, since these features are already engineered and anonymous, but it's useful to know that the train and test set don't differ too much in their populations. 

# In[ ]:


df['target'].describe()


# ## Feature Engineering
# 
# Feature engineering tends to be the most critical and time consuming process in these competitions. For this demo, since we're only using a small subset of the data, it'll be a bit simple. 
# 
# Essentially, the end goal of this process is to have a dataframe, indexed by card_id, with a bunch of columns of data that are expected to help predict *customer loyalty*
# 
# The role of the **model** is to **learn** from this data that has a non-null **target** field to predict the loyalty score for those in the test data

# In[ ]:


# since the columns labeled 'features' are already engineered, let's try to get useful information from the date field

df['first_active_mo'] = df['first_active_month'].dt.month
df['first_active_yr'] = df['first_active_month'].dt.year


# In[ ]:


# Our 'feature' features appear to be categorical - rather than actually numeric. We may get clearer signal if we
# don't treat them as sequential

cat_columns = ['feature_1', 'feature_2', 'feature_3', 'first_active_yr']
df = pd.get_dummies(df, columns=cat_columns)


# In[ ]:


df.head()


# ## Modeling and Cross-validation
# 
# At this point modeling can be relatively straight forward since our data is organized pretty well into train and test, and since our features have been developed. However, it is equally important to develop a technique for checking how accurate your model is at predicting the target. 
# 
# In kaggle competitions we can make submissions. However public submissions are only validated against a small portion of the test set. Additionally, since this test set is static, it is easy to fall into the trap of overfitting your model to the public submission set. Lastly, kaggle submissions are limited to ~5 per day, so measuring how small changes affect your model isn't really possible. This is why we set up cross-validation as part of our pipeline.
# 
# Cross validation is simply a technique for using your existing data to measure the performance of your model. The most basic way of doing this is simply splitting up your training data into 80% for training and 20% into validating it. The problem with this method is that we lose 20% of our training that could've helped us improve our model. One potential solution for this is to use **K-fold** validation.
# 
# The strategy is pretty simple. We split our training data into K *folds* (in this example we split it into fifths). For each fold, we train the model on the other four folds and predict on the fifth. This allows us train and validate over the entire dataset. The test set predictions are made from the average of the five models we just made.
# 
# K-Folds isn't always the best choice for cross-validation, but it's a strong place to start. One potential weakness of it is that it ignore any natural pattern that your train and test set follow. If you know that all of your test data occurs 6 months after your training data, than your K-fold tested model may not generalize well to the test set.
# 
# <img src="https://i.stack.imgur.com/1fXzJ.png" width="800px"/> 
# image from https://www.kaggle.com/dansbecker/cross-validation
# 
# 
# The code below runs through a simple 5 fold K-Fold and uses a gradient boosting model that tends to be popular in kaggle competitions. For more on the model, the documentation can ve found at https://lightgbm.readthedocs.io/en/latest/
# 

# In[ ]:


# params taken from public kernel https://www.kaggle.com/peterhurford/you-re-going-to-want-more-categories-lb-3-737
param = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.0041,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1}

# seperate our train, test, and targets
train_df = df.loc[df['is_train'] == True].set_index('card_id')
test_df = df.loc[df['is_train'] == False].set_index('card_id')
target = train_df['target']

# we only want to pass our import numeric features to the model
ignored_columns = ['first_active_month', 'target', 'is_train']
train_df = train_df[[ix for ix in train_df.columns if ix not in ignored_columns]]
test_df = test_df[[ix for ix in test_df.columns if ix not in ignored_columns]]

#initalize our outputs and our folds
folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
start = time.time()
features = list(train_df.columns)
feature_importance_df = pd.DataFrame()

# loop through the folds, train the model, and make a prediction
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    trn_data = lgb.Dataset(train_df.iloc[trn_idx].values, label=target.iloc[trn_idx].values)
    val_data = lgb.Dataset(train_df.iloc[val_idx].values, label=target.iloc[val_idx].values)
    
    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx].values, num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test_df.values, num_iteration=clf.best_iteration) / folds.n_splits
    
print("Total OOF RMSE: {}".format(np.sqrt(mean_squared_error(target, oof))))
print("Our Baseline model (predicting all 0's) RMSE:{}".format(np.sqrt(mean_squared_error(target, np.array([0] * len(oof))))) )


# ## Prepare Submission File
# Submitting from a kaggle kernel is pretty simple. Simply output a csv that resembles the sample submission. In this competition the column headers should be *card_id* and *target* where target is our predicted value

# In[ ]:


test_df['target'] = predictions
test_df.reset_index()[['card_id', 'target']].to_csv('submission.csv', index=False)


# ## Moving Forward
# 
# Overall, our model is pretty weak and barely outperforms simply guessing 0 for every target. Improvement over this score primarily comes from two sources:
# 
# 1) Feature engineering
# *  We only use one small data source, but this competition has tons of data that contains powerful information for predicting our target
# * Clever use of combining interactions between features will give information that the model may not learn
# 
# 2) Model selection and ensembling
# * We only used a simple, out of the box, solution to quickly get a prediction. It's possible that other models may perform better, or that better hyper parameters will perform better
