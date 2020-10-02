#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import gc
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


# # Loading features
# 
# I am using the features from @artgors [EDA and models kernel](https://www.kaggle.com/artgor/eda-and-models).

# In[ ]:


train_features_path = '../input/baseline-features/train_features.csv'
test_features_path = '../input/baseline-features/test_features.csv'

train = pd.read_csv(train_features_path)
test = pd.read_csv(test_features_path)


# In[ ]:


train = train.sort_values('TransactionDT')
test = test.sort_values('TransactionDT')


# Let's plot train and test by 'TransactionDT'.

# In[ ]:


import matplotlib.pyplot as plt

fig, axs = plt.subplots(1,2, figsize=(16,4))
train.groupby(['TransactionDT'])['TransactionDT'].size().plot(ax=axs[0])
test.groupby(['TransactionDT'])['TransactionDT'].size().plot(ax=axs[1])


# We can clearly see the time split.

# In[ ]:


del test
gc.collect()


# # Finding a good time split
# So let's try to decide what should be the proper time split.

# In[ ]:


split_perc = [p*0.01 for p in range(100)]
y_means_train, y_means_valid = [],[]
for p in split_perc:
    idx = int(p*len(train))
    y_means_train.append(train['isFraud'][:idx].mean())
    y_means_valid.append(train['isFraud'][idx:].mean())


# In[ ]:


fig, ax = plt.subplots(figsize=(16,4))
ax.plot(split_perc, y_means_train, label='train')
ax.plot(split_perc, y_means_valid, label='valid')


# In[ ]:


split_perc_df = pd.DataFrame({'perc':split_perc,'train':y_means_train, 'valid':y_means_valid})
split_perc_df['diff'] = abs(split_perc_df['train']-split_perc_df['valid'])
split_perc_df.sort_values('diff').head()


# Okey, it seems that seems that we have a few candidates that have similiar `isFraud` fraction.
# Let's now see what is the difference row-wise in those `0.74` and `0.87` splits.

# In[ ]:


0.13*len(train), 0.26*len(train)


# Ok, it's quite large and the difference in `isFraud` fraction is not, so I will go with the `0.26` of train as my validation set.
# 
# # Finding a good train sample
# 
# In imbalanced problems a lot of the times negative samples do not bringing that much to the table from a certain point.
# I want to explore whether I can subsample negative samples from the train (keeping valid as is) and get a similar score on valid.
# By doing that I can cut down the training time significantly and experiment faster as a result.
# 
# Let's implement a simple sampling function:

# In[ ]:


def sample_negative_class(train, perc):
    train_pos = train[train.isFraud==1]
    train_neg = train[train.isFraud==0].sample(frac=perc)
    
    train = pd.concat([train_pos, train_neg], axis=0)
    train = train.sort_values('TransactionDT')
    return train


# In[ ]:


def fit_predict(train, valid, model_params, training_params):
    X_train = train.drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
    y_train = train['isFraud']

    X_valid = valid.drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
    y_valid = valid['isFraud']
    
    trn_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_valid, y_valid)

    clf = lgb.train(model_params, trn_data, 
                    training_params['num_boosting_rounds'], 
                    valid_sets = [trn_data, val_data], 
                    early_stopping_rounds = training_params['early_stopping_rounds'],
                    verbose_eval=False
                   )
    train_preds = clf.predict(X_train, num_iteration=clf.best_iteration)
    valid_preds = clf.predict(X_valid, num_iteration=clf.best_iteration)
    return train_preds, valid_preds


# In[ ]:


idx_split = int(0.74*len(train))
train_split, valid_split = train[:idx_split], train[idx_split:]


# In[ ]:


model_params = {'num_leaves': 256,
                  'min_child_samples': 79,
                  'objective': 'binary',
                  'max_depth': 15,
                  'learning_rate': 0.05,
                  "boosting_type": "gbdt",
                  "subsample_freq": 3,
                  "subsample": 0.9,
                  "bagging_seed": 11,
                  "metric": 'auc',
                  "verbosity": -1,
                  'reg_alpha': 0.3,
                  'reg_lambda': 0.3,
                  'colsample_bytree': 0.9
                 }

training_params = {'num_boosting_rounds':1000,
                   'early_stopping_rounds':100,
               }


# Now we can go ahead and train on 20 different sample options.

# In[ ]:


train_sample_perc = [p*0.05 for p in range(1,20,1)]
train_scores, valid_scores = [],[]
for perc in train_sample_perc:
    print('processing for perc {}'.format(perc))
    train_sample = sample_negative_class(train_split, perc)
    train_preds, valid_preds = fit_predict(train_sample, valid_split, model_params, training_params)
    
    train_score = roc_auc_score(train_sample['isFraud'], train_preds)
    valid_score = roc_auc_score(valid_split['isFraud'], valid_preds)
    print(perc, train_score, valid_score)
    train_scores.append(train_score)
    valid_scores.append(valid_score)


# In[ ]:


fig, axs = plt.subplots(2,1, figsize=(16,4))
axs[0].plot(train_sample_perc, train_scores, label='train')
axs[1].plot(train_sample_perc, valid_scores, label='valid')


# In[ ]:


sample_perc_df = pd.DataFrame({'perc':train_sample_perc,'train':train_scores, 'valid':valid_scores})
sample_perc_df.sort_values('valid', ascending=False).head()


# Ok, so we can see that the difference in scores from `0.1` to `1.0` of the negative samples is only `0.002`.
# What is interesting is that for those larger samples like `0.75` we get a bump which suggest a lot of noise here. 
# This is more of the reason (in my opinion) to experiment with a smaller sample of negatives and from time to time check it
# on the entire dataset.
# 
# What do you think?
# 
# **Note**
# 
# I will be adding this to the project that I discuss [here](https://www.kaggle.com/c/ieee-fraud-detection/discussion/100311#latest-578849).

# In[ ]:




