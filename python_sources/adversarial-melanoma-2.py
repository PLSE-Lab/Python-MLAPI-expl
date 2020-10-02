#!/usr/bin/env python
# coding: utf-8

# In this notebook we'll try to assess the degree of difference between the train and test sets using adversarial validation approach. We have already done the same exercise with the rescaled raw image data, and that approach can be found [here](https://www.kaggle.com/tunguz/adversarial-melanoma). In this notebook we'll just use the metadata and the image size data. We have already preprocessed the datasets [in this notebook](https://www.kaggle.com/tunguz/melanoma-train-test-creator/), and we'll just use the output here.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn import model_selection, preprocessing, metrics

from sklearn import preprocessing
import gc


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os


# In[ ]:


train = pd.read_csv('../input/melanoma-train-test-creator/train_meta_size.csv')
test = pd.read_csv('../input/melanoma-train-test-creator/test_meta_size.csv')


# In[ ]:


features = test.columns


# In[ ]:


target = np.hstack([np.zeros(train.shape[0],), np.ones(test.shape[0],)])
train_test = np.vstack([train[features].values, test.values])
print(train_test.shape)
print(target.shape)
del train, test
gc.collect()


# In[ ]:


train, test, y_train, y_test = model_selection.train_test_split(train_test, target, test_size=0.33, random_state=42, shuffle=True)
del target, train_test
gc.collect()


# In[ ]:


train = lgb.Dataset(train, label=y_train)
test = lgb.Dataset(test, label=y_test)


# In[ ]:


param = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'binary',
         'max_depth': 8,
         'learning_rate': 0.05,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 56,
         "metric": 'auc',
         "verbosity": -1}


# In[ ]:


num_round = 2000
clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)


# For raw image data we were able to get an AUC of 0.65, while here we get 0.70. Seems that the image size and image metafeatrue data has more discrepancy between the train and test sets than the raw rescaled images.
# 
# Let's look at the top features and theri relative importances.

# In[ ]:


feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])

plt.figure(figsize=(20, 20))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(5))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')


# In[ ]:


feature_imp.sort_values(by="Value", ascending=False).head(5)


# So the biggest "culprits" for the train/test discrepancy are the age, followed by the image width.

# We'll now look at the train and test sets with more "rigorously" impoted missing values:

# In[ ]:


train = pd.read_csv('../input/melanoma-train-test-creator/train_meta_size_2.csv')
test = pd.read_csv('../input/melanoma-train-test-creator/test_meta_size_2.csv')


# In[ ]:


features = test.columns


# In[ ]:


target = np.hstack([np.zeros(train.shape[0],), np.ones(test.shape[0],)])
train_test = np.vstack([train[features].values, test.values])
print(train_test.shape)
print(target.shape)
del train, test
gc.collect()


# In[ ]:


train, test, y_train, y_test = model_selection.train_test_split(train_test, target, test_size=0.33, random_state=42, shuffle=True)
del target, train_test
gc.collect()


# In[ ]:


train = lgb.Dataset(train, label=y_train)
test = lgb.Dataset(test, label=y_test)


# In[ ]:


param = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'binary',
         'max_depth': 8,
         'learning_rate': 0.05,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 56,
         "metric": 'auc',
         "verbosity": -1}


# In[ ]:


num_round = 2000
clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)


# So there is a little bit of an "improvement", but overall it is still a fairly significant distinction.

# In[ ]:


feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])

plt.figure(figsize=(20, 20))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(5))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-02.png')


# In[ ]:




