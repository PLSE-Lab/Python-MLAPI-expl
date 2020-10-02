#!/usr/bin/env python
# coding: utf-8

# This kernel is forked from [Bojan Tunguz's useful kernel](https://www.kaggle.com/tunguz/adversarial-ieee) as always, which says that adversarial AUC is approximately 1.0. However, once you look into it carefully, you will notice that the features should not include 'TransactionDT' since it is `a timedelta from a given reference datetime (not an actual timestamp)`.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn import model_selection, preprocessing, metrics
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import shap
import os
print(os.listdir("../input"))
from sklearn import preprocessing
import xgboost as xgb
import gc

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/standalone-train-and-test-preprocessing/train.csv')
test = pd.read_csv('../input/standalone-train-and-test-preprocessing/test.csv')


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


print(train.TransactionDT.min(), train.TransactionDT.max())
print(test.TransactionDT.min(), test.TransactionDT.max())


# You can distinguish train and test by just checking 'TransactionDT', so you should remove it.

# In[ ]:


features = test.drop('TransactionDT', axis=1).columns


# In[ ]:


train = train[features]
test = test[features]


# In[ ]:


train['target'] = 0
test['target'] = 1


# In[ ]:


train_test = pd.concat([train, test], axis =0)

target = train_test['target'].values


# In[ ]:


object_columns = np.load('../input/standalone-train-and-test-preprocessing/object_columns.npy')


# In[ ]:


del train, test
gc.collect()


# In[ ]:


# Label Encoding
for f in object_columns:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_test[f].values) )
    train_test[f] = lbl.transform(list(train_test[f].values))


# In[ ]:


train, test = model_selection.train_test_split(train_test, test_size=0.33, random_state=42, shuffle=True)


# In[ ]:


del train_test
gc.collect()


# In[ ]:


train_y = train['target'].values
test_y = test['target'].values
del train['target'], test['target']
gc.collect()


# In[ ]:


train = lgb.Dataset(train, label=train_y)
test = lgb.Dataset(test, label=test_y)


# In[ ]:


param = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'binary',
         'max_depth': 5,
         'learning_rate': 0.2,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 44,
         "metric": 'auc',
         "verbosity": -1}


# In[ ]:


num_round = 100
clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)


# Still, adversial AUC is so high that there can be bugs in my codes.
# 
# Let's look now at the top 20 "adversarial" features.

# In[ ]:


feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(20))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')


# Need to look into some of the important features: `id_31`, `D10`, `D15`, `id_13`

# In[ ]:


del train, test
gc.collect()


# ---

# (I know this is not efficient, but this is just an adhoc kernel...)

# In[ ]:


train = pd.read_csv('../input/standalone-train-and-test-preprocessing/train.csv')
test = pd.read_csv('../input/standalone-train-and-test-preprocessing/test.csv')


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(16,10))
train_id_31 = train.id_31.value_counts().iloc[:30]
test_id_31 = test.id_31.value_counts().iloc[:30]
sns.barplot(y=train_id_31.index, x=train_id_31.values, ax=ax[0])
sns.barplot(y=test_id_31.index, x=test_id_31.values, ax=ax[1])
plt.tight_layout()
plt.show()


# OS **version** seems to give you a hint to separate train and test since test data has later datatime.

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(16,10))
train_d10 = train.D10.value_counts().iloc[1:30]
test_d10 = test.D10.value_counts().iloc[1:30]
sns.barplot(x=train_d10.index, y=train_d10.values, ax=ax[0])
sns.barplot(x=test_d10.index, y=test_d10.values, ax=ax[1])
plt.tight_layout()
plt.show()


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(16,10))
train_d15 = train.D15.value_counts().iloc[1:30]
test_d15 = test.D15.value_counts().iloc[1:30]
sns.barplot(x=train_d15.index, y=train_d15.values, ax=ax[0])
sns.barplot(x=test_d15.index, y=test_d15.values, ax=ax[1])
plt.tight_layout()
plt.show()


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(16,10))
sns.distplot(train.id_13.fillna(-9).values, ax=ax[0])
sns.distplot(test.id_13.fillna(-9).values, ax=ax[1])
plt.tight_layout()
plt.show()


# Hmm...I did not figure out whether these plots give us insight. Someone has an idea?
