#!/usr/bin/env python
# coding: utf-8

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


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Load data
train = np.load('../input/multi-cat-encodings/X_train_le.npy')
test = np.load('../input/multi-cat-encodings/X_test_le.npy')

print(train.shape)
print(test.shape)


# In[ ]:


columns = np.load('../input/multi-cat-encodings/columns.npy', allow_pickle=True)


# In[ ]:


columns


# In[ ]:


train_test = np.vstack([train, test])


# In[ ]:


target = np.hstack([np.zeros(600000,), np.ones(400000,)])


# In[ ]:


train, test, y_train, y_test = model_selection.train_test_split(train_test, target, test_size=0.33, random_state=42, shuffle=True)
del train_test, target
gc.collect()


# In[ ]:


train = lgb.Dataset(train, label=y_train)
test = lgb.Dataset(test, label=y_test)


# In[ ]:


param = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'binary',
         'max_depth': 5,
         'learning_rate': 0.05,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 44,
         "metric": 'auc',
         "verbosity": -1}


# In[ ]:


num_round = 1000
clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 500)


# The AUC of 0.63 is well beyond "randon", and could be pretty significant in terms of distinguishing between train and test sets. It's certainly well byond almost perfect 0.5 AUC of the fist Categorical Encoding competition. 
# 
# Let's take a look at what features are the most responsible for the discepancy.

# In[ ]:


feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 20))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(100))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')


# This is interesting - seems like the top five features by importance are all nominals. This could mean that a better nominal encoding could help with the discrepancy between the train and test sets. 

# In[ ]:




