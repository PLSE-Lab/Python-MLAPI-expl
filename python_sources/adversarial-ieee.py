#!/usr/bin/env python
# coding: utf-8

# Let's take a look how similar train and test sets are.

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

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/standalone-train-and-test-preprocessing/train.csv')
test = pd.read_csv('../input/standalone-train-and-test-preprocessing/test.csv')


# In[ ]:


features = test.columns
features = set(features) - set(['TransactionID','TransactionDT','id_30','id_31'])


# In[ ]:


train = train[features]
test = test[features]


# In[ ]:


train['target'] = 0
test['target'] = 1


# In[ ]:


train_test = pd.concat([train, test], axis =0)
del train, test
gc.collect()

target = train_test['target'].values


# In[ ]:


object_columns = np.load('../input/standalone-train-and-test-preprocessing/object_columns.npy')


# In[ ]:


# Label Encoding
for f in object_columns:
    if f in train_test.columns:
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
         'learning_rate': 0.1,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 44,
         "metric": 'auc',
         "verbosity": -1}


# In[ ]:


num_round = 500
clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)


# In[ ]:


feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(20))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')
