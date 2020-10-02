#!/usr/bin/env python
# coding: utf-8

# * Adverserial validation scrip. Based entirely on features from: https://www.kaggle.com/danofer/baseline-feature-engineering-geotab-69-5-lb
# * Which was forked from : https://www.kaggle.com/pulkitmehtawork1985/beating-benchmark
# * Copies feature code over from Dan's other kernel; https://www.kaggle.com/danofer/basic-features-geotab-intersections
# 
# If you find this notebook useful, please upvote the obove other notebooks as well.

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


train = pd.read_csv('../input/baseline-feature-engineering-geotab-69-5-lb/train_danFeatsV1.csv.gz')
test = pd.read_csv('../input/baseline-feature-engineering-geotab-69-5-lb/test_danFeatsV1.csv.gz')


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


test.head()


# In[ ]:


FEAT_COLS = ["IntersectionId",
             'Intersection',
           'diffHeading',  'same_street_exact',
           "Hour","Weekend","Month",
          'Latitude', 'Longitude',
          'EntryHeading', 'ExitHeading',
            'Atlanta', 'Boston', 'Chicago',
       'Philadelphia']


# In[ ]:


train = train[FEAT_COLS]
test = test[FEAT_COLS]


# In[ ]:


train['target'] = 0
test['target'] = 1


# In[ ]:


train_test = pd.concat([train, test], axis =0)

target = train_test['target'].values


# In[ ]:


del train, test
gc.collect()


# In[ ]:


train, test = model_selection.train_test_split(train_test, test_size=0.33, random_state=42, shuffle=True)
del train_test
gc.collect()


# In[ ]:


train_y = train['target'].values
test_y = test['target'].values
del train['target'], test['target']
gc.collect()


# In[ ]:


train.head()


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


num_round = 500
clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)


# So with AUC of about 0.74, there seems to be a farily significant difference in distribution of features that we have used between the train and test sets. So let's take a look at what features are most responsible for this difference:

# In[ ]:


feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),FEAT_COLS)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(20))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')


# So it looks like the actual geographical distribution of locations is very different between the two sets.

# In[ ]:





# In[ ]:




