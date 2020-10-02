#!/usr/bin/env python
# coding: utf-8

# In this adversarial validation notebook we'll take a look at the difference between sites 1 and 2.

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
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")
loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")

fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")
reveal_ID_site2 = pd.read_csv("../input/trends-assessment-prediction/reveal_ID_site2.csv")
labels_df.head()


# In[ ]:


labels_df.shape


# In[ ]:


reveal_ID_site2.head()


# In[ ]:


reveal_ID_site2.shape


# In[ ]:


labels_df.shape


# In[ ]:


df['target'] = np.nan


# In[ ]:


df.isna().sum().sum()


# In[ ]:


df.loc[df.Id.isin(labels_df.Id), 'target'] = 0
df.loc[df.Id.isin(reveal_ID_site2.Id), 'target'] = 1
df.dropna(inplace=True)

df.shape


# In[ ]:


features = df.columns[1:-1]
train = df[features].values
target = df['target'].values


# In[ ]:


train, test, y_train, y_test = model_selection.train_test_split(train, target, test_size=0.33, random_state=42, shuffle=True)
del target
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
         "bagging_seed": 56,
         "metric": 'auc',
         "verbosity": -1}


# In[ ]:


num_round = 10000
clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 100)


# The AUC of 0.88 is much higher than the AUC of 0.72 for the "regular" train-test adversarial validation. Seems taht we really ahve to be very careful about the differences between sites 1 and 2.
# 
# Let us now take a look at the most important features 

# In[ ]:


feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])

plt.figure(figsize=(20, 20))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(100))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')


# Here it would appear that the worst "culprits" are the IC features. Let's again list the top 20 most important ones, at least according to this measure.

# In[ ]:


feature_imp.sort_values(by="Value", ascending=False).head(20)


# In[ ]:




