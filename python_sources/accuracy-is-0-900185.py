#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


import pandas as pd
import numpy as np
import gc

# Gradient Boosting
import lightgbm as lgb
import xgboost as xgb

# Scikit-learn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Skopt functions
from skopt import BayesSearchCV
from skopt import gp_minimize # Bayesian optimization using Gaussian Processes
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args # decorator to convert a list of parameters to named arguments
from skopt.callbacks import DeadlineStopper # Stop the optimization before running out of a fixed budget of time.
from skopt.callbacks import VerboseCallback # Callback to control the verbosity
from skopt.callbacks import DeltaXStopper # Stop the optimization If the last two positions at which the objective has been evaluated are less than delta

# Hyperparameters distributions
from scipy.stats import randint
from scipy.stats import uniform

# Metrics
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error

import os
import warnings; warnings.filterwarnings("ignore")


# **Reading the Dataset**

# In[3]:


santander_data = pd.read_csv('../input/train.csv')
santander_data_test = pd.read_csv('../input/test.csv')
santander_submission = pd.read_csv('../input/sample_submission.csv')
print('Training df shape',santander_data.shape)
print('Test df shape',santander_data_test.shape)


# **Let's check the distribution of 'target' value in train dataset**

# In[4]:


sns.countplot(santander_data['target'])


# **Storing the 'target' feature in a separate variable**

# In[5]:


label_df = santander_data['target']


# **Dropping 'ID_code' and 'target' feature from the Train data and 'ID_code' from the Test data simultaneously**

# In[6]:


santander_data.drop(['ID_code','target'], axis=1, inplace=True)

santander_data_test.drop('ID_code', axis=1, inplace=True)
santander_data.head(5)


# **Checking the Datatypes for all the columns**

# In[7]:


santander_data.select_dtypes(exclude=np.number).columns


# **Storing the train data in 'len_train' variable**

# In[8]:


len_train = len(santander_data)
len_train


# In[9]:


get_ipython().run_cell_magic('time', '', 'santander_data.describe()')


# In[10]:


get_ipython().run_cell_magic('time', '', 'santander_data_test.describe()')


# Things we can infer from above: 
# 1. SD is high for both the Train and Test Dataset.
# 2. Min, Max , Mean is nearly close to each other for Train and Test set.
# 3. Mean is largely distributed.

# **Merging the Train and Test Dataset**

# In[11]:


merged = pd.concat([santander_data, santander_data_test])


# **Saving the list of original features in a new list `original_features`**

# In[12]:


original_features = merged.columns
merged.shape


# **Creating new features**

# In[13]:


idx = features = merged.columns.values[0:200]
for df in [merged]:
    df['sum'] = df[idx].sum(axis=1)  
    df['min'] = df[idx].min(axis=1)
    df['max'] = df[idx].max(axis=1)
    df['mean'] = df[idx].mean(axis=1)
    df['std'] = df[idx].std(axis=1)
    df['skew'] = df[idx].skew(axis=1)
    df['kurt'] = df[idx].kurtosis(axis=1)
    df['med'] = df[idx].median(axis=1)


# In[14]:


print("Total number of features: ",merged.shape[1])


# In[15]:


train_df = merged.iloc[:len_train]
train_df.head()


# In[16]:


X_test = merged.iloc[len_train:]
X_test.head()


# In[17]:


train_df = santander_data
X_test = santander_data_test
del santander_data
del santander_data_test
gc.collect()


# In[18]:


skf_three= StratifiedKFold(n_splits=5, shuffle=True, random_state=2319)


# In[19]:


param = {'bagging_freq': 5,
    'bagging_fraction': 0.335,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.041,
    'learning_rate': 0.0083,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': -1}


# In[20]:


oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(len(X_test))
feats = [f for f in train_df.columns]
    
for n_fold, (train_idx, valid_idx) in enumerate(skf_three.split(train_df[feats], label_df)):
    trn_data = lgb.Dataset(train_df.iloc[train_idx][feats], label=label_df.iloc[train_idx])
    val_data = lgb.Dataset(train_df.iloc[valid_idx][feats], label=label_df.iloc[valid_idx])
        
    clf = lgb.train(param, trn_data,1000000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3500)
        

    oof_preds[valid_idx] = clf.predict(train_df.iloc[valid_idx][feats], num_iteration=clf.best_iteration)
    sub_preds += clf.predict(X_test[feats], num_iteration=clf.best_iteration) / 5


print('Full AUC score %.6f' % roc_auc_score(label_df, oof_preds))

pred3=sub_preds


# In[22]:


santander_submission['target'] = pred3
santander_submission.to_csv('submission23.csv', index=False)

