#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# This notebook is inspired from Ruslan Talipov's  baseline model. And contains few plots.

# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# In[ ]:


train.head(3)


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.dtypes


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# - Most features are sparse

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


unique_df = train.nunique().reset_index()
unique_df.columns = ["col_name", "unique_count"]
constant_df = unique_df[unique_df["unique_count"]==1]
constant_df.shape


# - It can be seen that 256 columns have single values, so it is better to remove them.

# In[ ]:


train.drop(list(constant_df.col_name),axis=1,inplace=True)


# In[ ]:


test.drop(list(constant_df.col_name),axis=1,inplace=True)


# In[ ]:


train.shape


# #### Still a lot more columns than rows

# In[ ]:


plt.figure(figsize=(20,10))
sns.distplot(train['target'],kde=True)


# - This is highly left skewed, so lets see the plot after normalizing it

# In[ ]:


plt.figure(figsize=(20,10))
sns.distplot(np.log(1+train['target']))
plt.show()


# In[ ]:


def rmsle(y,pred):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(pred), 2)))


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train['log_target']=np.log(1+train['target'])


# In[ ]:


train['log_target'].describe()


# In[ ]:


y=train['log_target']
train.drop(['target','log_target'], axis=1, inplace=True)


# In[ ]:


test_id=test['ID']
train_id=train['ID']
test.drop(['ID'], axis=1, inplace=True)
train.drop(['ID'], axis=1, inplace=True)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(train,y,test_size=0.2,random_state=42)


# In[ ]:


X_train.shape,X_test.shape


# In[ ]:


from sklearn.cross_validation import StratifiedKFold,KFold


# In[ ]:


from sklearn.linear_model import ElasticNet,LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# In[ ]:


import lightgbm as lgb
import time


# In[ ]:


lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': -1,
    'num_leaves': 255,  # 63, 127, 255
    'feature_fraction': 0.5, # 0.1, 0.01
    'bagging_fraction': 0.5,
    'learning_rate': 2e-4, #0.00625,#125,#0.025,#05,
    'verbose': 1,
    'bagging_freq':10,
    #'device':'gpu',
    "reg_alpha": 0.3,
    "reg_lambda": 0.1,
    "min_child_weight":10,
    'zero_as_missing':True
}


# ### Trying with Random Forest

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=1, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)


# In[ ]:





# In[ ]:


model1=RandomForestRegressor()


# In[ ]:


model1.fit((X_train),y_train)


# In[ ]:


print(rmsle(y_test, model1.predict(X_test)))


# In[ ]:


pred1=model1.predict(test)


# In[ ]:


pred1


# In[ ]:


pred1=np.exp(pred1)-1


# In[ ]:


pred1


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = pd.Series(pred1)
sub.to_csv('sub_rbf_baseline.csv', index=False)


# In[ ]:


Y_target = []
for fold_id,(train_idx, val_idx) in enumerate(KFold(n=train.shape[0],n_folds=10, random_state=42,shuffle=True)):
    print('FOLD:',fold_id)
    X_train = train.values[train_idx]
    y_train = y.values[train_idx]
    X_valid = train.values[val_idx]
    y_valid =  y.values[val_idx]
    
    
    lgtrain = lgb.Dataset(X_train, y_train,
                feature_name=train.columns.tolist(),
    #             categorical_feature = categorical
                         )

    lgvalid = lgb.Dataset(X_valid, y_valid,
                feature_name=train.columns.tolist(),
    #             categorical_feature = categorical
                         )

    modelstart = time.time()
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=30000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    test_pred = lgb_clf.predict(test.values)
    Y_target.append(np.exp(test_pred)-1)
    print('fold finish after', time.time()-modelstart)


# In[ ]:


Y_target = np.array(Y_target)


# In[ ]:


Y_target.shape


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = Y_target.mean(axis=0)
sub.to_csv('sub_lgb_baseline.csv', index=False)


# In[ ]:




