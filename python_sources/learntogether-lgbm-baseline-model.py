#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/learn-together/train.csv')
train.head()


# In[ ]:


test = pd.read_csv('../input/learn-together/test.csv')
test.head()


# In[ ]:


import numpy as np
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[ ]:


print("Light GBM version:", lgb.__version__)


# In[ ]:



from sklearn.svm import SVC


# In[ ]:


cover = train['Cover_Type']
feats = train.drop(['Cover_Type','Id'],axis =1)

X_train, X_valid, y_train, y_valid = train_test_split(
        feats, cover, test_size=0.2, random_state=42)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

# to record eval results for plotting
evals_result = {} 


# In[ ]:


get_ipython().run_line_magic('time', '')

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class':8,
    'num_leaves': 50,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf':4,
     #'min_sum_hessian_in_leaf': 5,
    'verbose':10
}

print('Start training...')

# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=200,
                valid_sets=[lgb_train, lgb_valid],
                evals_result=evals_result,
                verbose_eval=10,
                early_stopping_rounds=50)


# In[ ]:


print('Start predicting...')
# predict
y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)


# In[ ]:


# print feature names
print('\nFeature names:', gbm.feature_name())

print('\nCalculate feature importances...')

# feature importances
print('Feature importances:', list(gbm.feature_importance()))

print('Plot feature importances...')
ax = lgb.plot_importance(gbm, max_num_features=10)
plt.show()


# In[ ]:


print('Plot metrics during training...')
ax = lgb.plot_metric(evals_result, metric='multi_logloss')
plt.show()


# In[ ]:


submission = pd.read_csv('../input/learn-together/sample_submission.csv')


# In[ ]:


test2 = test.drop('Id',axis =1)

print('\nPredicting test set...')
y_pred = gbm.predict(test2, num_iteration=gbm.best_iteration)


# In[ ]:


# y_pred = model.predict(dtest)
submission['Cover_Type'] = y_pred
#submission['quantity'] = df['quantity'].apply(lambda x: x*-1)


# In[ ]:


y_pred2 = pd.DataFrame(y_pred)
y_pred2 = y_pred2[y_pred2.columns.drop(0)]
target = y_pred2.idxmax(axis=1)
submission['Cover_Type'] = target
submission.head()


# In[ ]:


#LB probing
# target = y_pred2[[1,2]].idxmax(axis=1)
# submission['Cover_Type'] = target
# submission.head()


# In[ ]:


submission.to_csv('submit-lightgbm.csv', index=False)

print("Finished.")

