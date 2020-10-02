#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv('/kaggle/input/train_new.csv')
test = pd.read_csv('/kaggle/input/test_new.csv')


# In[ ]:


(train.shape, test.shape)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


import pandas_profiling as pp
#pp.ProfileReport(train)


# In[ ]:


test_id = test['ID']


# In[ ]:


train = train.drop('ID',axis = 1)
test = test.drop('ID',axis = 1)


# In[ ]:


outcome_name = 'target'
features_for_model = [f for f in list(train) if outcome_name !=1]
features_for_model = [f for f in list(test)]


# In[ ]:


X = train[features_for_model]
y = train[outcome_name]


# In[ ]:


#Data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train[features_for_model], train[outcome_name],
                                                    test_size = 0.20,
                                                    random_state = 42)


# In[ ]:


train_categories = np.where(train[features_for_model].dtypes !=np.float)[0]
test_categories = np.where(test[features_for_model].dtypes !=np.float)[0]


# In[ ]:


pip install catboost


# In[ ]:


from sklearn.metrics import log_loss, mean_squared_error
from sklearn.metrics import roc_auc_score


# In[ ]:


from catboost import CatBoostClassifier, Pool

train_data = train[features_for_model]
train_labels = train[outcome_name]

model = CatBoostClassifier()

model.fit(train_data,
          train_labels,
          verbose=False) #,
         #plot=True)                # plot=True


# In[ ]:


params = {'iterations': 248,
          'learning_rate':0.1,
          'depth':5,
          'bagging_temperature':10.197316968394073,
          'subsample': 0.7467983561008608,
          'loss_function':"CrossEntropy",
          'eval_metric':'AUC',
          'random_seed':42,
          'od_type':'Iter', # overfit detector
          'metric_period':20, # calculate metric once per 50 iterations
          'od_wait':50, # most recent best iteration to wait before stopping
          'verbose':True,
          'use_best_model':True}


# In[ ]:


from catboost import CatBoostClassifier

model = CatBoostClassifier(**params)

model.fit(X_train, y_train, eval_set = (X_test, y_test), use_best_model = True) #, plot = True);


# In[ ]:


pred = model.predict(X_test)


# In[ ]:


from sklearn import metrics

ypred = model.predict(X_test)
score = metrics.mean_squared_error(y_test, pred)
print(f"Test score: {score}")


# In[ ]:


from math import sqrt
rmse = np.sqrt(score)
print("RMSE: %f" % (rmse))


# In[ ]:


pred = model.predict(test)   #[:, 1]


# In[ ]:


submiss = pd.DataFrame({"ID": test_id})
submiss['target'] = pred
submiss.to_csv('CatBoostRegressor.csv', index=False)


# In[ ]:


submiss.head()


# In[ ]:


model.get_feature_importance(prettified=True)

