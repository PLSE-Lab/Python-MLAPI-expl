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


train=pd.read_csv('../input/learn-together/train.csv')
test=pd.read_csv('../input/learn-together/test.csv')
sample_submission=pd.read_csv('../input/learn-together/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


cols=list(train.columns) 


# In[ ]:


X_train=train.copy()
x_test=test.copy()


# In[ ]:


other_var=[]   #list of columns other than soil
for i in range(1,11):
    other_var.append(cols[i])
print(other_var)


# In[ ]:


train_trans=X_train[other_var]
test_trans=x_test[other_var]


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler()
# X_train_minmax = mm_scaler.fit_transform(X_train[other_var])
# mm_scaler.transform(x_test[other_var])
mm_scaler.fit(train_trans)
MinMaxScaler(copy=True, feature_range=(0, 1))
trans1=mm_scaler.transform(train_trans)
trans2=mm_scaler.transform(test_trans)


# In[ ]:


temp1=pd.DataFrame(trans1)
temp2=pd.DataFrame(trans2)
temp2.head()


# In[ ]:


for i in range(0,10):
    temp1.rename(columns={i:other_var[i]},inplace=True)
    temp2.rename(columns={i:other_var[i]},inplace=True)
temp2.head()


# In[ ]:


X_train[other_var]=temp1[other_var]
x_test[other_var]=temp2[other_var]
x_test.head()


# In[ ]:


y_train=train['Cover_Type']
X_train.drop(['Cover_Type','Id'],axis=1,inplace=True) #removing target and id column from train set
x_test.drop(['Id'],axis=1,inplace=True) 


# In[ ]:


import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import gc


# In[ ]:


def model_training(model,X_train,y_train):
    scores =  cross_val_score(model, X_train, y_train,
                              cv=5)
    return scores.mean()


# In[ ]:


# Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 8)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# print(random_grid)


# In[ ]:


# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestClassifier()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# # Fit the random search model
# rf_random.fit(X_train, y_train)


# In[ ]:


# rf_random.best_params_


# In[ ]:


my_model=RandomForestClassifier(n_estimators=885,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',max_depth=110,bootstrap=False)


# In[ ]:


my_model.fit(X_train,y_train)


# In[ ]:


test_preds=my_model.predict(x_test)


# In[ ]:


output = pd.DataFrame({'Id': test.Id,
                      'Cover_Type': test_preds})


# In[ ]:


output.to_csv('sample_submission.csv', index=False)

