#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split #train_split to split train and test
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **Import(read) DataSet**

# In[ ]:


file_path='../input/home-data-for-ml-course/train.csv'
data_set=pd.read_csv(file_path,index_col='Id')
data_set.head()


# # Define X and y

# In[ ]:


X_full=data_set.copy()
X_full.dropna(axis=0,subset=['SalePrice'],inplace=True)
y=X_full.SalePrice
X_full.drop(['SalePrice'],axis=1,inplace=True)
X=X_full.select_dtypes(exclude=['object'])
X.head()


# **show colums**

# In[ ]:


X.columns


# ## Training my data

# In[ ]:


X_train,X_valid,y_train,y_valid=train_test_split(X,y,random_state=0)


# **Define my parameters to determaine best choices for best model**

# In[ ]:


para=[{'n_est':list(range(100,1001,50)),
      'learn_rate':[x/100 for x in range(5,51,4)],
      'max_dep':list(range(6,73,6))
      }]
para


# ### Define gSearch with my hypter_para 

# In[ ]:


gsearch=GridSearchCV(
estimator=XGBRegressor(),
param_grid=para,
scoring='neg_mean_absolute_error',
n_jobs=4,
cv=5,
verbose=20)


# **fit my model (gsearch)**

# In[ ]:


gsearch.fit(X,y)


# ## **get best para from model fit**

# In[ ]:


best_n_est=gsearch.best_params_.get('n_est')
best_lean_rate=gsearch.best_params_.get('learn_rate')
best_max_dep=gsearch.best_params_.get('max_dep')

best_n_est,best_lean_rate,best_max_dep


# # Finally find Best(final) model 
# ### and fit it

# In[ ]:


best_model=XGBRegressor(n_estimators=best_n_est,
                       learning_rate=best_lean_rate,
                       max_depth=best_max_dep)
best_model.fit(X,y)


# ### **Define test dataSet for my production **

# In[ ]:


test_file_path='../input/home-data-for-ml-course/test.csv'
test_data_set=pd.read_csv(test_file_path,index_col='Id')
X_test=test_data_set.select_dtypes(exclude=['object'])
X_test.head()


# ## Prediction of test Data

# In[ ]:


my_preds_test=best_model.predict(X_test)
my_preds_test


# **My OutPut**

# In[ ]:


my_output=pd.DataFrame(
{'Id':X_test.index,
'SalePrice':my_preds_test})
my_output.to_csv('submission.csv',index=False)

