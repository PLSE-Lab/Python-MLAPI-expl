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


# In[ ]:


PATH = "../input/"


# In[ ]:


with open(PATH + 'data_description.txt', 'r') as f:
    print(f.read())


# In[ ]:


train_df = pd.read_csv(PATH + 'train.csv')
test_df  = pd.read_csv(PATH + 'test.csv')


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


cols_to_keep = []
for col in train_df.columns:
    if train_df[col].notnull().sum() > 1200:
        cols_to_keep.append(col)

print(cols_to_keep)


# In[ ]:


cleaned_train_df = train_df[cols_to_keep]
cleaned_train_df.head(5)


# In[ ]:


cleaned_test_df = test_df
cleaned_test_df['SalePrice'] = 0
cleaned_test_df = cleaned_test_df[cols_to_keep]
cleaned_test_df.head(5)


# In[ ]:


#check the length of the df 
len(cleaned_train_df)


# In[ ]:


cleaned_train_df.info()


# In[ ]:


for col in cleaned_train_df.columns:
    cleaned_train_df[col].fillna(cleaned_train_df[col].mode()[0], inplace=True)


# In[ ]:


cleaned_train_df.info()


# In[ ]:


#Now the the train and test datasets have the same columns.  Check any missing in the test dataset 


# In[ ]:


cleaned_test_df.info()


# In[ ]:


# There are some missings but they are a few, so I'll fill them with the mode 
for col in cleaned_test_df.columns:
    cleaned_test_df[col].fillna(cleaned_test_df[col].mode()[0], inplace=True)


# In[ ]:


cleaned_test_df.info()


# In[ ]:


#Train and test datasets seem consistent, I'll remove the SalePrice column in the test dataset as it was needed
# only to apply filtering on columns


# In[ ]:


#cleaned_test_df = cleaned_test_df.drop('SalePrice', axis = 1 )


# In[ ]:


train_len = len(cleaned_train_df)
test_len = len(cleaned_test_df)


# In[ ]:


full_df = pd.concat([cleaned_train_df,cleaned_test_df], sort=False)


# In[ ]:


from sklearn.linear_model import Lasso


# In[ ]:


full_df_dummies= pd.get_dummies(full_df)
X_train = full_df_dummies.drop(['SalePrice','Id'],axis=1).iloc[0:train_len]
X_test = full_df_dummies.drop(['SalePrice','Id'],axis=1).iloc[train_len:]
y_train = full_df_dummies['SalePrice'].iloc[0:train_len]


# In[ ]:


model = Lasso()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.score(X_train,y_train)


# In[ ]:


# Check the sample submission 
sample_sub = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


outputs = pd.DataFrame({'Id':full_df.iloc[train_len:].Id, 'SalePrice':model.predict(X_test)})


# In[ ]:


outputs.to_csv('submission_lasso.csv', index=False)


# In[ ]:


outputs


# In[ ]:


feature_importance = pd.Series(index = X_train.columns, 
                              data = np.abs(model.coef_))


# In[ ]:


n_selected_features = (feature_importance > 0).sum()


# In[ ]:


print('{0:d} features, reduction of {1:2.2f}%'.format(
    n_selected_features,(1-n_selected_features/len(feature_importance))*100))


# In[ ]:


feature_importance.sort_values().tail(30).plot(kind = 'bar', figsize = (18,6))


# In[ ]:


X_train[feature_importance.index]


# ### XGBoost

# In[ ]:


import xgboost as xgb


# In[ ]:


xgb_model_full = xgb.XGBRegressor(objective="reg:linear", random_state=0)
xgb_model_red = xgb.XGBRegressor(objective="reg:linear", random_state =0)


# In[ ]:


xgb_model_full.fit(X_train,y_train)
xgb_model_red.fit(X_train[feature_importance.index],y_train)


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt


# In[ ]:


sqrt(mean_squared_error(y_train, model.predict(X_train)))


# In[ ]:


sqrt(mean_squared_error(y_train,xgb_model_full.predict(X_train)))


# In[ ]:


sqrt(mean_squared_error(y_train,xgb_model_red.predict(X_train)))


# In[ ]:


outputs = pd.DataFrame({'Id':full_df.iloc[train_len:].Id, 'SalePrice':xgb_model_full.predict(X_test)})


# In[ ]:


outputs.to_csv("submission_xgb.csv", index=False)


# #XGBoost with RandomSearch

# In[ ]:


# Grid Search for XGB
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.4,0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
xgb_random = xgb.XGBRegressor()


# In[ ]:


random_search_xgb = RandomizedSearchCV(xgb_random, param_distributions=params,
                                   n_jobs=-1, cv=5, verbose=3, random_state=0 )


# In[ ]:


random_search_xgb.fit(X_train,y_train)


# In[ ]:


random_search_xgb.best_estimator_


# In[ ]:


sqrt(mean_squared_error(y_train,random_search_xgb.predict(X_train)))


# In[ ]:


outputs = pd.DataFrame({'Id':full_df.iloc[train_len:].Id, 'SalePrice':random_search_xgb.predict(X_test)})


# In[ ]:


outputs.to_csv("submission_gridSearch_xgb.csv",index=False)


# ### XGBoost with RandomGridSearch - Different Params

# In[ ]:


'''params = {'booster':['gbtree','gblinear'],
        'min_child_weight': np.linspace(1,10,10),
        'gamma': [0.5, 1, 1.5, 2],
        'subsample': [0.4,0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'learning_rate': np.linspace(0.01,0.2,10),
        'n_estimators':[10,20,50,100,200],
        'max_depth':[1,2,3,4,5,6,7,8,9,10,20,50]
        }
    '''


# In[ ]:


#grid_xgb = RandomizedSearchCV(xgb.XGBRegressor(), param_distributions= params, n_jobs=-1, cv=5, verbose=10, n_iter = 200)


# In[ ]:


#grid_xgb.fit(X_train,y_train)


# In[ ]:


#found through grid search
estimator = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.8, gamma=1, importance_type='gain',
       learning_rate=0.2, max_delta_step=0, max_depth=4,
       min_child_weight=3.0, missing=None, n_estimators=200, n_jobs=1,
       nthread=None, objective='reg:linear', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1.0)
estimator.fit(X_train,y_train)


# In[ ]:


estimator.score(X_train,y_train)


# In[ ]:


outputs = pd.DataFrame({'Id':full_df.iloc[train_len:].Id, 'SalePrice':estimator.predict(X_test)})


# In[ ]:


sqrt(mean_squared_error(y_train,estimator.predict(X_train)))#


# In[ ]:


outputs.to_csv("XGBoostrandomized_sub.csv",index=False)


# In[ ]:




