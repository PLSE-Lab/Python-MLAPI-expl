#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install xgboost')


# ## Feature Phase

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df_train=pd.read_csv('train.csv')


# ### Train Dataframes

# In[ ]:


X_train = df_train[df_train.columns[:-1]]
y_train = df_train['SalePrice']


# A big problem in this project is that train and test data has different compositions. For example, there is no NA rows in train, but there is in the test, which makes pandas identify different data types.
# So, my strategy is first convert all columns int to float and check if train-test dataframes has the same dtypes

# In[ ]:


#Converts all int to float
column_types = dict(X_train.dtypes)
for k,v in column_types.items():
    if str(v) == 'int64':
        X_train[k] = X_train[k].astype('float64')


# In[ ]:


#read the test DF based on train DF data types
column_types = dict(X_train.dtypes)
df_test=pd.read_csv('test.csv', dtype=column_types)
X_test = df_test


# In[ ]:


#Check column types
for k,v in dict(X_train.dtypes == X_test.dtypes).items():
    if v==False:
        print("Column %s with diff data types" %k)


# ### Null Columns

# In[ ]:


null_columns=X_train.columns[X_train.isnull().any()]
X_train[null_columns].isnull().sum()


# In[ ]:


null_columns=X_test.columns[X_test.isnull().any()]
X_test[null_columns].isnull().sum()


# In[ ]:


X_train = X_train.fillna(0)
X_test = X_test.fillna(0)


# In[ ]:


#instead of one hot encoding, i'im going to convert the categorical values to numbers.
for k, v in dict(X_train.dtypes).items():
    if v=='object':
        X_train[k] = pd.factorize(X_train[k])[0] + 1
        X_test[k] = pd.factorize(X_test[k])[0] + 1


# In[ ]:


#remove Ids from datasets
X_train_set = X_train[X_train.columns[1:]]
X_test_set = X_test[X_test.columns[1:]]


# In[ ]:


print('%s - %s' %(X_train_set.shape, X_test_set.shape))


# In[ ]:


from sklearn import preprocessing

X_train_set_values = X_train_set.values 
min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train_set_values)
X_train_scaled = pd.DataFrame(X_train_scaled)

#transform test set
X_test_set_values = X_test_set.values
X_test_scaled = min_max_scaler.transform(X_test_set_values)
X_test_scaled = pd.DataFrame(X_test_scaled)


# In[ ]:


print('%s - %s' %(X_train_set.shape, X_test_set.shape))


# ## Model

# In[ ]:


import xgboost as xgb


# In[ ]:


def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# In[ ]:


xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

xgb_model = xgb.XGBRegressor(objective="reg:squarederror")

params = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(100, 150), # default 100
    "subsample": uniform(0.6, 0.4)
}

search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=10, cv=3, verbose=1, n_jobs=1, return_train_score=True)

search.fit(X_train_set, y_train)

report_best_scores(search.cv_results_, 1)


# ## Test Phase - Final Model

# In[ ]:


params = {'colsample_bytree': 0.8275467623473733, 'gamma': 0.10397083143409441, 'learning_rate': 0.20031009834599744, 'max_depth': 2, 'n_estimators': 117, 'subsample': 0.9100531293444458}
xgb_model_final = xgb.XGBRegressor(objective="reg:squarederror", params=params)
xgb_model_final.fit(X_train_scaled, y_train)


# In[ ]:


y_test = xgb_model_final.predict(X_test_scaled)


# In[ ]:


#add predict to X_test, which has Id for submission
y_test = pd.DataFrame(y_test)
X_test_final = X_test
X_test_final['SalePrice'] = y_test


# In[ ]:


X_test_final['Id'] = X_test_final['Id'].astype('int64')
X_test_final[['Id','SalePrice']].head()


# In[ ]:


X_test_final[['Id','SalePrice']].to_csv('submission.csv',index=False)


# In[ ]:




