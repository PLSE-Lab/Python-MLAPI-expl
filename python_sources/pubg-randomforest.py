#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from collections import Counter
from pprint import pprint
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)


# In[ ]:


train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
sample_submission = pd.read_csv('../input/sample_submission_V2.csv')


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


#Check for null values
train.isnull().any().values


# In[ ]:


train = train.dropna()


# In[ ]:


test.isnull().any().values


# In[ ]:


train.winPlacePerc.plot(kind='hist')


# In[ ]:


train[train.killStreaks > 7].shape


# In[ ]:


#82 players were killed more than 7 enimies in a short time


# In[ ]:


#players with more headshotkills
train.headshotKills.value_counts()


# In[ ]:


#Lets add some more features
train['total_dist'] = train['swimDistance'] + train['walkDistance'] + train['rideDistance']


# In[ ]:


test['total_dist'] = test['swimDistance'] + test['walkDistance'] + test['rideDistance']


# In[ ]:


train['kills_with_assist'] = train['kills'] + train['assists']


# In[ ]:


test['kills_with_assist'] = test['kills'] + test['assists']


# In[ ]:


print("Average distance travelled by player is ",train['total_dist'].mean())


# In[ ]:


train.DBNOs.value_counts().head(10).plot(kind='bar')


# In[ ]:


plt.scatter(train['rideDistance'],train['roadKills'])


# In[ ]:


train['headshot_over_kills'] = train['headshotKills'] / train['kills']
train['headshot_over_kills'].fillna(0, inplace=True)


# In[ ]:


test['headshot_over_kills'] = test['headshotKills'] / test['kills']
test['headshot_over_kills'].fillna(0, inplace=True)


# In[ ]:


train['headshot_over_kills'].value_counts().head(5)


# In[ ]:


train.head(2)


# In[ ]:


train = train.drop(['Id','groupId','matchId'],axis=1)


# In[ ]:


train.shape


# In[ ]:


matchtype = train.matchType.unique()


# In[ ]:


matchtype.__len__()


# In[ ]:


match_dict = {}
for i,each in enumerate(matchtype):
    match_dict[each] = i


# In[ ]:


match_dict


# In[ ]:


train.matchType = train.matchType.map(match_dict)


# In[ ]:


matchtype_test = test.matchType.unique()
match_dict_test = {}
for i,each in enumerate(matchtype_test):
    match_dict_test[each] = i
test.matchType = test.matchType.map(match_dict_test)


# In[ ]:


y = train['winPlacePerc']


# In[ ]:


X = train.drop(['winPlacePerc'],axis=1)


# In[ ]:


X.shape,y.shape


# In[ ]:


y[:2]


# In[ ]:


X[:2]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:





# In[ ]:


#Lets Normalize the train data
sc_X = StandardScaler()
X_trainsc = sc_X.fit_transform(X_train)
X_testsc = sc_X.transform(X_test)


# ### Linear Regression

# In[ ]:


lr = LinearRegression()
lr.fit(X_trainsc, y_train)


# In[ ]:


y_pred = lr.predict(X_testsc)


# In[ ]:


y_pred[:10]


# In[ ]:


rmse = sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)


# In[ ]:


print("RMSE = >",rmse)
print("MSE = >",mse)
print("R Squared = >",r2)


# In[ ]:


res = pd.DataFrame()
res['Actual'] = y_test
res['Predicted'] = y_pred
res['Difference'] = abs(y_test-y_pred)


# In[ ]:


res.head(10)


# In[ ]:


#Decision Tree Regressor
dt = DecisionTreeRegressor()
dt.fit(X_trainsc,y_train)


# In[ ]:


y_pred_dt = dt.predict(X_testsc)


# In[ ]:


rmse = sqrt(mean_squared_error(y_test, y_pred_dt))
mse = mean_squared_error(y_test, y_pred_dt)
r2 = r2_score(y_test,y_pred_dt)


# In[ ]:


print("RMSE = >",rmse)
print("MSE = >",mse)
print("R Squared = >",r2)


# In[ ]:


dt = pd.DataFrame()
dt['Actual'] = y_test
dt['Predicted'] = y_pred_dt
dt['Difference'] = abs(y_test-y_pred_dt)


# In[ ]:


dt.head(10)


# ### LightGBM

# In[ ]:


parameters = {
                'max_depth': 1,'min_data_in_leaf': 85,'feature_fraction': 0.80,'bagging_fraction':0.8,'boosting_type':'gbdt',
                'learning_rate': 0.1, 'num_leaves': 30,'subsample': 0.8,'lambda_l2': 4,'objective': 'regression_l2',
                'application':'regression','num_boost_round':5000,'zero_as_missing': True,
                'early_stopping_rounds':100,'metric': 'mae','seed': 2
             }


# In[ ]:


train_data = lgb.Dataset(X_trainsc, y_train, silent=False)
test_data = lgb.Dataset(X_testsc, y_test, silent=False)
model = lgb.train(parameters, train_set = train_data,verbose_eval=500, valid_sets=test_data)


# In[ ]:


test = test.drop(['Id','groupId','matchId'],axis=1)


# In[ ]:


#Lets check the prediction with x_testsc 
pred_lgb_samp_sc = model.predict(X_testsc, num_iteration = model.best_iteration)


# In[ ]:


lgb_res= pd.DataFrame()
lgb_res['Actual'] = y_test
lgb_res['Predicted_sc'] = pred_lgb_samp_sc
lgb_res['Difference'] = abs(y_test-pred_lgb_samp_sc)


# In[ ]:


lgb_res.head(10)


# In[ ]:


# We'll normalize the test data aswell for better prediction

sc_test = StandardScaler()
test_sc = sc_test.fit_transform(test)


# In[ ]:


# prediction
pred_lgb_sc = model.predict(test_sc, num_iteration = model.best_iteration)


# In[ ]:


pred_lgb_sc[:10]


# In[ ]:


# Replace the prediction which is greator than 1 by 1 and less than 0 by 0

pred_lgb_sc[pred_lgb_sc > 1] = 1
pred_lgb_sc[pred_lgb_sc < 0] = 0


# In[ ]:


pred_lgb_sc.__len__()


# In[ ]:


sample_submission['winPlacePerc'] = pred_lgb_sc


# In[ ]:


sample_submission.to_csv('sample_submission.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:




