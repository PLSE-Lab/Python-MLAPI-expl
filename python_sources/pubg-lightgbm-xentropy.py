#!/usr/bin/env python
# coding: utf-8

# # LightGBM Xentropy
# 
# ### V8
# #### submit score: 0.0492
# X_test score: 0.04500
# * MatchID can get match totals, means etc
# * Create feature_list with columns, for Match totals and player percentage
# * Dropping previous custom features
# * Already at 62 columns
# * Only keeping 4 matchtypes from V1
# * Setting leaves back to 62, from original 124 to prevent overfitting
# * Test looks promising
# 
# ## V9 - current
# #### submit score: ......
# X_test score: ......
# * V8 was significant improvement
# * Setting leaves back to default (31)
# * Adding more features to feature list
# * Doing full train, so no X_test this time 
# 
# #### Ideas for further improvement
# * Handlle outliers
# * feature selection
# * There's probarbly some extra data to get with groupby groupId, kills per group etc
# * Add some custum features again
# * Do an average feature list
# * Look into memory management

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import warnings
warnings.simplefilter('ignore')


# In[ ]:


test_set = pd.read_csv("../input/test_V2.csv")
train = pd.read_csv("../input/train_V2.csv")


# In[ ]:


#only one missing WinplacePerc dropping that row
train.dropna(inplace=True)


# In[ ]:


print(f'Nan values: {train.isnull().values.any()}')


# #### Get Totals per match

# In[ ]:


feature_list = ['assists', 'boosts', 'damageDealt', 'DBNOs',
                'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
                'killStreaks', 'longestKill', 
                'rankPoints', 'revives', 'rideDistance', 'roadKills',
                'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance',
                'weaponsAcquired', 'winPoints']


# In[ ]:


# Match Totals for all columns in feature_list
for x in feature_list:
    train[f'match_{x}'] = train.groupby('matchId')[x].transform(sum)
    test_set[f'match_{x}'] = test_set.groupby('matchId')[x].transform(sum)


# In[ ]:


# Player rate vs match Total 
for x in feature_list:
    train[f'{x}_rate'] = (train[x] / train[f'match_{x}']).replace([np.inf, -np.inf, np.nan], 0)
    test_set[f'{x}_rate'] = (test_set[x] / test_set[f'match_{x}']).replace([np.inf, -np.inf, np.nan], 0)


# In[ ]:


print(f'Nan values: {train.isnull().values.any()}')


# In[ ]:


# Dropping Id's setting y,X and test
y = train['winPlacePerc']
X = train.drop(['winPlacePerc','matchType','Id','groupId','matchId'],axis=1)
test_pred = test_set.drop(['matchType','Id','groupId','matchId'],axis=1)


# In[ ]:


# Create dummy variables for Matchtypes
match = pd.get_dummies(train['matchType'],drop_first=False)
tmatch = pd.get_dummies(test_set['matchType'],drop_first=False)


# In[ ]:


# Matches to Keep
keep_list = ['squad-fpp','squad','solo-fpp','solo']
X = pd.concat([X,match[keep_list]],axis=1)
test_pred = pd.concat([test_pred,tmatch[keep_list]],axis=1)


# In[ ]:


# save this variable for the end, so we can delete test_set
test_id = test_set['Id']


# In[ ]:


# Clearing some RAM
del test_set
del train
del match
del tmatch
del feature_list
del keep_list


# In[ ]:


X.head()


# In[ ]:


# 
plt.figure(figsize=(28,22))
sns.heatmap(X.corr(),annot=True);


# In[ ]:


X.info()


# # Train/Test Split

# In[ ]:


# Not this time full  train
#from sklearn.model_selection import train_test_split


# In[ ]:


#X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0, random_state=0)


# # LightGBM

# In[ ]:


import lightgbm as lgb


# In[ ]:


# Xentropy seems to be outperforming other linear models
my_model = lgb.LGBMRegressor(n_estimators = 10000,
                             objective= 'xentropy',
                             metric='l1',
                             learning_rate = 0.05,
                             bagging_fraction = 0.9,
                             colsample_bytree = 0.8,
                             num_leaves=31)


# In[ ]:


my_model.fit(X, y,
            eval_set=[(X, y)],
            eval_metric= 'l1',
            early_stopping_rounds=15,
             )


# # Prediction and Evaluation
# #### Not this time, no Xtest

# In[ ]:


#from sklearn import metrics


# In[ ]:


#y_pred = my_model.predict(X_test)


# In[ ]:


#orig_mae = metrics.mean_absolute_error(y_test, y_pred)
#print('MAE:', orig_mae)
#print('MSE:', metrics.mean_squared_error(y_test, y_pred))
#print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


#print(f'y_pred range: {y_pred.min(), y_pred.max()}')


# In[ ]:


#plt.figure(figsize=(28,14))
#plt.scatter(y_test,y_pred,alpha=0.05);
#plt.title('y_test VS y_pred');


# In[ ]:


#plt.figure(figsize=(12,8));
#sns.distplot(y, label= 'y',bins=50);
#sns.distplot(y_pred, label='y_pred',bins=50);
#plt.title('y vs y_pred distribution');
#plt.legend();
#plt.savefig('y_vs_pred_dist.jpg');


# In[ ]:


#plt.figure(figsize=(10,7.5));
#plt.title('y_test vs y_pred error');
#sns.distplot((y_test-y_pred),bins=100);


# In[ ]:


lgb.plot_importance(my_model,figsize=(24,18));


# In[ ]:


lgb.plot_importance(my_model,importance_type = 'gain',figsize=(24,18));


# # Result

# In[ ]:


test_pred = my_model.predict(test_pred)


# In[ ]:


test_pred


# In[ ]:


print(f'Length test_pred: {len(test_pred)}')


# In[ ]:


print(f'test_pred range: {test_pred.min(),test_pred.max()}')


# In[ ]:


# just to be sure
test_pred[test_pred > 1] = 1
test_pred[test_pred < 0] = 0


# In[ ]:


plt.figure(figsize=(12,8));
sns.distplot(y, label= 'y',bins=50);
sns.distplot(test_pred, label='test_pred',bins=50);
plt.title('y vs test_pred distribution')
plt.legend();


# In[ ]:


df = pd.DataFrame(test_pred)
df.columns = ['winPlacePerc']


# In[ ]:


df = pd.concat([test_id,df],axis=1)


# In[ ]:


print(f'y:\n{y.describe()}\n')
print(f'test_pred:\n{df.describe()}')


# In[ ]:


df.head()


# In[ ]:


df.to_csv('xen_retry_V8.csv', index=False)

