#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

train_data_df = pd.read_csv('../input/train.csv')
test_data_df = pd.read_csv('../input/test.csv')


# In[ ]:


new_train_data_df = train_data_df.groupby(['matchId','groupId'])['assists', 'boosts', 'damageDealt', 'DBNOs','headshotKills', 'heals', 'killPlace', 'killPoints', 'kills','killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives','rideDistance', 'roadKills', 'swimDistance', 'teamKills','vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints'].agg('mean').reset_index()
new_train_data_df['winPlacePerc'] = train_data_df['winPlacePerc']
new_train_data_df['winPlacePerc'] = train_data_df.groupby(['matchId', 'groupId'])['winPlacePerc'].agg('mean').reset_index()['winPlacePerc']


# In[ ]:


new_test_data_df = test_data_df.groupby(['matchId','groupId'])['assists', 'boosts', 'damageDealt', 'DBNOs','headshotKills', 'heals', 'killPlace', 'killPoints', 'kills','killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives','rideDistance', 'roadKills', 'swimDistance', 'teamKills','vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints'].agg('mean').reset_index()


# In[ ]:


import numpy as np


# In[ ]:


train_data = train_data_df.values
test_data = test_data_df.values

train_features = new_train_data_df.values[:, 2:24][0:np.int32(0.8*len(new_train_data_df))]
train_targets = new_train_data_df.values[:, 24][0:np.int32(0.8*len(new_train_data_df))]

val_features = new_train_data_df.values[:, 2:24][np.int32(0.8*len(new_train_data_df)):len(new_train_data_df)]
val_targets = new_train_data_df.values[:, 24][np.int32(0.8*len(new_train_data_df)):len(new_train_data_df)]

test_features = new_test_data_df.values[:, 2:24]
test_data = test_data[:, 0:3]


# In[ ]:


import sklearn


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_features)

train_features = scaler.transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)


# In[ ]:


import catboost
from catboost import CatBoostRegressor

model = CatBoostRegressor(iterations=20000, learning_rate=0.1, eval_metric='MAE')
model.fit(train_features, train_targets, eval_set=(val_features, val_targets))


# In[ ]:


predictions = model.predict(test_features)


# In[ ]:


new_test_data_df['winPlacePercPred'] = predictions


# In[ ]:


group_preds = new_test_data_df.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()


# In[ ]:


group_preds = group_preds['winPlacePercPred']


# In[ ]:


dictionary = dict(zip(new_test_data_df['groupId'].values, group_preds))


# In[ ]:


new_preds = []

for i in test_data_df['groupId'].values:
    new_preds.append(dictionary[i])


# In[ ]:


test_data_df['winPlacePercPred'] = new_preds


# In[ ]:


test_data_df.sort_values(by=['matchId', 'groupId'])


# In[ ]:


import numpy as np


# In[ ]:


predictions = pd.DataFrame(np.transpose(np.array([test_data[:, 0], test_data_df['winPlacePercPred']])))
predictions.columns = ['Id', 'winPlacePerc']


# In[ ]:


predictions['Id'] = np.int32(predictions['Id'])


# In[ ]:


predictions.head(10)


# In[ ]:


predictions.to_csv('PUBG_preds3.csv', index=False)


# Courtesy to [Fernando Vendrameto](http://www.kaggle.com/fvendrameto/catboost-by-a-newbie) for giving me the idea to use Cat Boosting.
