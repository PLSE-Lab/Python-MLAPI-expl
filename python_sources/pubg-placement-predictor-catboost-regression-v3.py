#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

train_data_df = pd.read_csv('../input/train_V2.csv')
test_data_df = pd.read_csv('../input/test_V2.csv')


# In[ ]:


train_data_df.insert(loc=28, column='totalDistance', value=train_data_df['swimDistance'] + train_data_df['walkDistance'] + train_data_df['rideDistance'])
test_data_df.insert(loc=28, column='totalDistance', value=test_data_df['swimDistance'] + test_data_df['walkDistance'] + test_data_df['rideDistance'])


# In[ ]:


train_data_df = train_data_df.replace({'matchType' : {'crashfpp':0, 'crashtpp':1, 'duo':2, 'duo-fpp':3, 'flarefpp':4, 'flaretpp':5, 'normal-duo':6, 'normal-duo-fpp':7, 'normal-solo':8, 'normal-solo-fpp':9, 'normal-squad':10, 'normal-squad-fpp':11, 'solo':12, 'solo-fpp':13, 'squad':14, 'squad-fpp':15}})
test_data_df = test_data_df.replace({'matchType' : {'crashfpp':0, 'crashtpp':1, 'duo':2, 'duo-fpp':3, 'flarefpp':4, 'flaretpp':5, 'normal-duo':6, 'normal-duo-fpp':7, 'normal-solo':8, 'normal-solo-fpp':9, 'normal-squad':10, 'normal-squad-fpp':11, 'solo':12, 'solo-fpp':13, 'squad':14, 'squad-fpp':15}})


# In[ ]:


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# In[ ]:


import numpy as np

train_data_df = reduce_mem_usage(train_data_df)
test_data_df = reduce_mem_usage(test_data_df)


# In[ ]:


columns = train_data_df.columns[3:-1]


# In[ ]:


groupedGroups = train_data_df.groupby(['matchId','groupId'])[columns]

print("Add group mean features")
group_features = groupedGroups.agg('mean')
group_rank_features = group_features.groupby(['matchId'])[columns].rank(pct=True).reset_index()
features = pd.merge(group_features, group_rank_features, on=['matchId', 'groupId'], how='right', suffixes=['_group_mean', '_group_mean_rank'])

print("Add group max features")
group_features = groupedGroups.agg('max')
group_rank_features = group_features.groupby(['matchId'])[columns].rank(pct=True).reset_index()
features = pd.merge(features, group_features, on=['matchId', 'groupId'], how='right', suffixes=['', '_group_max'])
features = pd.merge(features, group_rank_features, on=['matchId', 'groupId'], how='right', suffixes=['', '_group_max_rank'])

print("Add group min features")
group_features = groupedGroups.agg('min')
group_rank_features = group_features.groupby(['matchId'])[columns].rank(pct=True).reset_index()
features = pd.merge(features, group_features, on=['matchId', 'groupId'], how='right', suffixes=['', '_group_min'])
features = pd.merge(features, group_rank_features, on=['matchId', 'groupId'], how='right', suffixes=['', '_group_min_rank'])

print("Add group size features")
group_features = train_data_df.groupby(['matchId','groupId']).size()
group_features.columns = ['matchId', 'groupId', 'group_size']
group_features = pd.DataFrame(group_features)
features = pd.merge(features, group_features, on=['matchId', 'groupId'], how='right', suffixes=['', ''])


# In[ ]:


groupedMatches = train_data_df.groupby(['matchId'])[columns]

print("Add match mean features")
match_features = groupedMatches.agg('mean')[columns].reset_index()
features = pd.merge(features, match_features, on=['matchId'], how='right', suffixes=['', '_match_mean'])

print("Add match size features")
match_features = train_data_df.groupby(['matchId']).size().reset_index()
match_features.columns = ['matchId', 'match_size']
features = pd.merge(features, match_features, on=['matchId'], how='right', suffixes=['', ''])


# In[ ]:


features


# In[ ]:


targets = train_data_df.groupby(['matchId', 'groupId'])['winPlacePerc'].agg('mean').reset_index()['winPlacePerc']


# In[ ]:


targets = targets.values


# In[ ]:


import sklearn


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

features = features.drop(['matchId', 'groupId'], axis=1).values
scaler = MinMaxScaler(feature_range=(-1, 1)).fit(features)
features = scaler.transform(features)


# In[ ]:


train_features = features[0:np.int32(0.8*len(features))]
train_targets = targets[0:np.int32(0.8*len(features))]

val_features = features[np.int32(0.8*len(features)):len(features)]
val_targets = targets[np.int32(0.8*len(features)):len(features)]


# In[ ]:


train_features = np.delete(train_features, 266069, 0)
train_targets = np.delete(train_targets, 266069, 0)


# In[ ]:


# targets = targets**2 - 1


# In[ ]:


import catboost
from catboost import CatBoostRegressor

model = CatBoostRegressor(learning_rate=0.1, max_depth=8, iterations=500, eval_metric='MAE')
model.fit(train_features, train_targets, eval_set=(val_features, val_targets))


# In[ ]:


groupedGroups = test_data_df.groupby(['matchId','groupId'])[columns]

print("Add group mean features")
group_features = groupedGroups.agg('mean')
group_rank_features = group_features.groupby(['matchId'])[columns].rank(pct=True).reset_index()
features = pd.merge(group_features, group_rank_features, on=['matchId', 'groupId'], how='right', suffixes=['_group_mean', '_group_mean_rank'])

print("Add group max features")
group_features = groupedGroups.agg('max')
group_rank_features = group_features.groupby(['matchId'])[columns].rank(pct=True).reset_index()
features = pd.merge(features, group_features, on=['matchId', 'groupId'], how='right', suffixes=['', '_group_max'])
features = pd.merge(features, group_rank_features, on=['matchId', 'groupId'], how='right', suffixes=['', '_group_max_rank'])

print("Add group min features")
group_features = groupedGroups.agg('min')
group_rank_features = group_features.groupby(['matchId'])[columns].rank(pct=True).reset_index()
features = pd.merge(features, group_features, on=['matchId', 'groupId'], how='right', suffixes=['', '_group_min'])
features = pd.merge(features, group_rank_features, on=['matchId', 'groupId'], how='right', suffixes=['', '_group_min_rank'])

print("Add group size features")
group_features = test_data_df.groupby(['matchId','groupId']).size()
group_features.columns = ['matchId', 'groupId', 'group_size']
group_features = pd.DataFrame(group_features)
features = pd.merge(features, group_features, on=['matchId', 'groupId'], how='right', suffixes=['', ''])


# In[ ]:


groupedMatches = test_data_df.groupby(['matchId'])[columns]

print("Add match mean features")
match_features = groupedMatches.agg('mean')[columns].reset_index()
features = pd.merge(features, match_features, on=['matchId'], how='right', suffixes=['', '_match_mean'])

print("Add match size features")
match_features = test_data_df.groupby(['matchId']).size().reset_index()
match_features.columns = ['matchId', 'match_size']
features = pd.merge(features, match_features, on=['matchId'], how='right', suffixes=['', ''])


# In[ ]:


features


# In[ ]:


test_features = features.drop(['matchId', 'groupId'], axis=1).values
test_features = scaler.transform(test_features)


# In[ ]:


predictions = model.predict(test_features)


# In[ ]:


# predictions = (predictions + 1)/2


# In[ ]:


features['winPlacePercPred'] = predictions
group_preds = features.groupby(['matchId', 'groupId'])['winPlacePercPred'].agg('mean').groupby(['matchId']).rank(pct=True)


# In[ ]:


dictionary = dict(zip(features['groupId'].values, group_preds.values))


# In[ ]:


individual_preds = []
    
for i in test_data_df['groupId'].values:
    individual_preds.append(dictionary[i])
    
test_data_df['winPlacePercPred'] = individual_preds


# In[ ]:


import numpy as np

predictions = pd.DataFrame(np.transpose(np.array([test_data_df['Id'], test_data_df['winPlacePercPred']])))
predictions.columns = ['Id', 'winPlacePerc']


# In[ ]:


maxPlaces = test_data_df['maxPlace'].values
new_predictions = predictions['winPlacePerc'].values

for i in range(0, len(test_data_df)):
    if maxPlaces[i] == 0:
        new_predictions[i] = 0.0
    if maxPlaces[i] == 1:
        new_predictions[i] = 1.0
    else:
        gap = 1.0 / (maxPlaces[i] - 1.0)
        new_predictions[i] = round(new_predictions[i]/gap)*gap


# In[ ]:


predictions['winPlacePerc'] = new_predictions


# In[ ]:


predictions.head(20)


# In[ ]:


predictions.to_csv('PUBG_preds.csv', index=False)


# In[ ]:




