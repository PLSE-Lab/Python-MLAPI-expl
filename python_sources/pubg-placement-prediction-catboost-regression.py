#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

train_data_df = pd.read_csv('../input/train.csv')
test_data_df = pd.read_csv('../input/test.csv')


# In[ ]:


mean_group_features = train_data_df.groupby(['matchId','groupId'])[train_data_df.columns[3:-1]].agg('mean').reset_index().loc[:, 'assists':'winPoints']
max_group_features = train_data_df.groupby(['matchId','groupId'])[train_data_df.columns[3:-1]].agg('max').reset_index().loc[:, 'assists':'winPoints']
min_group_features = train_data_df.groupby(['matchId','groupId'])[train_data_df.columns[3:-1]].agg('min').reset_index().loc[:, 'assists':'winPoints']
std_group_features = train_data_df.groupby(['matchId','groupId'])[train_data_df.columns[3:-1]].agg('std').reset_index().loc[:, 'assists':'winPoints']


# In[ ]:


features_one = mean_group_features.join(max_group_features, lsuffix='_mean', rsuffix='_max')
features_two = min_group_features.join(std_group_features, lsuffix='_min', rsuffix='_std')
features = features_one.join(features_two)
features = features.fillna(0.0)
features


# In[ ]:


targets = train_data_df.groupby(['matchId', 'groupId'])['winPlacePerc'].agg('mean').reset_index()['winPlacePerc']
targets


# In[ ]:


# import numpy as np

# train_features = features.values[0:np.int32(0.8*len(features))]
# train_targets = targets.values[0:np.int32(0.8*len(features))]

# val_features = features.values[np.int32(0.8*len(features)):len(features)]
# val_targets = targets.values[np.int32(0.8*len(features)):len(features)]


# In[ ]:


import sklearn


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1)).fit(features.values)

# train_features = scaler.transform(train_features)
# val_features = scaler.transform(val_features)


# In[ ]:


import catboost
from catboost import CatBoostRegressor

import skopt
from skopt import BayesSearchCV
from sklearn.model_selection import KFold

bayes_cv_tuner = BayesSearchCV(
    estimator = CatBoostRegressor(iterations = 1500, eval_metric='MAE')
    ,
    search_spaces = {
        'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        'max_depth': (4, 6),
        
    },    
    scoring = 'neg_mean_absolute_error',
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 1,
    n_iter = 6,   
    verbose = 0,
    refit = True,
    random_state = 42
)


# In[ ]:


get_ipython().run_line_magic('env', 'JOBLIB_TEMP_FOLDER=/tmp')


# In[ ]:


# Fit the model
bayes_cv_tuner.fit(scaler.transform(features.values), targets.values)


# In[ ]:


model = bayes_cv_tuner.best_estimator_


# In[ ]:


mean_group_features = test_data_df.groupby(['matchId','groupId'])[train_data_df.columns[3:-1]].agg('mean').reset_index().loc[:, 'assists':'winPoints']
max_group_features = test_data_df.groupby(['matchId','groupId'])[train_data_df.columns[3:-1]].agg('max').reset_index().loc[:, 'assists':'winPoints']
min_group_features = test_data_df.groupby(['matchId','groupId'])[train_data_df.columns[3:-1]].agg('min').reset_index().loc[:, 'assists':'winPoints']
std_group_features = test_data_df.groupby(['matchId','groupId'])[train_data_df.columns[3:-1]].agg('std').reset_index().loc[:, 'assists':'winPoints']


# In[ ]:


features_one = mean_group_features.join(max_group_features, lsuffix='_mean', rsuffix='_max')
features_two = min_group_features.join(std_group_features, lsuffix='_min', rsuffix='_std')
features = features_one.join(features_two)
features = features.fillna(0.0)
features


# In[ ]:


matches = test_data_df.groupby(['matchId', 'groupId'])['matchId'].agg('mean').values


# In[ ]:


groups = test_data_df.groupby(['matchId', 'groupId'])['groupId'].agg('mean').values


# In[ ]:


test_features = features.values
test_features = scaler.transform(test_features)


# In[ ]:


predictions = model.predict(test_features)


# In[ ]:


predictions


# In[ ]:


features['winPlacePercPred'] = predictions
features['matchId'] = matches
features['groupId'] = groups
group_preds = features.groupby(['matchId', 'groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
group_preds = group_preds['winPlacePercPred']


# In[ ]:


test_data_df = test_data_df.sort_values(['matchId', 'groupId'])


# In[ ]:


dictionary = dict(zip(features['groupId'].values, group_preds))


# In[ ]:


new_ranking_preds = []
    
for i in test_data_df['groupId'].values:
    new_ranking_preds.append(dictionary[i])
    
test_data_df['winPlacePercPred'] = new_ranking_preds


# In[ ]:


import numpy as np

predictions = pd.DataFrame(np.transpose(np.array([test_data_df.loc[:, 'Id'], test_data_df['winPlacePercPred']])))
predictions.columns = ['Id', 'winPlacePerc']
predictions['Id'] = np.int32(predictions['Id'])
predictions = predictions.sort_values(by=['Id'])

predictions.head(10)


# In[ ]:


predictions.to_csv('PUBG_preds.csv', index=False)


# In[ ]:




