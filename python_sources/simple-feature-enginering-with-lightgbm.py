#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# This notebooks
# contains 
# * season score aggregation
# * one hot encoding for location
# * simple prediction with LightGBM, Xgboost, Neural Network
# * lag features (we can use data of past seasons) 
# 
# will contain
# * enhancement of location for "Neutral"
# * model selection and hypermarameter chuning

# ## Import Library & Load Data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm


# In[ ]:


results = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneyCompactResults.csv')
seeds = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneySeeds.csv')
submission_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WSampleSubmissionStage1_2020.csv')


# In[ ]:


results.head()


# In[ ]:


seeds.head()


# In[ ]:


# it seems home team has high probability to win
results.WLoc.value_counts()


# In[ ]:


#convert results which have result columns
def convert_results(results):
    results['result'] = 1 # win
    Lresults = results.copy()
    Lresults['result'] = 0 # lose
    Lresults['WTeamID'] = results['LTeamID']
    Lresults['LTeamID'] = results['WTeamID']
    Lresults['WScore'] = results['LScore']
    Lresults['LScore'] = results['WScore']
    Lresults['WLoc'].replace({'H': 'A'}, inplace=True)
    results = pd.concat([results, Lresults]).reset_index(drop=True)
    
    results.rename(columns={'WTeamID': 'TeamID1', 'LTeamID': 'TeamID2'}, inplace=True)
    results.rename(columns={'WScore': 'Score1', 'LScore': 'Score2'}, inplace=True)
    results.rename(columns={'WLoc': 'Loc'}, inplace=True)

    return results

data = convert_results(results)


# In[ ]:


def get_seed(x):
    return int(x[1:])

#merge seed data
def setup_seed(results, seeds):
    data = pd.merge(results, seeds, left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'])
    data.rename(columns={'Seed': "Seed1"}, inplace=True)
    data.drop('TeamID', axis=1, inplace=True)

    data = pd.merge(data, seeds, left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'])
    data.rename(columns={'Seed': "Seed2"}, inplace=True)
    data.drop('TeamID', axis=1, inplace=True)
    data['Seed1'] = data['Seed1'].map(get_seed)
    data['Seed2'] = data['Seed2'].map(get_seed)
    
    data['seed_diff'] = data['Seed1'] - data['Seed2']
    
    return data


# In[ ]:


data = setup_seed(data, seeds)
data


# In[ ]:


# aggregate seasonal socre
# off_score is the score which a team got in the season
# def_score is the score which a team lost in the season

season_score = data.groupby(['Season', 'TeamID1']).mean().reset_index()[['Season', 'TeamID1', 'Score1', 'Score2']]
season_score.rename(columns={'TeamID1': 'TeamID', 'Score1': 'off_score', 'Score2': 'def_score'}, inplace=True)
season_score


# In[ ]:


def get_lag_score(scores):
    scores_1y = scores.copy()
    scores_1y['Season'] += 1
    scores_1y.rename(columns={'off_score': 'off_score_1y', 'def_score': 'def_score_1y'}, inplace=True)
    scores = scores.merge(scores_1y, on=['Season', 'TeamID'], how='left')

    return scores
season_score_lag = get_lag_score(season_score)
season_score_lag


# # Prepare data for training or testing
# Since given data for prediction is only season and teamIDs, we have to aggregate data

# In[ ]:


def convert_location(data):
    tmp = pd.get_dummies(data['Loc'], drop_first=True, prefix='location')
    data = pd.concat([data, tmp], axis=1)
    return data

#convert Location to one hot vectors
data = convert_location(data)


# In[ ]:


data.head()


# In[ ]:


delete_columns = [
    'DayNum',
    'TeamID1',
    'Score1',
    'TeamID2',
    'Score2',
    'Loc',
    'NumOT',
    'result'
]

def gen_datasets(data, season_score):
    #merge season scores
    data = pd.merge(data, season_score, left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'])
    data.rename(columns={'off_score': 'off_score1', 'def_score': 'def_score1'}, inplace=True)
    data.rename(columns={'off_score_1y': 'off_score1_1y', 'def_score_1y': 'def_score1_1y'}, inplace=True)

    data.drop('TeamID', axis=1, inplace=True)
    data = pd.merge(data, season_score, left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'])
    data.rename(columns={'off_score': 'off_score2', 'def_score': 'def_score2'}, inplace=True)
    data.rename(columns={'off_score_1y': 'off_score2_1y', 'def_score_1y': 'def_score2_1y'}, inplace=True)

    data.drop('TeamID', axis=1, inplace=True)
    
    #compare seasonal scores
    data['score_diff1'] = data['off_score1'] - data['def_score2']
    data['score_diff2'] = data['off_score2'] - data['def_score1']

    y = data['result']
    X = data.drop(delete_columns, axis=1)
   
    return X, y

train_x, train_y = gen_datasets(data, season_score_lag)
train_x


# In[ ]:


#only for Stage1
def prepare_test(df, data):
    df['Season'] = df['ID'].map(lambda x:int(x.split('_')[0]))
    df['TeamID1'] = df['ID'].map(lambda x:int(x.split('_')[1]))
    df['TeamID2'] = df['ID'].map(lambda x:int(x.split('_')[2]))
    #df.drop('ID', axis=1, inplace=True)
    
    tmp = data.drop_duplicates(['Season', 'TeamID1', 'Seed1'])
    df = pd.merge(df, tmp[['Season', 'TeamID1', 'Seed1']],  on=['Season', 'TeamID1'], how='inner')
    tmp = data.drop_duplicates(['Season', 'TeamID2', 'Seed2'])
    df = pd.merge(df, tmp[['Season', 'TeamID2', 'Seed2']],  on=['Season', 'TeamID2'], how='inner') 
    df['seed_diff'] = df['Seed1'] - df['Seed2']
    
    df['Loc'] = pd.merge(df, data[['Season', 'TeamID1', 'TeamID2', 'Loc']],  on=['Season', 'TeamID1', 'TeamID2'], how='inner')['Loc']
    df['Loc'].fillna('N', inplace=True)
    df = convert_location(df)

    #insert dummy columns
    df['result'] = 9999
    df['DayNum'] = 9999
    df['Score1'] = 9999
    df['Score2'] = 9999
    df['NumOT'] = 9999

    test_x, _ = gen_datasets(df, season_score_lag)
    
    return test_x
test_x = prepare_test(submission_df, data)


# In[ ]:


train_x.columns, test_x.columns


# # Build [LightGBM](https://lightgbm.readthedocs.io/en/latest/) Model
# 
# now, let's build a simple model and predict result
# 
# Note: To make codes simple, we accept leak at the moment

# In[ ]:


import lightgbm as lgbm
params_lgb = {'num_leaves': 127,
          'min_data_in_leaf': 10,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'logloss',
          "verbosity": 0
          }


# In[ ]:


train_lgb = lgbm.Dataset(train_x, train_y)
clf_lgb = lgbm.train(params_lgb, train_lgb)


# In[ ]:


pred_y_lgb = pd.Series(clf_lgb.predict(test_x.drop(['ID','Pred'], axis=1)), name='Pred')
pred_y_lgb = pd.concat([test_x['ID'], pred_y_lgb], axis=1)


# In[ ]:


pred_y_lgb.Pred.hist()


# In[ ]:


pd.Series(clf_lgb.feature_importance(), index=train_x.columns).sort_values().plot(kind='bar')


# # Build XGBModel

# In[ ]:


import xgboost as xgb
params_xgb = {'max_depth': 50,
              'objective': 'binary:logistic',
              'eta'      : 0.3,
              'subsample': 0.8,
              'lambda '  : 4,
              'eval_metric': 'logloss',
              'n_estimators': 1000,
              'colsample_bytree ': 0.9,
              'colsample_bylevel': 1
              }
train_xgb = xgb.DMatrix(train_x, train_y)
clf_xgb = xgb.train(params_xgb, train_xgb)


# In[ ]:


pred_y_xgb = pd.Series(clf_xgb.predict(xgb.DMatrix(test_x.drop(['ID','Pred'], axis=1))), name='Pred')
pred_y_xgb = pd.concat([test_x['ID'], pred_y_xgb], axis=1)


# In[ ]:


pred_y_xgb.Pred.hist()


# In[ ]:


pd.Series(clf_lgb.feature_importance(), index=train_x.columns).sort_values().plot(kind='bar')


# # Build NN model
# build NN model with keras w/ tensorflow backend

# In[ ]:


from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from sklearn.model_selection import train_test_split
def gen_NN_model():
    model = Sequential()
    model.add(Dense(128, input_shape=(16, )))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    #model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


# In[ ]:


NN_model = gen_NN_model()
NN_model.fit(train_x.fillna(-9999), train_y, batch_size=100,
            epochs=20, verbose=1)


# In[ ]:


pred_y_nn = pd.Series(NN_model.predict(test_x.drop(['ID','Pred'], axis=1).fillna(-9999)).reshape(len(test_x)), name='Pred')
pred_y_nn = pd.concat([test_x['ID'], pred_y_nn], axis=1)


# In[ ]:


pred_y_nn.Pred.hist()


# # Model Comparison

# In[ ]:


from sklearn.metrics import log_loss
def mixup_prediction(true_y, pred_lgb, pred_xgb, pred_nn):
    min_loss = 10
    min_w1 = 10
    min_w2 = 10
    for w1 in np.linspace(0, 1):
        for w2 in np.linspace(0, 1-w1):
            pred_mix = pred_lgb*w1 + pred_xgb*w2 + pred_nn*(1-w1-w2)
            ans = log_loss(true_y, pred_mix)
            if ans<min_loss: 
                min_w1 = w1
                min_w2 = w2
                min_loss = ans
    print('log_loss: {}, lgb_weight: {}, xgb_weight:{}, NN_weight:{}'.format(min_loss, min_w1, min_w2, 1-min_w1-min_w2))
    return min_w1, min_w2


# In[ ]:


from sklearn.model_selection import KFold
import warnings
warnings.simplefilter('ignore', FutureWarning)

kf = KFold(n_splits=5)
loss_lgb = []
loss_xgb = []
loss_nn = []
weights = []

for train_index, val_index in kf.split(train_x):
    tr_x = train_x.iloc[train_index]
    va_x = train_x.iloc[val_index]
    tr_y = train_y.iloc[train_index]
    va_y = train_y.iloc[val_index]
    
    #light gbm
    train_lgb = lgbm.Dataset(tr_x, tr_y)
    lgb_clf = lgbm.train(params_lgb, train_lgb)
    pred_lgb = lgb_clf.predict(va_x)
    loss_lgb.append(log_loss(va_y, pred_lgb))
    
    #xgboost
    train_xgb = xgb.DMatrix(tr_x, tr_y)
    clf_xgb = xgb.train(params_xgb, train_xgb)
    pred_xgb = pd.Series(clf_xgb.predict(xgb.DMatrix(va_x)), name='Pred')
    loss_xgb.append(log_loss(va_y, pred_xgb))
    
    #neural network
    NN_model = gen_NN_model()
    NN_model.fit(tr_x.fillna(-9999), tr_y, batch_size=100, epochs=20, verbose=0)
    pred_nn = NN_model.predict(va_x.fillna(-9999)).reshape(len(va_x))
    loss_nn.append(log_loss(va_y, pred_nn))
    
    weights.append(mixup_prediction(va_y, pred_lgb, pred_xgb, pred_nn))


# In[ ]:


np.mean(loss_lgb), np.mean(loss_xgb), np.mean(loss_nn)


# xgboost seems best model...[](http://)

# In[ ]:


test_x['Pred'] = clf_xgb.predict(xgb.DMatrix(test_x.drop(['ID','Pred'], axis=1)))


# In[ ]:


test_x['Pred'].hist()


# In[ ]:


test_x['Pred'] = test_x['Pred'].clip(0, 1)
test_x['Pred'].hist()


# In[ ]:


test_x[['ID', 'Pred']].to_csv('submission.csv', index=False)


# In[ ]:




