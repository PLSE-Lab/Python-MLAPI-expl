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


path = '../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/'

tourney_result = pd.read_csv(path + 'WDataFiles_Stage1/WNCAATourneyCompactResults.csv')
tourney_seed = pd.read_csv(path + 'WDataFiles_Stage1/WNCAATourneySeeds.csv')

season_result = pd.read_csv(path + 'WDataFiles_Stage1/WRegularSeasonCompactResults.csv')


# In[ ]:


test_df= pd.read_csv(path +'WSampleSubmissionStage1_2020.csv')


# In[ ]:


# deleting unnecessary columns
tourney_result = tourney_result.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)
tourney_result


# In[ ]:


tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney_result.rename(columns={'Seed':'WSeed'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney_result.rename(columns={'Seed':'LSeed'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_result


# In[ ]:


def get_seed(x):
    return int(x[1:3])

tourney_result['WSeed'] = tourney_result['WSeed'].map(lambda x: get_seed(x))
tourney_result['LSeed'] = tourney_result['LSeed'].map(lambda x: get_seed(x))
tourney_result


# In[ ]:


season_win_result = season_result[['Season', 'WTeamID', 'WScore']]
season_lose_result = season_result[['Season', 'LTeamID', 'LScore']]
season_win_result.rename(columns={'WTeamID':'TeamID', 'WScore':'Score'}, inplace=True)
season_lose_result.rename(columns={'LTeamID':'TeamID', 'LScore':'Score'}, inplace=True)
season_result = pd.concat((season_win_result, season_lose_result)).reset_index(drop=True)
season_result


# In[ ]:


season_score = season_result.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()
season_score


# In[ ]:


tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney_result.rename(columns={'Score':'WScoreT'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney_result.rename(columns={'Score':'LScoreT'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_result


# In[ ]:


tourney_win_result = tourney_result.drop(['Season', 'WTeamID', 'LTeamID'], axis=1)
tourney_win_result.rename(columns={'WSeed':'Seed1', 'LSeed':'Seed2', 'WScoreT':'ScoreT1', 'LScoreT':'ScoreT2'}, inplace=True)
tourney_win_result


# In[ ]:


tourney_lose_result = tourney_win_result.copy()
tourney_lose_result['Seed1'] = tourney_win_result['Seed2']
tourney_lose_result['Seed2'] = tourney_win_result['Seed1']
tourney_lose_result['ScoreT1'] = tourney_win_result['ScoreT2']
tourney_lose_result['ScoreT2'] = tourney_win_result['ScoreT1']
tourney_lose_result


# In[ ]:


tourney_win_result['Seed_diff'] = tourney_win_result['Seed1'] - tourney_win_result['Seed2']
tourney_win_result['ScoreT_diff'] = tourney_win_result['ScoreT1'] - tourney_win_result['ScoreT2']
tourney_lose_result['Seed_diff'] = tourney_lose_result['Seed1'] - tourney_lose_result['Seed2']
tourney_lose_result['ScoreT_diff'] = tourney_lose_result['ScoreT1'] - tourney_lose_result['ScoreT2']


# In[ ]:


test_df['Season'] = test_df['ID'].map(lambda x: int(x[:4]))
test_df['WTeamID'] = test_df['ID'].map(lambda x: int(x[5:9]))
test_df['LTeamID'] = test_df['ID'].map(lambda x: int(x[10:14]))
test_df


# In[ ]:


test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Seed':'Seed1'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Seed':'Seed2'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Score':'ScoreT1'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Score':'ScoreT2'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df


# In[ ]:


test_df['Seed1'] = test_df['Seed1'].map(lambda x: get_seed(x))
test_df['Seed2'] = test_df['Seed2'].map(lambda x: get_seed(x))
test_df['Seed_diff'] = test_df['Seed1'] - test_df['Seed2']
test_df['ScoreT_diff'] = test_df['ScoreT1'] - test_df['ScoreT2']
test_df = test_df.drop(['ID', 'Pred', 'Season', 'WTeamID', 'LTeamID'], axis=1)
test_df


# In[ ]:


tourney_win_result['result'] = 1
tourney_lose_result['result'] = 0
tourney_result = pd.concat((tourney_win_result, tourney_lose_result)).reset_index(drop=True)
tourney_result


# In[ ]:


X_train = tourney_result.drop('result', axis=1)
y_train = tourney_result.result


# In[ ]:


X_train.shape


# In[ ]:


X_train


# In[ ]:


test_df


# In[ ]:


from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier
import gc


# In[ ]:


params_lgb = {'num_leaves': 400,
              'min_child_weight': 0.034,
              'feature_fraction': 0.379,
              'bagging_fraction': 0.418,
              'min_data_in_leaf': 106,
              'objective': 'binary',
              'max_depth': 50,
              'learning_rate': 0.0068,
              "boosting_type": "gbdt",
              "bagging_seed": 11,
              "metric": 'logloss',
              "verbosity": -1,
              'reg_alpha': 0.3899,
              'reg_lambda': 0.648,
              'random_state': 47,
              }

params_xgb = {'colsample_bytree': 0.8,                 
              'learning_rate': 0.0004,
              'max_depth': 31,
              'subsample': 1,
              'objective':'binary:logistic',
              'eval_metric':'logloss',
              'min_child_weight':3,
              'gamma':0.25,
              'n_estimators':5000
              }


# In[ ]:


NFOLDS = 200
folds = KFold(n_splits=NFOLDS)

columns = X_train.columns
splits = folds.split(X_train, y_train)


# In[ ]:


y_preds_lgb = np.zeros(test_df.shape[0])
y_oof_lgb = np.zeros(X_train.shape[0])


# In[ ]:


for fold_n, (train_index, valid_index) in enumerate(splits):
    print('Fold:',fold_n+1)
    X_train1, X_valid1 = X_train[columns].iloc[train_index], X_train[columns].iloc[valid_index]
    y_train1, y_valid1 = y_train.iloc[train_index], y_train.iloc[valid_index]
    
    dtrain = lgb.Dataset(X_train1, label=y_train1)
    dvalid = lgb.Dataset(X_valid1, label=y_valid1)

    clf = lgb.train(params_lgb, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200)
    
    y_pred_valid = clf.predict(X_valid1)
    y_oof_lgb[valid_index] = y_pred_valid
    
    y_preds_lgb += clf.predict(test_df) / NFOLDS
    
    del X_train1, X_valid1, y_train1, y_valid1
    gc.collect()


# In[ ]:


#NFOLDS = 100
#folds = KFold(n_splits=NFOLDS)

#columns = X_train.columns
#splits = folds.split(X_train, y_train)

#y_preds_xgb = np.zeros(test_df.shape[0])
#y_oof_xgb = np.zeros(X_train.shape[0])
  
#for fold_n, (train_index, valid_index) in enumerate(splits):
#    print('Fold:',fold_n+1)
#    X_train2, X_valid2 = X_train[columns].iloc[train_index], X_train[columns].iloc[valid_index]
#    y_train2, y_valid2 = y_train.iloc[train_index], y_train.iloc[valid_index]
    
#    train_set = xgb.DMatrix(X_train2, y_train2)
#    val_set = xgb.DMatrix(X_valid2, y_valid2)
#    test_set = xgb.DMatrix(test_df)
#    
#    clf = xgb.train(params_xgb, train_set,num_boost_round=5000, evals=[(train_set, 'train'), (val_set, 'val')], early_stopping_rounds=100, verbose_eval=100)
    
#   y_preds_xgb += clf.predict(test_set) / NFOLDS
    
    del X_train2, X_valid2, y_train2, y_valid2
    gc.collect()


# In[ ]:


#print(len(y_preds_xgb))
print(len(y_preds_lgb))


# In[ ]:


submission_df = pd.read_csv(path + 'WSampleSubmissionStage1_2020.csv')
#submission_df['Pred'] = 0.94*y_preds_lgb + 0.06*y_preds_xgb
submission_df['Pred'] = y_preds_lgb
submission_df


# In[ ]:


submission_df['Pred'].hist()


# In[ ]:


submission_df.to_csv('submission.csv', index=False)

