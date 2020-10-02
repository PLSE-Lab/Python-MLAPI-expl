#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


tr = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')
ts = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv')


# In[ ]:


tr


# In[ ]:


ts


# In[ ]:


tr = tr.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)


# In[ ]:


tr = pd.merge(tr, ts, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
tr.rename(columns={'Seed':'WSeed'}, inplace=True)

tr = tr.drop('TeamID', axis=1)

tr = pd.merge(tr, ts, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
tr.rename(columns={'Seed':'LSeed'}, inplace=True)

tr = tr.drop('TeamID', axis=1)


# In[ ]:


tr


# In[ ]:


def get_seed(x):
    return int(x[1:3])

tr['WSeed'] = tr['WSeed'].map(lambda x: get_seed(x))
tr['LSeed'] = tr['LSeed'].map(lambda x: get_seed(x))


# In[ ]:


sr = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')


# In[ ]:


sr


# In[ ]:


swr = sr[['Season', 'WTeamID', 'WScore']]
slr = sr[['Season', 'LTeamID', 'LScore']]


# In[ ]:


swr


# In[ ]:


slr


# In[ ]:


swr.rename(columns={'WTeamID':'TeamID', 'WScore':'Score'}, inplace=True)
slr.rename(columns={'LTeamID':'TeamID', 'LScore':'Score'}, inplace=True)


# In[ ]:


slr


# In[ ]:


swr


# In[ ]:


sr = pd.concat((swr, slr)).reset_index(drop=True)
ss = sr.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()


# In[ ]:


sr


# In[ ]:


ss


# In[ ]:


tr = pd.merge(tr, ss, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
tr.rename(columns={'Score':'WScoreT'}, inplace=True)


# In[ ]:


tr = tr.drop('TeamID', axis=1)


# In[ ]:


tr = pd.merge(tr, ss, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
tr.rename(columns={'Score':'LScoreT'}, inplace=True)


# In[ ]:


tr = tr.drop('TeamID', axis=1)


# In[ ]:


tr


# In[ ]:


twr = tr.drop(['Season', 'WTeamID', 'LTeamID'], axis=1)
twr.rename(columns={'WSeed':'Seed1', 'LSeed':'Seed2', 'WScoreT':'ScoreT1', 'LScoreT':'ScoreT2'}, inplace=True)


# In[ ]:


twr


# In[ ]:


tlr = twr.copy()
tlr['Seed1'] = twr['Seed2']
tlr['Seed2'] = twr['Seed1']
tlr['ScoreT1'] = twr['ScoreT2']
tlr['ScoreT2'] = twr['ScoreT1']


# In[ ]:


twr['Seed_diff'] = twr['Seed1'] - twr['Seed2']
twr['ScoreT_diff'] = twr['ScoreT1'] - twr['ScoreT2']
tlr['Seed_diff'] = tlr['Seed1'] - tlr['Seed2']
tlr['ScoreT_diff'] = tlr['ScoreT1'] - tlr['ScoreT2']


# In[ ]:


twr['result'] = 1
tlr['result'] = 0
train_df = pd.concat((twr, tlr)).reset_index(drop=True)
train_df


# In[ ]:


test_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')


# In[ ]:


test_df


# In[ ]:


test_df['Season'] = test_df['ID'].map(lambda x: int(x[:4]))
test_df['WTeamID'] = test_df['ID'].map(lambda x: int(x[5:9]))
test_df['LTeamID'] = test_df['ID'].map(lambda x: int(x[10:14]))


# In[ ]:


test_df = pd.merge(test_df, ts, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Seed':'Seed1'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, ts, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Seed':'Seed2'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, ss, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Score':'ScoreT1'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, ss, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Score':'ScoreT2'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)


# In[ ]:


test_df['Seed1'] = test_df['Seed1'].map(lambda x: get_seed(x))
test_df['Seed2'] = test_df['Seed2'].map(lambda x: get_seed(x))
test_df['Seed_diff'] = test_df['Seed1'] - test_df['Seed2']
test_df['ScoreT_diff'] = test_df['ScoreT1'] - test_df['ScoreT2']
test_df = test_df.drop(['ID', 'Pred', 'Season', 'WTeamID', 'LTeamID'], axis=1)
test_df


# In[ ]:


X = train_df.drop('result', axis=1)
y = train_df.result


# In[ ]:


sub = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')
MNCAATourneyCompactResults = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')
sub = pd.concat([sub, sub['ID'].str.split('_', expand=True).rename(columns={0: 'Season', 1: 'Team1', 2: 'Team2'}).astype(np.int64)], axis=1)
merge = pd.merge(sub, MNCAATourneyCompactResults[['Season', 'WTeamID', 'LTeamID']], how='left', left_on=['Season', 'Team1', 'Team2'], right_on=['Season', 'WTeamID', 'LTeamID'])
sub.loc[~merge['WTeamID'].isnull(), 'Pred'] = 1
merge = pd.merge(sub, MNCAATourneyCompactResults[['Season', 'WTeamID', 'LTeamID']], how='left', left_on=['Season', 'Team2', 'Team1'], right_on=['Season', 'WTeamID', 'LTeamID'])
sub.loc[~merge['WTeamID'].isnull(), 'Pred'] = 0
sub = sub.drop(['Season', 'Team1', 'Team2'], axis=1)


# In[ ]:


sub.info()


# In[ ]:


sub.shape


# In[ ]:


from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor

model = GradientBoostingRegressor(loss='ls',
    learning_rate=0.1,
    n_estimators=4556,
    subsample=1.0,
    criterion='friedman_mse',
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_depth=3,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    init=None,
    random_state=None,
    max_features=None,
    alpha=0.9,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False,
    presort='deprecated',
    validation_fraction=0.1,
    n_iter_no_change=None,
    tol=0.0001,
    ccp_alpha=0.0)

model.fit(X,y)


# In[ ]:





# In[ ]:





# In[ ]:


y_pred = model.predict(test_df)


# In[ ]:


sub['Pred'] = y_pred
sub.head()


# In[ ]:


sub.to_csv('submission.csv', index=False)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor

model = GradientBoostingRegressor(loss='ls',
    learning_rate=0.1,
    n_estimators=25000,
    subsample=1.0,
    criterion='friedman_mse',
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_depth=10,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    init=None,
    random_state=None,
    max_features=None,
    alpha=0.9,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False,
    presort='deprecated',
    validation_fraction=0.1,
    n_iter_no_change=None,
    tol=0.0001,
    ccp_alpha=0.0)

model.fit(X,y)


# In[ ]:


y_pred = model.predict(test_df)

sub['Pred'] = y_pred
sub.head()

sub.to_csv('Results.csv', index=False)


# In[ ]:


from xgboost import XGBRegressor,XGBRFRegressor,XGBModel


# In[ ]:


model = XGBRegressor()
model.fit(X,y)


# In[ ]:


y_pred = model.predict(test_df)

sub['Pred'] = y_pred
sub.head()

sub.to_csv('XGBoost.csv', index=False)


# In[ ]:


model = XGBRFRegressor(learning_rate=1,
    subsample=0.8,
    colsample_bynode=0.8,
    reg_lambda=1e-05)
model.fit(X,y)


# In[ ]:


y_pred = model.predict(test_df)

sub['Pred'] = y_pred
sub.head()

sub.to_csv('xgbrf.csv', index=False)


# In[ ]:


from lightgbm import LGBMRegressor


# In[ ]:


model = LGBMRegressor(boosting_type='gbdt',
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.1,
    n_estimators=100000,
    subsample_for_bin=200000,
    objective=None,
    class_weight=None,
    min_split_gain=0.0,
    min_child_weight=0.001,
    min_child_samples=20,
    subsample=1.0,
    subsample_freq=0,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=None,
    n_jobs=-1,
    silent=True,
    importance_type='split')
model.fit(X,y)


# In[ ]:


y_pred = model.predict(test_df)

sub['Pred'] = y_pred
sub.head()

sub.to_csv('FinalSubmission.csv', index=False)


# In[ ]:


from sklearn.ensemble import BaggingRegressor


# In[ ]:


model = BaggingRegressor(base_estimator=None,
    n_estimators=100000,
    max_samples=1.0,
    max_features=1.0,
    bootstrap=True,
    bootstrap_features=False,
    oob_score=False,
    warm_start=False,
    n_jobs=None,
    random_state=None,
    verbose=0)
model.fit(X,y)


# In[ ]:


y_pred = model.predict(test_df)

sub['Pred'] = y_pred
sub.head()

sub.to_csv('Bagging.csv', index=False)


# In[ ]:




