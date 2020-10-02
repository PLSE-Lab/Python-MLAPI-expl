#!/usr/bin/env python
# coding: utf-8

#     Based on this kernel:
#     https://www.kaggle.com/artgor/march-madness-2020-ncaam-eda-and-baseline
#     
#     Looking through some of the notebooks from previous years, I noticed people getting better scores by using a spline on their test predictions.  In the discussion
#     section people are also recommending you clip your predictions to avoid getting a large penalty when your model predicts too confidently a team that doesn't win.
#     
#     In this kernel I compare the base prediction scores with the scores where the predictions have been adjusted by the spline and optimal clips.  The spline and clips are fit 
#     on a holdback set using the games from 2019 since the public leaderboard doesn't seem very useful in this competition.
#     
#     This is also my first real attempt at writing a class in python as a mostly self taught programmer so your comments/criticism/hate mail is welcome.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


tourney_result = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv')
tourney_seed = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv')
season_result = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv')
test_df = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')


# In[ ]:


season_win_result = season_result[['Season', 'WTeamID', 'WScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',
                                  'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']]
season_lose_result = season_result[['Season', 'LTeamID', 'LScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR',
                                   'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']]
season_win_result.rename(columns={'WTeamID':'TeamID', 'WScore':'Score', 'WFGM':'FGM', 'WFGA':'FGA', 'WFGM3':'FGM3', 'WFGA3':'FGA3',
                                  'WFTM':'FTM', 'WFTA':'FTA', 'WOR':'OR', 'WDR':'DR', 'WAst':'Ast', 'WTO':'TO', 'WStl':'Stl',
                                  'WBlk':'Blk', 'WPF':'PF'}, inplace=True)
season_lose_result.rename(columns={'LTeamID':'TeamID', 'LScore':'Score', 'LFGM':'FGM', 'LFGA':'FGA', 'LFGM3':'FGM3', 'LFGA3':'FGA3',
                                  'LFTM':'FTM', 'LFTA':'FTA', 'LOR':'OR', 'LDR':'DR', 'LAst':'Ast', 'LTO':'TO', 'LStl':'Stl',
                                  'LBlk':'Blk', 'LPF':'PF'}, inplace=True)
season_result = pd.concat((season_win_result, season_lose_result)).reset_index(drop=True)


# In[ ]:


tourney_result['Score_difference'] = tourney_result['WScore'] - tourney_result['LScore']

tourney_result = tourney_result[['Season', 'WTeamID', 'LTeamID', 'Score_difference']]

tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney_result.rename(columns={'Seed':'WSeed'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney_result.rename(columns={'Seed':'LSeed'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_result['WSeed'] = tourney_result['WSeed'].apply(lambda x: int(x[1:3]))
tourney_result['LSeed'] = tourney_result['LSeed'].apply(lambda x: int(x[1:3]))
print(tourney_result.info(null_counts=True))


# In[ ]:


test_df['Season'] = test_df['ID'].map(lambda x: int(x[:4]))
test_df['WTeamID'] = test_df['ID'].map(lambda x: int(x[5:9]))
test_df['LTeamID'] = test_df['ID'].map(lambda x: int(x[10:14]))

test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Seed':'Seed1'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Seed':'Seed2'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)


# In[ ]:


for col in season_result.columns[2:]:
    season_result_map_mean = season_result.groupby(['Season', 'TeamID'])[col].mean().reset_index()

    tourney_result = pd.merge(tourney_result, season_result_map_mean, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
    tourney_result.rename(columns={f'{col}':f'W{col}MeanT'}, inplace=True)
    tourney_result = tourney_result.drop('TeamID', axis=1)
    tourney_result = pd.merge(tourney_result, season_result_map_mean, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
    tourney_result.rename(columns={f'{col}':f'L{col}MeanT'}, inplace=True)
    tourney_result = tourney_result.drop('TeamID', axis=1)
        
    test_df = pd.merge(test_df, season_result_map_mean, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
    test_df.rename(columns={f'{col}':f'W{col}MeanT'}, inplace=True)
    test_df = test_df.drop('TeamID', axis=1)
    
    test_df = pd.merge(test_df, season_result_map_mean, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
    test_df.rename(columns={f'{col}':f'L{col}MeanT'}, inplace=True)
    test_df = test_df.drop('TeamID', axis=1)


# In[ ]:


tourney_win_result = tourney_result.drop(['WTeamID', 'LTeamID'], axis=1)
for col in tourney_win_result.columns[2:]:
    if col[0] == 'W':
        tourney_win_result.rename(columns={f'{col}':f'{col[1:]+"1"}'}, inplace=True)
    elif col[0] == 'L':
        tourney_win_result.rename(columns={f'{col}':f'{col[1:]+"2"}'}, inplace=True)
        
tourney_lose_result = tourney_win_result.copy()
for col in tourney_lose_result.columns:
    if col[-1] == '1':
        col2 = col[:-1] + '2'
        tourney_lose_result[col] = tourney_win_result[col2]
        tourney_lose_result[col2] = tourney_win_result[col]
tourney_lose_result.columns


# In[ ]:


tourney_win_result['Seed_diff'] = tourney_win_result['Seed1'] - tourney_win_result['Seed2']
tourney_win_result['ScoreMeanT_diff'] = tourney_win_result['ScoreMeanT1'] - tourney_win_result['ScoreMeanT2']

tourney_lose_result['Seed_diff'] = tourney_lose_result['Seed1'] - tourney_lose_result['Seed2']
tourney_lose_result['ScoreMeanT_diff'] = tourney_lose_result['ScoreMeanT1'] - tourney_lose_result['ScoreMeanT2']

tourney_lose_result['Score_difference'] = -tourney_lose_result['Score_difference']
tourney_win_result['result'] = 1
tourney_lose_result['result'] = 0

tourney_result = pd.concat((tourney_win_result, tourney_lose_result)).reset_index(drop=True)


# In[ ]:


for col in test_df.columns[2:]:
    if col[0] == 'W':
        test_df.rename(columns={f'{col}':f'{col[1:]+"1"}'}, inplace=True)
    elif col[0] == 'L':
        test_df.rename(columns={f'{col}':f'{col[1:]+"2"}'}, inplace=True)
        
test_df['Seed1'] = test_df['Seed1'].apply(lambda x: int(x[1:3]))
test_df['Seed2'] = test_df['Seed2'].apply(lambda x: int(x[1:3]))
test_df['Seed_diff'] = test_df['Seed1'] - test_df['Seed2']
test_df['ScoreMeanT_diff'] = test_df['ScoreMeanT1'] - test_df['ScoreMeanT2']
test_df = test_df.drop(['ID', 'Pred', 'Season'], axis=1)


# In[ ]:


from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from functools import partial

def minimize_clipper(labels, preds, clips):
    clipped = np.clip(preds, clips[0], clips[1])
    return log_loss(labels, clipped)

def spline_model(labels, preds):
    comb = pd.DataFrame({'labels':labels, 'preds':preds})
    comb = comb.sort_values(by='preds').reset_index(drop=True)
    spline_model = UnivariateSpline(comb['preds'].values, comb['labels'].values)
    adjusted = spline_model(preds)
    return spline_model, log_loss(labels, adjusted)

class NCAA_model():
    
    def __init__(self, params, train_df, test_df, use_holdback=True, regression=False, verbose=True):
        self.params = params
        self.verbose = verbose
        self.test_df = test_df
        self.has_trained_models = False
        self.models = []
        if use_holdback == True:
            self.use_holdback=2019
        else:
            self.use_holdback = use_holdback
            
        if regression:
            self.params['objective'] = 'regression'
            self.params['metric'] = 'mae'
            self.target = 'Score_difference'
            self.eval_func = mean_absolute_error
        else:
            self.params['objective'] = 'binary'
            self.params['metric'] = 'logloss'
            self.target = 'result'
            self.eval_func = log_loss
            
        if not self.verbose:
            self.params['verbosity'] = -1 
            
        if self.use_holdback:
            self.holdback_df = train_df.query(f'Season == {self.use_holdback}')
            self.holdback_target = self.holdback_df[self.target]
            self.train_df = train_df.query(f'Season != {self.use_holdback}')
        else:
            self.train_df = train_df
            
        self.target = self.train_df[self.target]
        
    def train(self, features, n_splits, n_boost_round=5000, early_stopping_rounds=None, verbose_eval=1000):
        self.feature_importances = pd.DataFrame(columns=features)
        self.preds = np.zeros(shape=(self.test_df.shape[0]))
        self.train_preds = np.zeros(shape=self.train_df.shape[0])
        self.oof = np.zeros(shape=(self.train_df.shape[0]))
        if self.use_holdback:
            self.holdback_preds = np.zeros(shape=(self.holdback_df.shape[0]))
        
        cv = KFold(n_splits=n_splits, random_state=0, shuffle=True)        
        for fold, (tr_idx, v_idx) in enumerate(cv.split(self.train_df)):
            if self.verbose:
                print(f'Fold: {fold}')
                
            x_train, y_train = self.train_df.iloc[tr_idx][features], self.target.iloc[tr_idx]
            x_valid, y_valid = self.train_df.iloc[v_idx][features], self.target.iloc[v_idx]
            X_t = lgb.Dataset(x_train, y_train)
            X_v = lgb.Dataset(x_valid, y_valid)
            
            if self.has_trained_models:
                self.models[fold] = lgb.train(params, X_t, num_boost_round = n_boost_round, early_stopping_rounds=early_stopping_rounds,
                                                  valid_sets = [X_t, X_v], verbose_eval=(verbose_eval if self.verbose else None),
                                                                                        init_model=self.models[fold])                
            else:
                model = lgb.train(params, X_t, num_boost_round = n_boost_round, early_stopping_rounds=early_stopping_rounds,
                                                  valid_sets = [X_t, X_v], verbose_eval=(verbose_eval if self.verbose else None))
                self.models.append(model)
                
            self.oof[v_idx] = self.models[fold].predict(x_valid)
            self.train_preds[tr_idx] += self.models[fold].predict(x_train) / (n_splits-1)
            self.preds += self.models[fold].predict(self.test_df[features]) / n_splits
            self.feature_importances[f'fold_{fold}'] = self.models[fold].feature_importance()
            if self.use_holdback:
                self.holdback_preds += self.models[fold].predict(self.holdback_df[features]) / n_splits
            
            
        tr_score = self.eval_func(self.target, self.train_preds)
        oof_score = self.eval_func(self.target, self.oof)
        print(f'Training {self.params["metric"]}: {tr_score}')
        print(f'OOF {self.params["metric"]}: {oof_score}')
        self.has_trained_models = True
        if self.use_holdback:
            hb_score = self.eval_func(self.holdback_target, self.holdback_preds)
            print(f'Holdback set {self.params["metric"]}: {hb_score}')
            return tr_score, oof_score, hb_score
        return tr_score, oof_score
        
    def fit_clipper(self, verbose=True):
        preds = self.holdback_preds if self.use_holdback else self.oof
        conv_target = np.where(self.holdback_target>0,1,0) if self.use_holdback else np.where(self.target>0,1,0)

        partial_func = partial(minimize_clipper, conv_target, preds)
        opt = minimize(partial_func, x0=[0.08, 0.92], method='nelder-mead')
        if verbose:
            print(f'Clip score: {opt.fun}')
        clips = opt.x
        score = opt.fun
        return clips, score
    
    def fit_spline_model(self, verbose=True):
        preds = self.holdback_preds if self.use_holdback else self.oof
        conv_target = np.where(self.holdback_target>0,1,0) if self.use_holdback else np.where(self.target>0,1,0)
        spline, score = spline_model(conv_target, preds)
        if verbose:
            print(f'Spline score: {score}')

        return spline, score
    
    def postprocess_preds(self, opt_tool, method='clip', use_data='test', return_preds=False):
        pred_dict = {'test':self.preds, 'train':self.train_preds, 'hb':self.holdback_preds, 'oof':self.oof}
        label_dict = {'test':None, 'train':self.target, 'hb':self.holdback_target, 'oof':self.target}       
        
        if method == 'spline':
            adjusted_preds = opt_tool(pred_dict[use_data])
        elif method == 'clip':
            adjusted_preds = np.clip(pred_dict[use_data], opt_tool[0], opt_tool[1])
            
        if use_data == 'test':
            return adjusted_preds
        if return_preds:
            return adjusted_preds, self.eval_func(label_dict[use_data], adjusted_preds)
        return self.eval_func(label_dict[use_data], adjusted_preds)


# In[ ]:


features = [x for x in tourney_result.columns if x not in ['result', 'Score_difference', 'Season']]
params = {'num_leaves': 400,
          'min_child_weight': 0.034,
          'feature_fraction': 0.379,
          'bagging_fraction': 0.418,
          'min_data_in_leaf': 106,
          'max_depth': -1,
          'learning_rate': 0.0068,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          'reg_alpha': 0.3899,
          'reg_lambda': 0.648,
          'random_state': 47,
         }

step_size = 20
steps = 250
boosting_rounds = [step_size*(x+1) for x in range(steps)]
def run_boost_round_test(boosting_rounds, step_size):
    training_scores, oof_scores, holdback_scores = [], [], []
    model = NCAA_model(params, tourney_result, test_df, use_holdback=[2019], regression=False, verbose=False)   
    print(f'Training for {step_size*steps} rounds.')
    for rounds in range(step_size,boosting_rounds+1,step_size):
        print(f'{"*"*50}')
        print(f'Rounds: {rounds}')
        if model.use_holdback:
            tr_score, oof_score, hb_score = model.train(features, n_splits=10, n_boost_round=step_size, early_stopping_rounds=None)
        else:
            tr_score, oof_score = model.train(features, n_splits=10, n_boost_round=step_size, early_stopping_rounds=None)
        clips, clip_s = model.fit_clipper(verbose=True)
        spline, spline_s = model.fit_spline_model(verbose=True)
        
        training_scores.append([tr_score, model.postprocess_preds(clips, use_data = 'train'), 
                               model.postprocess_preds(spline, use_data = 'train', method='spline')])
        oof_scores.append([oof_score, model.postprocess_preds(clips, use_data = 'oof'),
                          model.postprocess_preds(spline, use_data = 'oof', method='spline')])
        holdback_scores.append([hb_score, model.postprocess_preds(clips, use_data = 'hb'),
                               model.postprocess_preds(spline, use_data = 'hb', method='spline')])


    return training_scores, oof_scores, holdback_scores, model, clips, spline

training_scores, oof_scores, holdback_scores, model, clips, spline = run_boost_round_test(boosting_rounds[-1], step_size)


# In[ ]:


training_scores, oof_scores, holdback_scores
fig,ax = plt.subplots(nrows=1,ncols=3, figsize=(20,5), sharey=True, sharex=True)
plot_df = pd.DataFrame(data=training_scores, columns=['Classifier', 'Clipped', 'Spline'], index=boosting_rounds)
plot_df.plot(ax=ax[0], title='Training')
plot_df = pd.DataFrame(data=oof_scores, columns=['Classifier', 'Clipped', 'Spline'], index=boosting_rounds)
plot_df.plot(ax=ax[1], title='Out of Fold')
plot_df = pd.DataFrame(data=holdback_scores, columns=['Classifier', 'Clipped', 'Spline'], index=boosting_rounds)
plot_df.plot(ax=ax[2], title='Holdback')


# In[ ]:


y_preds = model.postprocess_preds(spline, method='spline')
submission_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')
submission_df['Pred'] = y_preds
submission_df.to_csv('submission.csv', index=False)
submission_df.describe()

