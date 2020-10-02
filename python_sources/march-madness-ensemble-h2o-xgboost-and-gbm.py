#!/usr/bin/env python
# coding: utf-8

# # Overview
# #### This kernel is based on last year's [Basic Starter Kernel](https://www.kaggle.com/addisonhoward/basic-starter-kernel-ncaa-men-s-dataset-2019).
# #### Adding XGB and GBM Models

# #### Hi everyone welcome to this kernel,In this kernel i am building XGBoost model and GBM model using H2o framework. <br>
# #### <font color='blue'>If you find this kernel useful please upvote.Also dont forget to upvote the original kernel.</font> <br>

# ## Import Library & Load Data

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV


# In[ ]:


tourney_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')
tourney_seed = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv')


# In[ ]:


# deleting unnecessary columns
tourney_result = tourney_result.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)
tourney_result


# ## Merge Seed

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


# ## Merge Score

# In[ ]:


season_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')


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


# ## Prepare Training Data

# In[ ]:


tourney_win_result['Seed_diff'] = tourney_win_result['Seed1'] - tourney_win_result['Seed2']
tourney_win_result['ScoreT_diff'] = tourney_win_result['ScoreT1'] - tourney_win_result['ScoreT2']
tourney_lose_result['Seed_diff'] = tourney_lose_result['Seed1'] - tourney_lose_result['Seed2']
tourney_lose_result['ScoreT_diff'] = tourney_lose_result['ScoreT1'] - tourney_lose_result['ScoreT2']


# In[ ]:


tourney_win_result['result'] = 1
tourney_lose_result['result'] = 0
tourney_result = pd.concat((tourney_win_result, tourney_lose_result)).reset_index(drop=True)
tourney_result


# # Preparing testing data

# In[ ]:


test_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')


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


# #### Importing h2o framework library which runs on local host

# In[ ]:


import h2o
h2o.init(
  nthreads=-1,            ## -1: use all available threads
  max_mem_size = "8G")  


# In[ ]:


train_list=list(tourney_result.columns)[:-1]


# In[ ]:


train_1=h2o.H2OFrame(tourney_result)
train_1['result']=train_1['result'].asfactor()


# ### XGBoost Model training parameters

# In[ ]:


param = {
      "ntrees" : 2000
    , "max_depth" : 20
    , "learn_rate" : 0.02
    , "sample_rate" : 0.7
    , "col_sample_rate_per_tree" : 0.9
    , "min_rows" : 5
    , "seed": 4241
    , "score_tree_interval": 100,"stopping_metric" :"MSE","nfolds":8,"fold_assignment":"AUTO","keep_cross_validation_predictions" : True,"booster":"dart"
}
from h2o.estimators import H2OXGBoostEstimator
model_xgb = H2OXGBoostEstimator(**param)
model_xgb.train(x = train_list, y = 'result', training_frame = train_1)


# #### Summary Of XGBoost Model

# In[ ]:


model_xgb.summary


# ### GBM Model training parameters

# In[ ]:


param={
    "ntrees" : 1000
    , "max_depth" : 20
    , "learn_rate" : 0.02
    , "sample_rate" : 0.7
    , "col_sample_rate_per_tree" : 0.9
    , "min_rows" : 5
    , "seed": 4241
    , "score_tree_interval": 100,"stopping_metric" :"MSE","nfolds":8,"fold_assignment":"AUTO","keep_cross_validation_predictions" : True

 }
from h2o.estimators.gbm import H2OGradientBoostingEstimator  # import gbm estimator
model_gbm = H2OGradientBoostingEstimator(**param)
model_gbm.train(x = train_list, y = 'result', training_frame = train_1)


# #### GBM Model summary

# In[ ]:


model_gbm.summary


# ### Ensembling the XGBoost and GBM Model using GLM as meta algorithm

# In[ ]:


from h2o.estimators import H2OStackedEnsembleEstimator
stack = H2OStackedEnsembleEstimator(model_id="ensemble11",
                                       training_frame=train_1,
                                       #validation_frame=test,
                                       base_models=[model_xgb.model_id,model_gbm.model_id],metalearner_algorithm="glm")
stack.train(x=train_list, y="result", training_frame=train_1)
#stack.model_performance()


# #### Summary of ensemble model

# In[ ]:


stack.summary


# #### Not using the stacked ensemble model as it is performing relatively bad, so doing manual emsemble below

# In[ ]:


#test_df.head()


# In[ ]:


test_1=h2o.H2OFrame(test_df)


# In[ ]:


pred1=model_xgb.predict(test_1)
pred2=model_gbm.predict(test_1)


# In[ ]:


pred_df1=pred1.as_data_frame()
pred_df2=pred2.as_data_frame()


# In[ ]:


#pred_df


# #### Ensemble of GBM with weight 0.9 and XGB weight 0.1

# In[ ]:


submission_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')
submission_df['Pred'] = pred_df2['p1']*0.9+pred_df1['p1']*0.1
submission_df
submission_df.to_csv('submission.csv', index=False)


# ### To be continued.... will add random forest and deep learning models. Stay Tuned. Please upvote if found useful
