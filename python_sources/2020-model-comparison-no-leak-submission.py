#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import scale

#for dirname, _, filenames in os.walk('/kaggle/input/'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))


# Team power ranking idea and code is due to @raddar https://www.kaggle.com/raddar/team-power-rankings <br>
# KenPom data is from @paulorzp https://www.kaggle.com/paulorzp/kenpom-scraper-2020 <br>
# 

# In[ ]:


input_dir = 'google-cloud-ncaa-march-madness-2020-division-1-mens-tournament'

seeds = pd.read_csv('../input/{}/MDataFiles_Stage1/MNCAATourneySeeds.csv'.format(input_dir))
tourney_results = pd.read_csv('../input/{}/MDataFiles_Stage1/MNCAATourneyCompactResults.csv'.format(input_dir))
regular_results = pd.read_csv('../input/{}/MDataFiles_Stage1/MRegularSeasonCompactResults.csv'.format(input_dir))
regular_results_deets = pd.read_csv('../input/{}/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv'.format(input_dir))
teams = pd.read_csv('../input/{}/MDataFiles_Stage1/MTeams.csv'.format(input_dir))
kenpom = pd.read_csv('/kaggle/input/kenpom-2020/NCAA2020_Kenpom.csv')


# In[ ]:


def prepare_data(df):
    dfswap = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT']]

    dfswap.loc[df['WLoc'] == 'H', 'WLoc'] = 'A'
    dfswap.loc[df['WLoc'] == 'A', 'WLoc'] = 'H'
    df.columns.values[6] = 'location'
    dfswap.columns.values[6] = 'location'         
    df.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(df.columns)]
    dfswap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(dfswap.columns)]
    output = pd.concat([df, dfswap]).sort_index().reset_index(drop=True)
    
    return output


# In[ ]:


tourney_results = prepare_data(tourney_results)
regular_results = prepare_data(regular_results)


# In[ ]:


# convert to str, so the model would treat TeamID them as factors
regular_results['T1_TeamID'] = regular_results['T1_TeamID'].astype(str)
regular_results['T2_TeamID'] = regular_results['T2_TeamID'].astype(str)

# make it a binary task
regular_results['win'] = np.where(regular_results['T1_Score']>regular_results['T2_Score'], 1, 0)

def team_quality(season):
    """
    Calculate team quality for each season seperately. 
    Team strength changes from season to season (students playing change!)
    So pooling everything would be bad approach!
    """
    formula = 'win~-1+T1_TeamID+T2_TeamID'
    glm = sm.GLM.from_formula(formula=formula, 
                              data=regular_results.loc[regular_results.Season==season,:], 
                              family=sm.families.Binomial()).fit()
    
    # extracting parameters from glm
    quality = pd.DataFrame(glm.params).reset_index()
    quality.columns = ['TeamID','beta']
    quality['Season'] = season
    # taking exp due to binomial model being used
    quality['quality'] = np.exp(quality['beta'])
    # only interested in glm parameters with T1_, as T2_ should be mirroring T1_ ones
    quality = quality.loc[quality.TeamID.str.contains('T1_')].reset_index(drop=True)
    quality['TeamID'] = quality['TeamID'].apply(lambda x: x[10:14]).astype(int)
    return quality


# This is the team_quality feature, which is essentially a glm fit on the number of wins. Again, idea and code is from @raddar. This takes a while to run.

# In[ ]:


import time
start = time.time()
team_qual = pd.concat([team_quality(2010),
                       team_quality(2011),
                       team_quality(2012),
                       team_quality(2013),
                       team_quality(2014),
                       team_quality(2015),
                       team_quality(2016),
                       team_quality(2017),
                       team_quality(2018),
                       team_quality(2019)]).reset_index(drop=True)
end = time.time()
print("time elapsed:",end - start)


# In[ ]:


team_quality_T1 = team_qual[['TeamID','Season','quality']]
team_quality_T1.columns = ['T1_TeamID','Season','T1_quality']
team_quality_T2 = team_qual[['TeamID','Season','quality']]
team_quality_T2.columns = ['T2_TeamID','Season','T2_quality']

tourney_results['T1_TeamID'] = tourney_results['T1_TeamID'].astype(int)
tourney_results['T2_TeamID'] = tourney_results['T2_TeamID'].astype(int)
tourney_results = tourney_results.merge(team_quality_T1, on = ['T1_TeamID','Season'], how = 'left')
tourney_results = tourney_results.merge(team_quality_T2, on = ['T2_TeamID','Season'], how = 'left')


# In[ ]:


# we only have tourney results since year 2010
tourney_results = tourney_results.loc[tourney_results['Season'] >= 2010].reset_index(drop=True)

# not interested in pre-selection matches
tourney_results = tourney_results.loc[tourney_results['DayNum'] >= 136].reset_index(drop=True)


# In[ ]:


seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
seeds['division'] = seeds['Seed'].apply(lambda x: x[0])

seeds_T1 = seeds[['Season','TeamID','seed','division']].copy()
seeds_T2 = seeds[['Season','TeamID','seed','division']].copy()
seeds_T1.columns = ['Season','T1_TeamID','T1_seed','T1_division']
seeds_T2.columns = ['Season','T2_TeamID','T2_seed','T2_division']

tourney_results = tourney_results.merge(seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_results = tourney_results.merge(seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')


# Here we convert quality to powerrank, by grouping quality by division so we get a number similar to seed.

# In[ ]:


tourney_results['T1_powerrank'] = tourney_results.groupby(['Season','T1_division'])['T1_quality'].rank(method='dense', ascending=False).astype(int)
tourney_results['T2_powerrank'] = tourney_results.groupby(['Season','T2_division'])['T2_quality'].rank(method='dense', ascending=False).astype(int)


# This is where the KenPom data gets applied

# In[ ]:


kpcols = list(kenpom.columns)
a, b = kpcols.index('Season'), kpcols.index('TeamName')
kpcols[b], kpcols[a] = kpcols[a], kpcols[b]
kenpom = kenpom[kpcols]

kenpom_T1 = kenpom.copy()
kenpom_T2 = kenpom.copy()

kpT1cols = []; kpT2cols = [];
kpT1cols.append('Season');kpT2cols.append('Season')
for k in kenpom.columns:
    if k!='Season':
        kpT1cols.append('T1_{}'.format(k))
        kpT2cols.append('T2_{}'.format(k))
    
kenpom_T1.columns = kpT1cols
kenpom_T2.columns = kpT2cols

tourney_results = tourney_results.merge(kenpom_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_results = tourney_results.merge(kenpom_T2, on = ['Season', 'T2_TeamID'], how = 'left')


# # Regular season stats

# In[ ]:


tourney_results.head()


# In[ ]:


def prepare_data_deets(df):
    dfswap = df.copy()

    dfswap.loc[df['WLoc'] == 'H', 'WLoc'] = 'A'
    dfswap.loc[df['WLoc'] == 'A', 'WLoc'] = 'H'
    df.columns.values[6] = 'location'
    dfswap.columns.values[6] = 'location'         
    df.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(df.columns)]
    dfswap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(dfswap.columns)]
    output = pd.concat([df, dfswap]).sort_index().reset_index(drop=True)
    
    return output


# In[ ]:


regular_results_deets = prepare_data_deets(regular_results_deets)


# In[ ]:


regular_results_deets['T1_FGeff'] = regular_results_deets['T1_FGM']/regular_results_deets['T1_FGA']
regular_results_deets['T2_FGeff'] = regular_results_deets['T2_FGM']/regular_results_deets['T2_FGA']

regular_results_deets['T1_FG3eff'] = regular_results_deets['T1_FGM3']/regular_results_deets['T1_FGA3']
regular_results_deets['T2_FG3eff'] = regular_results_deets['T2_FGM3']/regular_results_deets['T2_FGA3']


# In[ ]:


T1_FGeffM_S = regular_results_deets.groupby(['T1_TeamID','Season']).agg({'T1_FGeff':['mean','std']})
T2_FGeffM_S = regular_results_deets.groupby(['T2_TeamID','Season']).agg({'T2_FGeff':['mean','std']})

T1_FG3effM_S = regular_results_deets.groupby(['T1_TeamID','Season']).agg({'T1_FG3eff':['mean','std']})
T2_FG3effM_S = regular_results_deets.groupby(['T2_TeamID','Season']).agg({'T2_FG3eff':['mean','std']})


# In[ ]:


tourney_results = tourney_results.merge(T1_FGeffM_S, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_results = tourney_results.merge(T2_FGeffM_S, on = ['Season', 'T2_TeamID'], how = 'left')
tourney_results = tourney_results.merge(T1_FG3effM_S, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_results = tourney_results.merge(T2_FG3effM_S, on = ['Season', 'T2_TeamID'], how = 'left')


# In[ ]:


tourney_results['win'] = np.where(tourney_results['T1_Score'] > tourney_results['T2_Score'], 1, 0)


# Dummy variables for the conference

# In[ ]:


T1_conf_dum = pd.get_dummies(tourney_results['T1_conference'])
T2_conf_dum = pd.get_dummies(tourney_results['T2_conference'])

T1_conf_dum.columns = ['T1_{}'.format(i) for i in T1_conf_dum.columns]
T2_conf_dum.columns = ['T2_{}'.format(i) for i in T2_conf_dum.columns]


# # Model

# In[ ]:


feats_to_use = ['Season','T1_powerrank','T2_powerrank',                'T1_seed','T2_seed','T1_rank','T2_rank',                ('T1_FGeff', 'mean'),('T2_FGeff', 'mean'),                ('T1_FGeff', 'std'),('T2_FGeff', 'std'),                ('T1_FG3eff', 'mean'),('T2_FG3eff', 'mean'),                ('T1_FG3eff', 'std'),('T2_FG3eff', 'std')]

model_df = tourney_results[feats_to_use+['win']]

model_df = pd.concat([model_df.loc[:, model_df.columns != 'win'],T1_conf_dum,T2_conf_dum,model_df['win']],axis=1)


# Scaling the continuous variables helps the ML algorithms converge.

# In[ ]:


model_df[['T1_powerrank','T2_powerrank','T1_seed','T2_seed','T1_rank','T2_rank']] =         scale(model_df[['T1_powerrank','T2_powerrank','T1_seed','T2_seed','T1_rank','T2_rank']])


# In[ ]:


from tqdm import tqdm_notebook

scores_lr = []
scores_rf = []
scores_nn = []
scores_ens = []

for i in tqdm_notebook(np.arange(2010,2020)):
    X_train = model_df[model_df.Season!=i].iloc[:,1:-1]
    y_train = model_df[model_df.Season!=i].iloc[:,-1]
    X_test = model_df[model_df.Season==i].iloc[:,1:-1]
    y_test = model_df[model_df.Season==i].iloc[:,-1]
    
    lr = LogisticRegression(random_state=4351)
    lr.fit(X_train,y_train)

    rf = RandomForestClassifier(n_estimators=1000,random_state=342)
    rf.fit(X_train,y_train)
    
    nn = MLPClassifier(hidden_layer_sizes=(5,7,),random_state=222)
    nn.fit(X_train,y_train)
    
    lr_yhat_prob = lr.predict_proba(X_test)[:,1]
    lr_yhat = lr.predict(X_test)

    rf_yhat_prob = rf.predict_proba(X_test)[:,1]
    rf_yhat = rf.predict(X_test)
    
    nn_yhat_prob = nn.predict_proba(X_test)[:,1]
    nn_yhat = nn.predict(X_test)
    
    ens_yhat_prob = 0.33*lr_yhat_prob + 0.33*rf_yhat_prob + 0.33*nn_yhat_prob
    
    scores_lr.append(log_loss(y_test.values,lr_yhat_prob))
    scores_rf.append(log_loss(y_test.values,rf_yhat_prob))
    scores_nn.append(log_loss(y_test.values,nn_yhat_prob))
    scores_ens.append(log_loss(y_test.values,ens_yhat_prob))


# We can examine the logistic regression coefficients for clues on relationships. A positive value means it correlated positively with a win for team 1, a negative value means that variable correlated negatively with a win for team 1.

# In[ ]:


#plt.figure(figsize=(10,15))
#y_pos = np.arange(len(X_train.columns))
#plt.barh(y_pos, lr.coef_[0])
 
# Create names on the y-axis
#plt.yticks(y_pos, X_train.columns)
#plt.show()


# We can also show feature importances from the Random Forest model.

# In[ ]:


#plt.figure(figsize=(10,15))
#y_pos = np.arange(len(X_train.columns))
#plt.barh(y_pos, rf.feature_importances_)
 
# Create names on the y-axis
#plt.yticks(y_pos, X_train.columns)
#plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
plt.plot(np.arange(2010,2020),scores_lr,'b.')
plt.plot(np.arange(2010,2020),scores_rf,'r.')
plt.plot(np.arange(2010,2020),scores_nn,'g.')
plt.plot(np.arange(2010,2020),scores_ens,'m.')


# In[ ]:


print(np.mean(scores_lr),np.std(scores_lr))
print(np.mean(scores_rf),np.std(scores_rf))
print(np.mean(scores_nn),np.std(scores_nn))
print(np.mean(scores_ens),np.std(scores_ens))


# As you can see, depending on which year you test, a different model performs better.

# # Submission Stage 1

# @catadanna made this function to remove test samples from the training set
# https://www.kaggle.com/catadanna/delete-leaked-from-training-ncaam-ncaaw-stage1

# In[ ]:


def concat_row(r):
    if r['WTeamID'] < r['LTeamID']:
        res = str(r['Season'])+"_"+str(r['WTeamID'])+"_"+str(r['LTeamID'])
    else:
        res = str(r['Season'])+"_"+str(r['LTeamID'])+"_"+str(r['WTeamID'])
    return res

def delete_leaked_from_df_train(df_train, df_test):
    df_train['Concats'] = df_train.apply(concat_row, axis=1)
    df_train_duplicates = df_train[df_train['Concats'].isin(df_test['ID'].unique())]
    df_train_idx = df_train_duplicates.index.values
    df_train = df_train.drop(df_train_idx)
    df_train = df_train.drop('Concats', axis=1)
    
    return df_train 


# In[ ]:


feats_to_use = ['Season','seed_diff','rank_diff','powerrank_diff']


# In[ ]:


train_df = pd.read_csv('../input/{}/MDataFiles_Stage1/MNCAATourneyCompactResults.csv'.format(input_dir))
test_df = pd.read_csv('../input/{}/MSampleSubmissionStage1_2020.csv'.format(input_dir))


# In[ ]:


train_df = delete_leaked_from_df_train(train_df, test_df)


# In[ ]:


train_df = prepare_data(train_df)

train_df['T1_TeamID'] = train_df['T1_TeamID'].astype(int)
train_df['T2_TeamID'] = train_df['T2_TeamID'].astype(int)
train_df = train_df.merge(team_quality_T1, on = ['T1_TeamID','Season'], how = 'left')
train_df = train_df.merge(team_quality_T2, on = ['T2_TeamID','Season'], how = 'left')

# we only have tourney results since year 2010
train_df = train_df.loc[train_df['Season'] >= 2010].reset_index(drop=True)

# not interested in pre-selection matches
train_df = train_df.loc[train_df['DayNum'] >= 136].reset_index(drop=True)

train_df = train_df.merge(seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
train_df = train_df.merge(seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')

train_df['seed_diff'] = train_df['T1_seed']-train_df['T2_seed']

train_df['T1_powerrank'] = train_df.groupby(['Season','T1_division'])['T1_quality'].rank(method='dense', ascending=False).astype(int)
train_df['T2_powerrank'] = train_df.groupby(['Season','T2_division'])['T2_quality'].rank(method='dense', ascending=False).astype(int)

train_df = train_df.merge(kenpom_T1, on = ['Season', 'T1_TeamID'], how = 'left')
train_df = train_df.merge(kenpom_T2, on = ['Season', 'T2_TeamID'], how = 'left')

train_df['rank_diff'] = train_df['T1_rank']-train_df['T2_rank']
train_df['powerrank_diff'] = train_df['T1_powerrank']-train_df['T2_powerrank']

train_df = train_df.merge(T1_FGeffM_S, on = ['Season', 'T1_TeamID'], how = 'left')
train_df = train_df.merge(T2_FGeffM_S, on = ['Season', 'T2_TeamID'], how = 'left')
train_df = train_df.merge(T1_FG3effM_S, on = ['Season', 'T1_TeamID'], how = 'left')
train_df = train_df.merge(T2_FG3effM_S, on = ['Season', 'T2_TeamID'], how = 'left')

train_df['win'] = np.where(train_df['T1_Score'] > train_df['T2_Score'], 1, 0)

T1_conf_dum = pd.get_dummies(train_df['T1_conference'])
T2_conf_dum = pd.get_dummies(train_df['T2_conference'])

T1_conf_dum.columns = ['T1_{}'.format(i) for i in T1_conf_dum.columns]
T2_conf_dum.columns = ['T2_{}'.format(i) for i in T2_conf_dum.columns]

train_df[['T1_powerrank','T2_powerrank','T1_seed','T2_seed','T1_rank','T2_rank','seed_diff','rank_diff','powerrank_diff']] =             scale(train_df[['T1_powerrank','T2_powerrank','T1_seed','T2_seed','T1_rank','T2_rank','seed_diff','rank_diff','powerrank_diff']])

model_df = pd.concat([model_df.loc[:, model_df.columns != 'win'],T1_conf_dum,T2_conf_dum,model_df['win']],axis=1)

model_df = train_df[feats_to_use+['win']]


# In[ ]:


model_df.head()


# Train

# In[ ]:


X_train = model_df.iloc[:,1:-1]
y_train = model_df.iloc[:,-1]

lr.fit(X_train,y_train)
rf.fit(X_train,y_train)
nn.fit(X_train,y_train)


# Test

# In[ ]:


test_df['Season'] = test_df['ID'].map(lambda x: int(x[:4]))
test_df['T1_TeamID'] = test_df['ID'].map(lambda x: int(x[5:9]))
test_df['T2_TeamID'] = test_df['ID'].map(lambda x: int(x[10:14]))


# In[ ]:


test_dfT1 = test_df.merge(team_quality_T1, on = ['Season','T1_TeamID'], how = 'left')
test_dfT1 = test_dfT1.merge(seeds_T1, on = ['Season','T1_TeamID'], how = 'left')
test_dfT2 = test_df.merge(team_quality_T2, on = ['Season','T2_TeamID'], how = 'left')
test_dfT2 = test_dfT2.merge(seeds_T2, on = ['Season','T2_TeamID'], how = 'left')


# In[ ]:


test_dfT1['T1_powerrank'] = test_dfT1.groupby(['Season','T1_division'])['T1_quality'].rank(method='dense', ascending=False).astype(int)
test_dfT2['T2_powerrank'] = test_dfT2.groupby(['Season','T2_division'])['T2_quality'].rank(method='dense', ascending=False).astype(int)


# In[ ]:


kenpom_T1['Season'] = kenpom_T1['Season'].astype('int64')
kenpom_T2['Season'] = kenpom_T2['Season'].astype('int64')

test_dfT1 = test_dfT1.merge(kenpom_T1, on = ['Season', 'T1_TeamID'], how = 'left')
test_dfT2 = test_dfT2.merge(kenpom_T2, on = ['Season', 'T2_TeamID'], how = 'left')


# In[ ]:


test_df = pd.concat([test_dfT1[['Season','T1_TeamID','T1_powerrank','T1_seed','T1_rank']],test_dfT2[['T2_TeamID','T2_powerrank','T2_seed','T2_rank']]],axis=1)


# In[ ]:


test_dfFGeffM_ST1 = test_df.merge(T1_FGeffM_S, on = ['Season', 'T1_TeamID'], how = 'left')
test_dfFGeffM_ST2 = test_df.merge(T2_FGeffM_S, on = ['Season', 'T2_TeamID'], how = 'left')


# In[ ]:


test_dfFGeffM_ST1 = test_dfFGeffM_ST1.merge(T1_FG3effM_S, on = ['Season', 'T1_TeamID'], how = 'left')
test_dfFGeffM_ST2 = test_dfFGeffM_ST2.merge(T2_FG3effM_S, on = ['Season', 'T2_TeamID'], how = 'left')


# In[ ]:


test_df = pd.concat([test_dfFGeffM_ST1[['Season','T1_TeamID','T1_powerrank',                                        'T1_seed','T1_rank',('T1_FGeff','mean'),('T1_FGeff','std'),                                        ('T1_FG3eff','mean'),('T1_FG3eff','std')]],                     test_dfFGeffM_ST2[['T2_TeamID','T2_powerrank',                                        'T2_seed','T2_rank',('T2_FGeff','mean'),('T2_FGeff','std'),                                        ('T2_FG3eff','mean'),('T2_FG3eff','std')]]],axis=1)


# In[ ]:


T1_conf_dum = pd.get_dummies(test_dfT1['T1_conference'])
T2_conf_dum = pd.get_dummies(test_dfT2['T2_conference'])

T1_conf_dum.columns = ['T1_{}'.format(i) for i in T1_conf_dum.columns]
T2_conf_dum.columns = ['T2_{}'.format(i) for i in T2_conf_dum.columns]

# No Pac 10 in this data
T1_conf_dum.insert(loc=22,column='T1_P10',value = 0)
T2_conf_dum.insert(loc=22,column='T2_P10',value = 0)


# In[ ]:


test_df = pd.concat([test_df['Season'],test_df[['T1_powerrank','T2_powerrank',                                                'T1_seed','T2_seed',                                                'T1_rank','T2_rank',                                               ('T1_FGeff', 'mean'),('T1_FGeff', 'std'),                                               ('T2_FGeff', 'mean'),('T2_FGeff', 'std'),                                               ('T1_FG3eff', 'mean'),('T1_FG3eff', 'std'),                                               ('T2_FG3eff', 'mean'),('T2_FG3eff', 'std')]],                     T1_conf_dum,T2_conf_dum],axis=1)


# In[ ]:


test_df['seed_diff'] = test_df['T1_seed']-test_df['T2_seed']
test_df['rank_diff'] = test_df['T1_rank']-test_df['T2_rank']
test_df['powerrank_diff'] = test_df['T1_powerrank']-test_df['T2_powerrank']

test_df[['T1_powerrank','T2_powerrank','T1_seed','T2_seed','T1_rank','T2_rank','seed_diff','rank_diff','powerrank_diff']] =     scale(test_df[['T1_powerrank','T2_powerrank','T1_seed','T2_seed','T1_rank','T2_rank','seed_diff','rank_diff','powerrank_diff']])


# In[ ]:


test_df = test_df[feats_to_use]


# In[ ]:


model_df.head()


# In[ ]:


test_df.head()


# In[ ]:


yhat = 0*nn.predict_proba(test_df.drop('Season', axis=1))+        0*rf.predict_proba(test_df.drop('Season', axis=1))+        1.0*lr.predict_proba(test_df.drop('Season', axis=1))


# In[ ]:


borderlineY = test_df[(yhat[:,1]>0.5) & (yhat[:,1]<0.55)]
borderlineY['V1'] = np.where(borderlineY['seed_diff']>0,1,0)
borderlineY['V2'] = np.where(borderlineY['rank_diff']>0,1,0)
borderlineY['V3'] = np.where(borderlineY['powerrank_diff']>0,1,0)

borderlineY['votes'] = borderlineY['V1']+borderlineY['V2']+borderlineY['V3']

borderlineN = test_df[(yhat[:,1]>0.5) & (yhat[:,1]<0.55)]
borderlineN['V1'] = np.where(borderlineN['seed_diff']<0,1,0)
borderlineN['V2'] = np.where(borderlineN['rank_diff']<0,1,0)
borderlineN['V3'] = np.where(borderlineN['powerrank_diff']<0,1,0)

borderlineN['votes'] = borderlineN['V1']+borderlineN['V2']+borderlineN['V3']


# In[ ]:


plt.hist(yhat[:,1])


# In[ ]:


yhat[borderlineY[borderlineY.votes==3].index,1]+=0.4
yhat[borderlineY[borderlineY.votes==2].index,1]+=0.3
yhat[borderlineY[borderlineY.votes==1].index,1]+=0.1

yhat[borderlineN[borderlineN.votes==3].index,1]-=0.4
yhat[borderlineN[borderlineN.votes==2].index,1]-=0.3
yhat[borderlineN[borderlineN.votes==1].index,1]-=0.1


# In[ ]:


plt.hist(yhat[:,1])


# In[ ]:


submit = pd.read_csv('../input/{}/MSampleSubmissionStage1_2020.csv'.format(input_dir))


# In[ ]:


submit['Pred'] = yhat[:,1]


# In[ ]:


submit.to_csv('SampleSubmissionStage1_Latimer.csv',index=False)


# Version 3 LB - 0.5917 <br>
# Version 4 LB - 0.51550 (lr) <br>
# Version 7 LB - 0.50762 (ensemble, equal weights) <br>
# Version 12 LB (fixed leaks) - 0.54538 (ens, equal weights), 0.63224 (nn), 0.55683 (rf), 0.55865 (lr) <br>
# Baseline with no team stat features: lr - 0.54262, rf - 0.51148, nn - 0.53148, ens - 0.51415 <br>
# Baseline with only the seed diff: lr - 0.55109 <br>
# Baseline with seed_diff and rank_diff: lr - 0.53932 <br>
# Baseline with seed_diff, rank_diff, and powerrank_diff: lr = 0.53929

# In[ ]:




