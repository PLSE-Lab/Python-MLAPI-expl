#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# Good Luck to everyone especially the ones who want to bet against Zion and his Blue devils.
# 
# ![ZionUrl](https://media.giphy.com/media/54ZSkrYYe8lk9td6r1/giphy.gif "zion")
# 
# In this kernel I would like to explore the NCAA data with a quick look at some classical and advanced stats in order to understand which statistics may be more useful for predicting the Ws & Ls in the tournament.
# 
# ICYMI: https://stats.nba.com/help/glossary/
# 
# Having a detailed information set about every single games of the different seasons, the first step is to create a dataset which contains aggregated stats for each single team.

# In[ ]:


## > LIBRARIES
import os
import re
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

## > DATA
path_datasets = '../input/datafiles/'

df_rs_c_res = pd.read_csv(path_datasets + 'RegularSeasonCompactResults.csv')
df_rs_d_res = pd.read_csv(path_datasets + 'RegularSeasonDetailedResults.csv')
df_teams = pd.read_csv(path_datasets + 'Teams.csv')
df_seeds = pd.read_csv(path_datasets + 'NCAATourneySeeds.csv')
coaches = pd.read_csv(path_datasets + 'TeamCoaches.csv')
df_tourn = pd.read_csv(path_datasets + 'NCAATourneyCompactResults.csv')


# ## Data Cleaning
# 
# I basically want to create a unique dataset which uses TeamID as a key. Then aggregate on this particular key every possible metric which seems to be interesting (here just a few).  
# Therefore I will end up having regular season statistics which may be helpful in order to predict tournament outcomes.

# In[ ]:


## > DATA CLEANING
# clean team information

df_teams_cl = df_teams.iloc[:,:2]

## > DATA CLEANING
# clean seed information

df_seeds_cl = df_seeds.loc[:, ['TeamID', 'Season', 'Seed']]

def clean_seed(seed):
    s_int = int(seed[1:3])
    return s_int

def extract_seed_region(seed):
    s_reg = seed[0:1]
    return s_reg

df_seeds_cl['seed_int'] = df_seeds_cl['Seed'].apply(lambda x: clean_seed(x))
df_seeds_cl['seed_region'] = df_seeds_cl['Seed'].apply(lambda x: extract_seed_region(x))
df_seeds_cl['top_seeded_teams'] = np.where(df_seeds_cl['Seed'].isnull(), 0, 1)

df_seeds_cl.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label
df_seeds_cl.head()


# In[ ]:


## > DATA CLEANING
# create games dataframe WINNERS

def new_name_w_1(old_name):
    match = re.match(r'^L', old_name)
    if match:
        out = re.sub('^L','', old_name)
        return out + '_opp'
    return old_name

def new_name_w_2(old_name):
    match = re.match(r'^W', old_name)
    if match:
        out = re.sub('^W','', old_name)
        return out
    return old_name

def prepare_stats_extended_winners(df_in, df_seed_in, df_teams_in):
    df_in['poss'] = df_in['WFGA'] + 0.475*df_in['WFTA'] - df_in['WOR'] + df_in['WTO']
    df_in['opp_poss'] = df_in['LFGA'] + 0.475*df_in['LFTA'] - df_in['LOR'] + df_in['LTO']
    df_in['off_rating'] = 100*(df_in['WScore'] / df_in['poss'])
    df_in['def_rating'] = 100*(df_in['LScore'] / df_in['opp_poss'])
    df_in['net_rating'] = df_in['off_rating'] - df_in['def_rating']
    df_in['pace'] = 48*((df_in['poss']+df_in['opp_poss'])/(2*(240/5)))
    
    df_in = df_in.rename(columns={'WTeamID':'TeamID', 
                                  'WLoc':'_Loc',
                                  'LTeamID':'TeamID_opp',
                                  'WScore':'Score_left', 
                                  'LScore':'Score_right'})
    
    df_seeds_opp = df_seed_in.rename(columns={'TeamID':'TeamID_opp',
                                              'seed_int':'seed_int_opp',
                                              'seed_region':'seed_region_opp',
                                              'top_seeded_teams':'top_seeded_teams_opp'})
    
    df_out = pd.merge(left=df_in, right=df_seeds_cl, how='left', on=['Season', 'TeamID'])
    df_out = pd.merge(left=df_out, right=df_seeds_opp, how='left', on=['Season', 'TeamID_opp'])
    df_out = pd.merge(left=df_out, right=df_teams_in, how='left', on=['TeamID'])
    
    df_out['DayNum'] = pd.to_numeric(df_out['DayNum'])
    df_out['win_dummy'] = 1
    
    df_out['seed_int'] = np.where(df_out['seed_int'].isnull(), 20, df_out['seed_int'])
    df_out['seed_region'] = np.where(df_out['seed_region'].isnull(), 'NoTour', df_out['seed_region'])
    df_out['top_seeded_teams'] = np.where(df_out['top_seeded_teams'].isnull(), 0, df_out['top_seeded_teams'])
    
    df_out['seed_int_opp'] = np.where(df_out['seed_int_opp'].isnull(), 20, df_out['seed_int_opp'])
    df_out['seed_region_opp'] = np.where(df_out['seed_region_opp'].isnull(), 'NoTour', df_out['seed_region_opp'])
    df_out['top_seeded_teams_opp'] = np.where(df_out['top_seeded_teams_opp'].isnull(), 0, df_out['top_seeded_teams_opp'])
    
    df_out = df_out.rename(columns=new_name_w_1)
    df_out = df_out.rename(columns=new_name_w_2)
    
    return df_out

df_games_w = prepare_stats_extended_winners(df_rs_d_res, df_seeds_cl, df_teams_cl)

df_games_w.head()


# In[ ]:


## > DATA CLEANING
# create games dataframe LOSERS

def new_name_l_1(old_name):
    match = re.match(r'^W', old_name)
    if match:
        out = re.sub('^W','', old_name)
        return out + '_opp'
    return old_name

def new_name_l_2(old_name):
    match = re.match(r'^L', old_name)
    if match:
        out = re.sub('^L','', old_name)
        return out
    return old_name

def prepare_stats_extended_losers(df_in, df_seed_in, df_teams_in):
    df_in['poss'] = df_in['LFGA'] + (0.475*df_in['LFTA']) - df_in['LOR'] + df_in['LTO']
    df_in['opp_poss'] = df_in['WFGA'] + (0.475*df_in['WFTA']) - df_in['WOR'] + df_in['WTO']
    df_in['off_rating'] = 100*(df_in['LScore'] / df_in['poss'])
    df_in['def_rating'] = 100*(df_in['WScore'] / df_in['opp_poss'])
    df_in['net_rating'] = df_in['off_rating'] - df_in['def_rating']
    df_in['pace'] = 48*((df_in['poss']+df_in['opp_poss'])/(2*(240/5)))
    
    df_in = df_in.rename(columns={'LTeamID':'TeamID', 
                                  'LLoc':'_Loc',
                                  'WTeamID':'TeamID_opp',
                                  'LScore':'Score_left', 
                                  'WScore':'Score_right'})
    
    df_seeds_opp = df_seed_in.rename(columns={'TeamID':'TeamID_opp',
                                              'seed_int':'seed_int_opp',
                                              'seed_region':'seed_region_opp',
                                              'top_seeded_teams':'top_seeded_teams_opp'})
    
    df_out = pd.merge(left=df_in, right=df_seeds_cl, how='left', on=['Season', 'TeamID'])
    df_out = pd.merge(left=df_out, right=df_seeds_opp, how='left', on=['Season', 'TeamID_opp'])
    df_out = pd.merge(left=df_out, right=df_teams_in, how='left', on=['TeamID'])
    
    df_out['DayNum'] = pd.to_numeric(df_out['DayNum'])
    df_out['win_dummy'] = 0
    
    df_out['seed_int'] = np.where(df_out['seed_int'].isnull(), 20, df_out['seed_int'])
    df_out['seed_region'] = np.where(df_out['seed_region'].isnull(), 'NoTour', df_out['seed_region'])
    df_out['top_seeded_teams'] = np.where(df_out['top_seeded_teams'].isnull(), 0, df_out['top_seeded_teams'])
    
    df_out['seed_int_opp'] = np.where(df_out['seed_int_opp'].isnull(), 20, df_out['seed_int_opp'])
    df_out['seed_region_opp'] = np.where(df_out['seed_region_opp'].isnull(), 'NoTour', df_out['seed_region_opp'])
    df_out['top_seeded_teams_opp'] = np.where(df_out['top_seeded_teams_opp'].isnull(), 0, df_out['top_seeded_teams_opp'])

    df_out = df_out.rename(columns=new_name_l_1)
    df_out = df_out.rename(columns=new_name_l_2)
    
    return df_out

df_games_l = prepare_stats_extended_losers(df_rs_d_res, df_seeds_cl, df_teams_cl)

df_games_l.head()


# In[ ]:


## > MERGE

df_games_t = pd.concat([df_games_w,df_games_l], sort=True)

## > AGGREGATED STATS BY TEAM AND SEASON

def aggr_stats(df):
    d = {}
    d['G'] = df['win_dummy'].count()
    d['W'] = df['win_dummy'].sum()
    d['L'] = np.sum(df['win_dummy'] == 0)
    d['G_vs_topseeds'] = np.sum(df['top_seeded_teams_opp'] == 1)
    d['W_vs_topseeds'] = np.sum((df['win_dummy'] == 1) & (df['top_seeded_teams_opp'] == 1))
    d['L_vs_topseeds'] = np.sum((df['win_dummy'] == 0) & (df['top_seeded_teams_opp'] == 1))
    d['G_last30D'] = np.sum((df['DayNum'] > 100))
    d['W_last30D'] = np.sum((df['win_dummy'] == 1) & (df['DayNum'] > 100))
    d['L_last30D'] = np.sum((df['win_dummy'] == 0) & (df['DayNum'] > 100))
    d['G_H'] = np.sum((df['_Loc'] == 'H'))
    d['W_H'] = np.sum((df['win_dummy'] == 1) & (df['_Loc'] == 'H'))
    d['L_H'] = np.sum((df['win_dummy'] == 0) & (df['_Loc'] == 'H'))
    d['G_A'] = np.sum((df['_Loc'] == 'A'))
    d['W_A'] = np.sum((df['win_dummy'] == 1) & (df['_Loc'] == 'A'))
    d['L_A'] = np.sum((df['win_dummy'] == 0) & (df['_Loc'] == 'A'))
    d['G_N'] = np.sum((df['_Loc'] == 'N'))
    d['W_N'] = np.sum((df['win_dummy'] == 1) & (df['_Loc'] == 'N'))
    d['L_N'] = np.sum((df['win_dummy'] == 0) & (df['_Loc'] == 'N'))
    
    d['PS'] = np.mean(df['Score_left'])
    d['PS_H'] = np.mean(df['Score_left'][df['_Loc'] == 'H'])
    d['PS_A'] = np.mean(df['Score_left'][df['_Loc'] == 'A'])
    d['PS_N'] = np.mean(df['Score_left'][df['_Loc'] == 'N'])
    d['PS_last30D'] = np.mean(df['Score_left'][df['DayNum'] > 100])
    
    d['PA'] = np.mean(df['Score_right'])
    d['PA_H'] = np.mean(df['Score_right'][df['_Loc'] == 'H'])
    d['PA_A'] = np.mean(df['Score_right'][df['_Loc'] == 'A'])
    d['PA_N'] = np.mean(df['Score_right'][df['_Loc'] == 'N'])
    d['PA_last30D'] = np.mean(df['Score_right'][df['DayNum'] > 100])
    
    d['poss_m'] = np.mean(df['poss'])
    d['opp_poss_m'] = np.mean(df['opp_poss'])
    d['off_rating_m'] = np.mean(df['off_rating'])
    d['def_rating_m'] = np.mean(df['def_rating'])
    d['net_rating_m'] = np.mean(df['net_rating'])
    d['pace_m'] = np.mean(df['pace'])
    
    d['off_rating_m_last30D'] = np.mean(df['off_rating'][df['DayNum'] > 100])
    d['def_rating_m_last30D'] = np.mean(df['def_rating'][df['DayNum'] > 100])
    d['net_rating_m_last30D'] = np.mean(df['net_rating'][df['DayNum'] > 100])
    
    d['off_rating_m_vs_topseeds'] = np.mean(df['off_rating'][df['top_seeded_teams_opp'] == 1])
    d['def_rating_m_vs_topseeds'] = np.mean(df['def_rating'][df['top_seeded_teams_opp'] == 1])
    d['net_rating_m_vs_topseeds'] = np.mean(df['net_rating'][df['top_seeded_teams_opp'] == 1])
    
    return pd.Series(d)


df_agg_stats = df_games_t.                          groupby([df_games_t['Season'], 
                                   df_games_t['TeamID'],
                                   df_games_t['TeamName'],
                                   df_games_t['seed_int'],
                                   df_games_t['seed_region']], 
                                  as_index=False).\
                          apply(aggr_stats).\
                          reset_index()


df_agg_stats['w_pct'] = df_agg_stats['W'] / df_agg_stats['G']
df_agg_stats['w_pct_last30D'] = df_agg_stats['W_last30D'] / df_agg_stats['G_last30D']
df_agg_stats['w_pct_vs_topseeds'] = df_agg_stats['W_vs_topseeds'] / df_agg_stats['G_vs_topseeds']

df_agg_stats.head(20)


# In[ ]:


## > DATA CLEANING 

# prepare tournament dataset
def prepare_tournament_datasets(df_tourn_in, df_agg_stats_in):
    
    df_tourn_in['TeamID'] = df_tourn_in[['WTeamID','LTeamID']].min(axis=1)
    df_tourn_in['TeamID_opp'] = df_tourn_in[['WTeamID','LTeamID']].max(axis=1)
    df_tourn_in['win_dummy'] = np.where(df_tourn_in['TeamID'] == df_tourn_in['WTeamID'], 1, 0)
    df_tourn_in['delta'] = np.where(df_tourn_in['win_dummy'] == 1,
                                    df_tourn_in['WScore'] - df_tourn['LScore'],
                                    df_tourn_in['LScore'] - df_tourn['WScore'])
    df_tourn_in['Score_left'] = np.where(df_tourn_in['win_dummy'] == 1,
                                         df_tourn_in['WScore'],
                                         df_tourn_in['LScore'])
    df_tourn_in['Score_right'] = np.where(df_tourn_in['win_dummy'] == 1,
                                          df_tourn_in['LScore'],
                                          df_tourn_in['WScore'])
                                 
    df_teams_gr_left = df_agg_stats_in.loc[:,['Season', 'TeamID',
                                              'w_pct', 'seed_int', 
                                              'net_rating_m_last30D',
                                              'net_rating_m_vs_topseeds',
                                              'net_rating_m']].\
                  rename(columns={'w_pct':'w_pct_left',
                                  'seed_int':'seed_int_left', 
                                  'net_rating_m_last30D':'net_rating_m_last30D_left', 
                                  'net_rating_m_vs_topseeds':'net_rating_m_vs_topseeds_left', 
                                  'net_rating_m':'net_rating_m_left'})
    
    df_teams_gr_right = df_agg_stats_in.loc[:,['Season', 'TeamID',
                                               'w_pct', 'seed_int',
                                               'net_rating_m_last30D',
                                               'net_rating_m_vs_topseeds',
                                               'net_rating_m']].\
                  rename(columns={'TeamID':'TeamID_opp',
                                  'w_pct':'w_pct_right',
                                  'seed_int':'seed_int_right', 
                                  'net_rating_m_last30D':'net_rating_m_last30D_right', 
                                  'net_rating_m_vs_topseeds':'net_rating_m_vs_topseeds_right', 
                                  'net_rating_m':'net_rating_m_right'})
    
    df_tourn_out = pd.merge(left=df_tourn_in, 
                            right=df_teams_gr_left, 
                            how='left', on=['Season', 'TeamID'])
    df_tourn_out = pd.merge(left=df_tourn_out, 
                            right=df_teams_gr_right, 
                            how='left', on=['Season', 'TeamID_opp'])

    df_tourn_out['delta_w_pct'] = df_tourn_out['w_pct_left'] -                                          df_tourn_out['w_pct_right']


    df_tourn_out['delta_seed_int'] = df_tourn_out['seed_int_left'] -                                           df_tourn_out['seed_int_right']


    df_tourn_out['delta_net_rating_m'] = df_tourn_out['net_rating_m_left'] - df_tourn_out['net_rating_m_right']
    
    df_tourn_out['delta_net_rating_m_last30D'] = df_tourn_out['net_rating_m_last30D_left'] - df_tourn_out['net_rating_m_last30D_right']
    
    df_tourn_out['delta_net_rating_m_vs_topseeds'] = df_tourn_out['net_rating_m_vs_topseeds_left'] - df_tourn_out['net_rating_m_vs_topseeds_right']
    
    df_out = df_tourn_out.loc[:, ['Season', 'DayNum',
                                  'TeamID', 'TeamID_opp',
                                  'Score_left', 'Score_right',
                                  'win_dummy', 
                                  'delta', 'NumOT', 'delta_w_pct', 
                                  'delta_net_rating_m_last30D',
                                  'delta_net_rating_m_vs_topseeds',
                                  'delta_net_rating_m', 'delta_seed_int']]
                                    
    return df_out

                                    
df_tourn_cl = prepare_tournament_datasets(df_tourn, df_agg_stats)                                    
df_tourn_cl[(df_tourn_cl['Season'].isin([2015, 2016, 2017, 2018]))].head(10)


# ## Quick exploration of Duke Team in 2018
# 
# Here's a quick overview of the Blue Devils performance in 2018.

# In[ ]:


## > DUKE RS
df_agg_stats[(df_agg_stats['TeamName'] == 'Duke') & (df_agg_stats['Season'] == 2018)].head()


# In[ ]:


## > DUKE TOURNAMENT
df_tourn_cl[((df_tourn_cl['TeamID'] == 1181) | (df_tourn_cl['TeamID_opp'] == 1181)) &             (df_tourn_cl['Season'] == 2018)].head(10)


# ## DATA VISUALIZATION
# 
# Here I want to explore:
# 1. Distribution of net ratings during regular season 
# 2. Boxplots of net rating applied to Tournament in order to predict game's outcome
# 3. Correlation plot

# In[ ]:


## > DATA VIZ RS
sns.set(style="ticks", color_codes=True)

df_teams_gr = df_agg_stats.loc[:,['w_pct',
                                  'net_rating_m', 'net_rating_m_last30D', 
                                  'net_rating_m_vs_topseeds', 'pace_m']]

df_teams_gr = df_teams_gr.fillna(0)

#df_teams_gr.describe()
sns.pairplot(df_teams_gr, palette="Set1")


# In[ ]:


## > DATA VIZ TOURNEY
sns.set(style="ticks", color_codes=True)

df_tourn_cl_gr = df_tourn_cl[(df_tourn_cl['Season'].isin([2015, 2016, 2017, 2018]))].reindex()

df_tourn_cl_gr = df_tourn_cl_gr.loc[:,['win_dummy',
                                       'delta_net_rating_m_last30D',
                                       'delta_net_rating_m_vs_topseeds',
                                       'delta_net_rating_m',  
                                       'delta_seed_int']]

fig, ax = plt.subplots(figsize=(11, 7))
sns.boxplot(x="variable", y="value", hue = 'win_dummy', ax=ax, 
            data=pd.melt(df_tourn_cl_gr, id_vars='win_dummy'), palette="Set2")
plt.xticks(rotation=45)


# In[ ]:


## > DATA VIZ TOURNEY
df_tourn_cl_gr = df_tourn_cl[(df_tourn_cl['Season'].isin([2015, 2016, 2017, 2018]))].reindex()

df_tourn_cl_gr = df_tourn_cl_gr.loc[:,['win_dummy',
                                       'delta_w_pct']]

fig, ax = plt.subplots(figsize=(9, 7))
sns.boxplot(x="variable", y="value", hue = 'win_dummy', ax=ax, 
            data=pd.melt(df_tourn_cl_gr, id_vars='win_dummy'), palette="Set2")


# In[ ]:


## > Correlation
# Compute the correlation matrix
df_tourn_cl_gr = df_tourn_cl[(df_tourn_cl['Season'].isin([2015, 2016, 2017, 2018]))].reindex()

df_tourn_cl_gr = df_tourn_cl_gr.loc[:,['win_dummy',
                                       'delta_net_rating_m_last30D',
                                       'delta_net_rating_m_vs_topseeds',                                       
                                       'delta_net_rating_m',  
                                       'delta_w_pct',
                                       'delta_seed_int']].fillna(0)

corr = df_tourn_cl_gr.corr()
fig, ax = plt.subplots(figsize=(11, 7))
sns.heatmap(corr, cmap="YlGnBu", ax = ax)


# In[ ]:


## > AR
def somers2_py(x, y):
    
    from sklearn.metrics import roc_auc_score
    
    C = roc_auc_score(y, x)
    Dxy = (2 * roc_auc_score(y, x))  - 1
    
    return Dxy, C

def apply_somers(df):
    
    d = {}
    
    dxy, cxy = somers2_py(df['value'],
                          df['win_dummy'])
    
    d['Dxy'] = dxy
    d['C'] = cxy
    
    
    return pd.Series(d)

df_tourn_cl_gr = df_tourn_cl[(df_tourn_cl['Season'].isin([2015, 2016, 2017, 2018]))].reindex()

df_tourn_cl_gr = df_tourn_cl_gr.loc[:,['win_dummy',
                                       'delta_net_rating_m_last30D',
                                       'delta_net_rating_m_vs_topseeds',                                       
                                       'delta_net_rating_m',  
                                       'delta_w_pct',
                                       'delta_seed_int']].fillna(0)

df_ar = pd.melt(df_tourn_cl_gr, id_vars='win_dummy')

df_ar.groupby(['variable']).                          apply(apply_somers).                          reset_index().                          sort_values(by=['Dxy'], ascending=False)


# ## Conclusion
# As all of you may have noticed the general and against top-seeded stats seems to display a low correlation with the seeds and high accuracy levels. Which could be good for modeling.  
# Unluckily instead, the last 30 days of the regular season seems to do not have high levels of accuracy; however, this information may become useful in order to predict the outcome of the first games of the tournament.
# 
# ![ZionDunkUrl](https://media.giphy.com/media/3MbRQm86C13FvUAyWV/giphy.gif "zionDunk")
# 
# Hope you guys have enjoyed this first dive into the NCAA data. I would probably add some relevant metrics in order to extend the number of regressors in future models.
# 
# I would like to hear from you if you have some comments and if someone would like to suggest some other advanced statistics which I missed and may have some predictive power.
# 
# If you liked the kernel (or simply loved the Zion gifs) please remember to upvote. ;)

# In[ ]:




