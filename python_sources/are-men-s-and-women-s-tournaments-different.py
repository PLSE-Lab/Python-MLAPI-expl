#!/usr/bin/env python
# coding: utf-8

# The goal of this notebook is to explore the data of both the Men's and Women's competitions and answering the following questions:
# 
# * How did the game evolved over the years?
# * What stats are most useful to predict the outcome of a game?
# * What are the differences between the Men's and Women's competitions?
# 
# To do so, we need to produce some aggregated statistics. The next few hidden cells have all the functions to do just that.

# In[ ]:


import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE

import explore_data as exp
import df_pipeline as df_p

pd.set_option("max_columns", 300)


# In[ ]:


def process_details(data):
    df = data.copy()
    stats = ['Score', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 
             'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 
             'PF', 'FGM2', 'FGA2', 'Tot_Reb', 'FGM_no_ast', 
             'Def_effort', 'Reb_opp', 'possessions', 
             'off_rating', 'def_rating', 'scoring_opp', 
             'TO_perposs', 'impact'] 
        
    for prefix in ['W', 'L']:
        df[prefix+'FG_perc'] = df[prefix+'FGM'] / df[prefix+'FGA']
        df[prefix+'FGM2'] = df[prefix+'FGM'] - df[prefix+'FGM3']
        df[prefix+'FGA2'] = df[prefix+'FGA'] - df[prefix+'FGA3']
        df[prefix+'FG2_perc'] = df[prefix+'FGM2'] / df[prefix+'FGA2']
        df[prefix+'FG3_perc'] = df[prefix+'FGM3'] / df[prefix+'FGA3']
        df[prefix+'FT_perc'] = df[prefix+'FTM'] / df[prefix+'FTA']
        df[prefix+'Tot_Reb'] = df[prefix+'OR'] + df[prefix+'DR']
        df[prefix+'FGM_no_ast'] = df[prefix+'FGM'] - df[prefix+'Ast']
        df[prefix+'FGM_no_ast_perc'] = df[prefix+'FGM_no_ast'] / df[prefix+'FGM']
        df[prefix+'possessions'] = df[prefix+'FGA'] - df[prefix+'OR'] + df[prefix+'TO'] + 0.475*df[prefix+'FTA']
        df[prefix+'off_rating'] = df[prefix+'Score'] / df[prefix+'possessions'] * 100
        df[prefix+'shtg_opportunity'] = 1 + (df[prefix+'OR'] - df[prefix+'TO']) / df[prefix+'possessions']
        df[prefix+'TO_perposs'] = df[prefix+'TO'] / df[prefix+'possessions']
        df[prefix+'IE_temp'] = df[prefix+'Score'] + df[prefix+'FTM'] + df[prefix+'FGM'] +                                 df[prefix+'DR'] + 0.5*df[prefix+'OR'] - df[prefix+'FTA'] - df[prefix+'FGA'] +                                 df[prefix+'Ast'] + df[prefix+'Stl'] + 0.5*df[prefix+'Blk'] - df[prefix+'PF']

    df['Wdef_rating'] = df['Loff_rating']
    df['Ldef_rating'] = df['Woff_rating']

    df['Wimpact'] = df['WIE_temp'] / (df['WIE_temp'] + df['LIE_temp'])
    df['Limpact'] = df['LIE_temp'] / (df['WIE_temp'] + df['LIE_temp'])

    del df['WIE_temp']
    del df['LIE_temp']

    df[[col for col in df.columns if 'perc' in col]] = df[[col for col in df.columns if 'perc' in col]].fillna(0)

    df['WReb_opp'] = df['WDR'] / (df['LFGA'] - df['LFGM'])
    df['LReb_opp'] = df['LDR'] / (df['WFGA'] - df['WFGM'])
    
    return df


def full_stats(data):
    df = data.copy()
    
    to_select = [col for col in df.columns if 'W' in col and '_perc' not in col]
    df_W = df[['Season', 'DayNum', 'NumOT'] + to_select].copy()
    df_W.columns = df_W.columns.str.replace('W','')
    df_W['N_wins'] = 1
    
    to_select = [col for col in df.columns if 'L' in col and '_perc' not in col]
    df_L = df[['Season', 'DayNum', 'NumOT'] + to_select].copy()
    df_L.columns = df_L.columns.str.replace('L','')
    df_L = df_L.rename(columns={'Woc': 'Loc'})
    df_L['N_wins'] = 0
    
    df = pd.concat([df_W, df_L])
    
    del df['DayNum']
    del df['Loc']
    
    to_use = [col for col in df.columns if col != 'NumOT']
    
    means = df[to_use].groupby(['Season','TeamID'], as_index=False).mean()
    
    sums = df[to_use].groupby(['Season','TeamID'], as_index=False).sum()
    sums['FGM_perc'] = sums.FGM / sums.FGA
    sums['FGM2_perc'] = sums.FGM2 / sums.FGA2
    sums['FGM3_perc'] = sums.FGM3 / sums.FGA3
    sums['FT_perc'] = sums.FTM / sums.FTA
    sums['FGM_no_ast_perc'] = sums.FGM_no_ast / sums.FGM
    to_use = ['Season', 'TeamID', 'FGM_perc',
              'FGM2_perc', 'FGM3_perc', 'FT_perc', 
              'FGM_no_ast_perc']
    
    sums = sums[to_use].fillna(0)
    
    stats_tot = pd.merge(means, sums, on=['Season', 'TeamID'])
  
    return stats_tot


def add_seed(seed_location, total):
    seed_data = pd.read_csv(seed_location)
    seed_data['Seed'] = seed_data['Seed'].apply(lambda x: int(x[1:3]))
    total = pd.merge(total, seed_data, how='left', on=['TeamID', 'Season'])
    return total


def make_teams_target(data, league):
    if league == 'men':
        limit = 2003
    else:
        limit = 2010

    df = data[data.Season >= limit].copy()

    df['Team1'] = np.where((df.WTeamID < df.LTeamID), df.WTeamID, df.LTeamID)
    df['Team2'] = np.where((df.WTeamID > df.LTeamID), df.WTeamID, df.LTeamID)
    df['target'] = np.where((df['WTeamID'] < df['LTeamID']),1,0)
    df['target_points'] = np.where((df['WTeamID'] < df['LTeamID']),df.WScore - df.LScore,df.LScore - df.WScore)
    df.loc[df.WLoc == 'N', 'LLoc'] = 'N'
    df.loc[df.WLoc == 'H', 'LLoc'] = 'A'
    df.loc[df.WLoc == 'A', 'LLoc'] = 'H'
    df['T1_Loc'] = np.where((df.WTeamID < df.LTeamID), df.WLoc, df.LLoc)
    df['T2_Loc'] = np.where((df.WTeamID > df.LTeamID), df.WLoc, df.LLoc)
    df['T1_Loc'] = df['T1_Loc'].map({'H': 1, 'A': -1, 'N': 0})
    df['T2_Loc'] = df['T2_Loc'].map({'H': 1, 'A': -1, 'N': 0})

    reverse = data[data.Season >= limit].copy()
    reverse['Team1'] = np.where((reverse.WTeamID > reverse.LTeamID), reverse.WTeamID, reverse.LTeamID)
    reverse['Team2'] = np.where((reverse.WTeamID < reverse.LTeamID), reverse.WTeamID, reverse.LTeamID)
    reverse['target'] = np.where((reverse['WTeamID'] > reverse['LTeamID']),1,0)
    reverse['target_points'] = np.where((reverse['WTeamID'] > reverse['LTeamID']),
                                        reverse.WScore - reverse.LScore,
                                        reverse.LScore - reverse.WScore)
    reverse.loc[reverse.WLoc == 'N', 'LLoc'] = 'N'
    reverse.loc[reverse.WLoc == 'H', 'LLoc'] = 'A'
    reverse.loc[reverse.WLoc == 'A', 'LLoc'] = 'H'
    reverse['T1_Loc'] = np.where((reverse.WTeamID > reverse.LTeamID), reverse.WLoc, reverse.LLoc)
    reverse['T2_Loc'] = np.where((reverse.WTeamID < reverse.LTeamID), reverse.WLoc, reverse.LLoc)
    reverse['T1_Loc'] = reverse['T1_Loc'].map({'H': 1, 'A': -1, 'N': 0})
    reverse['T2_Loc'] = reverse['T2_Loc'].map({'H': 1, 'A': -1, 'N': 0})
    
    df = pd.concat([df, reverse], ignore_index=True)

    to_drop = ['WScore','WTeamID', 'LTeamID', 'LScore', 'WLoc', 'LLoc', 'NumOT']
    for col in to_drop:
        del df[col]
    
    df.loc[:,'ID'] = df.Season.astype(str) + '_' + df.Team1.astype(str) + '_' + df.Team2.astype(str)
    return df


def make_training_data(details, targets):
    tmp = details.copy()
    tmp.columns = ['Season', 'Team1'] +                 ['T1_'+col for col in tmp.columns if col not in ['Season', 'TeamID']]
    total = pd.merge(targets, tmp, on=['Season', 'Team1'], how='left')

    tmp = details.copy()
    tmp.columns = ['Season', 'Team2'] +                 ['T2_'+col for col in tmp.columns if col not in ['Season', 'TeamID']]
    total = pd.merge(total, tmp, on=['Season', 'Team2'], how='left')
    
    if total.isnull().any().any():
        raise ValueError('Something went wrong')
        
    stats = [col[3:] for col in total.columns if 'T1_' in col]

    for stat in stats:
        total['delta_'+stat] = total['T1_'+stat] - total['T2_'+stat]
        
    return total


def prepare_data(league):
    save_loc = 'processed_data/' + league + '/'

    if league == 'women':
        main_loc = '../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/'
        regular_season = main_loc + 'WDataFiles_Stage1/WRegularSeasonDetailedResults.csv'
        playoff = main_loc + 'WDataFiles_Stage1/WNCAATourneyDetailedResults.csv'
        playoff_compact = main_loc + 'WDataFiles_Stage1/WNCAATourneyCompactResults.csv'
        seed = main_loc + 'WDataFiles_Stage1/WNCAATourneySeeds.csv'
    else:
        main_loc = '../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/'
        regular_season = main_loc + 'MDataFiles_Stage1/MRegularSeasonDetailedResults.csv'
        playoff = main_loc + 'MDataFiles_Stage1/MNCAATourneyDetailedResults.csv'
        playoff_compact = main_loc + 'MDataFiles_Stage1/MNCAATourneyCompactResults.csv'
        seed = main_loc + 'MDataFiles_Stage1/MNCAATourneySeeds.csv'
    
    # Season stats
    reg = pd.read_csv(regular_season)
    reg = process_details(reg)
    regular_stats = full_stats(reg)
    
    regular_stats = add_seed(seed, regular_stats)
    
    # Playoff stats
    play = pd.read_csv(playoff)
    play = process_details(play)
    playoff_stats = full_stats(play)
    
    playoff_stats = add_seed(seed, playoff_stats)
    
    # Target data generation
    target_data = pd.read_csv(playoff_compact)
    target_data = make_teams_target(target_data, league)
    
    all_reg = make_training_data(regular_stats, target_data)
    
    return all_reg, regular_stats, playoff_stats


def get_coef(pipe):
    '''
    Get dataframe with coefficients of a model in Pipeline
    The step before the model has to have a get_feature_name method
    '''
    imp = pipe.steps[-1][1].coef_.ravel().tolist()
    feats = pipe.steps[-2][1].get_feature_names()
    result = pd.DataFrame({'feat':feats,'score':imp})
    result['abs_res'] = abs(result['score'])
    result = result.sort_values(by=['abs_res'],ascending=False)
    del result['abs_res']
    return result


def get_feature_importance(pipe):
    '''
    Get dataframe with the feature importance of a model in Pipeline
    The step before the model has to have a get_feature_name method
    '''
    imp = pipe.steps[-1][1].feature_importances_.tolist() #it's a pipeline
    feats = pipe.steps[-2][1].get_feature_names()
    result = pd.DataFrame({'feat':feats,'score':imp})
    result = result.sort_values(by=['score'],ascending=False)
    return result


def cv_score(df_train, y_train, kfolds, pipeline, imp_coef=False, predict_proba=False):
    '''
    Train and test a pipeline in kfold cross validation
    Returns the oof predictions for the entire train set and a dataframe with the
    coefficients or feature importances, averaged across the folds, with standard deviation
    '''
    oof = np.zeros(len(df_train))
    train = df_train.copy()
    
    feat_df = pd.DataFrame()
    
    for n_fold, (train_index, test_index) in enumerate(kfolds.split(train.values)):
            
        trn_data = train.iloc[train_index][:]
        val_data = train.iloc[test_index][:]
        
        trn_target = y_train.iloc[train_index].values.ravel()
        val_target = y_train.iloc[test_index].values.ravel()
        
        pipeline.fit(trn_data, trn_target)
        
        if predict_proba:
            oof[test_index] = pipeline.predict_proba(val_data)[:,1]
        else:
            oof[test_index] = pipeline.predict(val_data).ravel()

        if imp_coef:
            try:
                fold_df = get_coef(pipeline)
            except AttributeError:
                fold_df = get_feature_importance(pipeline)
                
            fold_df['fold'] = n_fold + 1
            feat_df = pd.concat([feat_df, fold_df], axis=0)
       
    if imp_coef:
        feat_df = feat_df.groupby('feat')['score'].agg(['mean', 'std'])
        feat_df['abs_sco'] = (abs(feat_df['mean']))
        feat_df = feat_df.sort_values(by=['abs_sco'],ascending=False)
        del feat_df['abs_sco']
        return oof, feat_df
    else:    
        return oof


# In[ ]:


men_train, men_reg, men_play = prepare_data('men')

men_reg.head()


# In[ ]:


women_train, women_reg, women_play = prepare_data('women')

women_reg.head()


# # Teams over the years
# 
# It must first be noticed that both competitions saw an increase in the number of teams participating, showing a somewhat similar pattern in the period from 2010 to last year.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(15,6), sharey=True)

fig.suptitle('Number of teams competing', fontsize=18)
men_reg.groupby('Season').TeamID.nunique().plot(ax=ax[0])
women_reg.groupby('Season').TeamID.nunique().plot(ax=ax[1])

ax[0].set_title("Men's competition", fontsize=14)
ax[1].set_title("Women's competition", fontsize=14)

ax[0].axvline(2010, color='r', linestyle='--')

plt.show()


# We can then look at how the game has changed over the years by looking at how, on average, the stats are changing season by season

# In[ ]:


stats = ['Score', 'FGA', 'FGM', 'FGM_perc', 'FGA3', 'FGM3', 'FGM3_perc', 'FT_perc', 
         'DR', 'OR', 'Ast', 'TO', 'Stl', 'Blk', 'possessions', 'off_rating']

for col in stats:

    fig, ax = plt.subplots(1,2, figsize=(15,6), sharey=True)

    fig.suptitle(col, fontsize=18)
    men_reg.groupby('Season')[col].mean().plot(ax=ax[0], label='Men')
    women_reg.groupby('Season')[col].mean().plot(ax=ax[0], label='Women')

    men_play.groupby('Season')[col].mean().plot(ax=ax[1], label='Men')
    women_play.groupby('Season')[col].mean().plot(ax=ax[1], label='Women')

    ax[0].set_title("Regular Season", fontsize=14)
    ax[1].set_title("NCAA Tourney", fontsize=14)

    ax[0].axvline(2010, color='r', linestyle='--')
    ax[1].axvline(2010, color='r', linestyle='--')
    
    ax[0].legend()
    ax[1].legend()

    plt.show()


# * Men tend to score more than Women, although the Men score less during the playoff while the Women have scored a similar amount of points both during regular season and playoffs.
# * Both Men and Women are attempting more and more shots. In particular, the number of shots attempted in a Men's game increase quite dramatically in 2016.
# * Men are more accurate when shooting and the accuracy remained stable over the years. We again see a drop in shooting accuracy during the playoff, probably due to a more fierce defense.
# * The increased number of shot attempted seems to be coming from behind the 3 points line. We can also see that this did not affect accuracy. This is probably telling us that the game, very much as in the NBA and WNBA, has changed towards, focusing its action along the perimeter more and more.
# * The free throw percentage of Men and Women is very similar and it has been slightly increasing
# * Women used to get way more defensive rebounds, but since 2016 Men have caught them up and the difference, both in the regular season and in the playoffs, is minimal. This might be related to the larger number of shots taken from the perimeter, which are low percentage shots and thus increase the opportunities for a defensive rebound.
# * Similarly, the offensive rebounds have been consistently decreasing, with the Women having the upper hand.
# * If in the regular season the number of assists is staying the same for Men and Women (with the now usual jump in 2016 for the Men), we observe opposite trends in the NCAA Tourney. Here, Women are getting more and more assists while Men show a more individualistic game.
# * Women have more turnover but the number, for both Men and Women, is decreasing. Almost as a mirror image, Women steal the ball more.
# * The number of blocks is very similar between Men and Women and it is remaining reasonably stable.
# * Women have more possessions per game but Men are catching up since 2016
# 
# The main takeaway, once again, is that something seems to have changed in the Men's game since 2016.
# 
# # Turning up the heat
# 
# It is no mystery that teams want to bring their A-game in the playoff. This section will explore if this is always the case. We aggregate by seed to see if the team quality is somewhat influencing any pattern when we compare regular season and playoff statistics. The lines in the corresponding colors is the mean value of the given statistic in the regular season and in the playoffs.

# In[ ]:


def newline(ax, p1, p2, color='black'):
    l = mlines.Line2D([p1[0],p2[0]], [p1[1],p2[1]], color=color)
    ax.add_line(l)
    return ax


men_tot = pd.merge(men_reg, men_play, on=['Season', 'TeamID', 'Seed'], how='inner')
women_tot = pd.merge(women_reg, women_play, on=['Season', 'TeamID', 'Seed'], how='inner')

stats = ['Score', 'FGA', 'FGM', 'FGM_perc', 'FGA3', 'FGM3', 'FGM3_perc', 'FT_perc', 
         'DR', 'OR', 'Ast', 'TO', 'Stl', 'Blk', 'possessions', 'off_rating']

for stat in stats:
    
    fig, ax = plt.subplots(1,2, figsize=(15,8), sharex=True)
    
    fig.suptitle(stat, fontsize=18)
    
    men = men_tot[['Seed', f'{stat}_x', f'{stat}_y']].copy()
    men.rename(columns={f'{stat}_x': 'Regular', f'{stat}_y': 'Playoff'}, inplace=True)
    mean_reg = men['Regular'].mean()
    mean_play = men['Playoff'].mean()
    ax[0].axvline(mean_reg, color='#0e668b', linestyle='--')
    ax[0].axvline(mean_play, color='#ff0000', linestyle='--')
    men = men.groupby('Seed').mean().sort_values('Seed', ascending=True).reset_index()
    ax[0].scatter(y=men['Seed'], x=men['Regular'], s=80, color='#0e668b', alpha=0.5, label='Regular')
    ax[0].scatter(y=men['Seed'], x=men['Playoff'], s=80, color='#ff0000', alpha=0.6, label='Playoff')
    ax[0].legend()
    ax[0].set_ylabel('Seed', fontsize=12)
    ax[0].set_yticks(np.arange(1, 17, 1))
    
    women = women_tot[['Seed', f'{stat}_x', f'{stat}_y']].copy()
    women.rename(columns={f'{stat}_x': 'Regular', f'{stat}_y': 'Playoff'}, inplace=True)
    mean_reg = women['Regular'].mean()
    mean_play = women['Playoff'].mean()
    ax[1].axvline(mean_reg, color='#0e668b', linestyle='--')
    ax[1].axvline(mean_play, color='#ff0000', linestyle='--')
    women = women.groupby('Seed').mean().sort_values('Seed', ascending=True).reset_index()
    ax[1].scatter(y=women['Seed'], x=women['Regular'], s=80, color='#0e668b', alpha=0.5, label='Regular')
    ax[1].scatter(y=women['Seed'], x=women['Playoff'], s=80, color='#ff0000', alpha=0.6, label='Playoff')
    ax[1].legend()
    ax[1].set_ylabel('Seed', fontsize=12)
    ax[1].set_yticks(np.arange(1, 17, 1))
    
    for i, p1, p2 in zip(men['Seed'], men['Regular'], men['Playoff']):
        ax[0] = newline(ax[0], [p1, i], [p2, i])
    for i, p1, p2 in zip(women['Seed'], women['Regular'], women['Playoff']):
        ax[1] = newline(ax[1], [p1, i], [p2, i])
        
    ax[0].set_title("Men's Competition", fontsize=14)
    ax[1].set_title("Women's Competition", fontsize=14)
    ax[0].set_ylim(ax[0].get_ylim()[::-1])
    ax[1].set_ylim(ax[1].get_ylim()[::-1])
    
    plt.show()
    


# Nothing particularly surprising, but we can call out a few things:
# 
# * All the team generally score fewer points during the playoff, but this is more evident for lower seed teams. In particular, in the Women's competition, the drop is very large for low ranked teams. This is very understandable, considering that Seed 1 and Seed 16 teams generally meet in the very first round.
# * The Field Goals Attempted, however, tend to go higher during the playoffs
# * In particular, lower ranked teams tend to shot the ball from 3 more often, with less success.
# * The Free Throws percentage is not changing much during the playoffs, but higher-ranked teams are stepping up their game
# * Similarly to scoring, defensive rebounds are a key statistic for higher ranking teams. Moreover, top teams tend to have on average the same amount of rebounds they had in the regular season.
# * Both Men and Women tend to share the ball less during the playoffs but the gap between high and low seeds is more evident in the Women's tournament
# * The number of turnovers stays roughly the same but the number of steals decreases for both Men and Women
# * The same can be said for the number of Blocked shots, except for Seed 1 Women's teams that stay consistent with their regular season performance
# * The offensive rating of the Number 1 seed in the Women's tournament is much higher than the rest of the pack and it even goes up during the playoff.
# 
# # What makes a winner?
# 
# We turn our attention to the Turney games and on how the regular season statistics can help us to predict their outcome.
# 
# First, let's see what features most correlate with the point difference for the **Men's competition**

# In[ ]:


men_corr = high_corr = exp.plot_correlations(men_train, target='target_points', limit=12, annot=True)


# Ignoring the feature `target` as it is obviously very correlated, we see how the Seed of the competing teams is the most correlated feature, either as a difference between the two teams or simply as individual entry. This is very correlated with pretty much all the remaining top features. We can see this correlation even further if we plot these features against the point difference.

# In[ ]:


exp.corr_target(men_train, 'target_points', list(men_corr[2:].index), x_estimator=None)


# Which is suggesting that rather than the individual statistics of Team 1 and 2, it might be a good idea to focus on the differences between Team 1 and Team 2.
# 
# Similarly, for the **Women's competition** we get

# In[ ]:


women_corr = high_corr = exp.plot_correlations(women_train, target='target_points', limit=12, annot=True)


# We notice that **the correlations are much stronger**, confirming the importance of the Seed in predicting the outcome of the games.
# 
# Further investigation confirms this insight

# In[ ]:


exp.corr_target(women_train, 'target_points', list(women_corr[2:].index), x_estimator=None)


# Showing a much sharper trend than in the Men's counterpart. 
# 
# This invites a further investigation: **how frequently an advantage in a statistic translates into a victory?** 
# 
# *Note: the statistics are considering the performance during the regular season as a whole, we are not looking at game statistics vs outcome of the same game*

# In[ ]:


men_delta = men_train[['Season', 'target', 'target_points'] + [col for col in men_train if 'delta_' in col and 'Loc' not in col]].copy()
women_delta = women_train[['Season', 'target', 'target_points'] + [col for col in women_train if 'delta_' in col and 'Loc' not in col]].copy()

men_scores = []
men_feats = []
women_scores = []
women_feats = []

for col in [col for col in men_delta if 'delta_' in col]:
    men_delta[col] = np.sign(men_delta[col])
    women_delta[col] = np.sign(women_delta[col])
    if 'Seed' in col or col=='delta_TO':
        men_delta[col] = - men_delta[col]
        women_delta[col] = - women_delta[col]
    try:
        men_scores.append(men_delta.groupby(col)['target'].mean()[1])
        men_feats.append(col)
    except KeyError:
        pass
    try:
        women_scores.append(women_delta.groupby(col)['target'].mean()[1])
        women_feats.append(col)
    except KeyError:
        pass
    
men_prob = pd.DataFrame({'feat': men_feats, 'Men': men_scores})
women_prob = pd.DataFrame({'feat': women_feats, 'Women': women_scores})

tot_prob = pd.merge(men_prob, women_prob, on='feat').sort_values('Men', ascending=False)

tot_prob['feat'] = tot_prob.feat.str.replace('delta_', '')

fig, ax = plt.subplots(1, figsize=(8,15))

ax.scatter(y=tot_prob['feat'], x=tot_prob['Men'], s=80, color='g', alpha=0.6, label='Men')
ax.scatter(y=tot_prob['feat'], x=tot_prob['Women'], s=80, color='r', alpha=0.6, label='Women')
ax.legend()

ax.axvline(0.5, color='k', linestyle='--', alpha=0.3)
ax.set_ylabel('')
ax.set_xlim((0,1))

ax.set_title('Percentage of wins given the stat advantage', fontsize=16)
ax.grid(axis='x')
ax.set_xticklabels(['{:,.0%}'.format(x) for x in ax.get_xticks()])

for i, p1, p2 in zip(tot_prob['feat'], tot_prob['Men'], tot_prob['Women']):
        ax = newline(ax, [p1, i], [p2, i])

plt.show()


# We thus see how, with very few exceptions, dominance in one statistics during the regular season tends to translate to a victory in the Tourney more often for Women than for Men. In general, **having a better seed than your opponent translates to a victory more than 70% of the time**.
# 
# If we look at the distribution of score differences, we see how on average a team in the Men's tournament with a better seed wins by 6-7 points

# In[ ]:


exp.segm_target(men_delta, cat='delta_Seed', target='target_points')


# While in the Women's tournament the team with the better seed wins on average by 13 points

# In[ ]:


exp.segm_target(women_delta, cat='delta_Seed', target='target_points')


# Breaking this trend down by season, we see that we go from the very predictable Women's tourney of 2012 to the very unpredictable Men's tourney of 2014

# In[ ]:


def plot_perc_Season(men, women, feature):
    mean_men = men.groupby(feature).target.mean()[1]
    mean_women = women.groupby(feature).target.mean()[1]
    
    fig, ax = plt.subplots(1, figsize=(15, 6))
    
    tmp = men.groupby(['Season'] +[feature], as_index=False).target.mean().rename(columns={'target': 'Men'})
    tmp = tmp[tmp[feature] == 1]
    tmp.plot(x='Season', y='Men', ax=ax, color='g')
    ax.axhline(mean_men, color='g', linestyle='--', alpha=0.5)
    
    tmp = women.groupby(['Season'] +[feature], as_index=False).target.mean().rename(columns={'target': 'Women'})
    tmp = tmp[tmp[feature] == 1]
    tmp.plot(x='Season', y='Women', ax=ax, color='r')
    ax.axhline(mean_women, color='r', linestyle='--', alpha=0.5)
    
    ax.set_xlim((2003,2019))
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
    
    ax.set_title(f'Percentage of wins given the advantage in {feature.replace("delta_", "")}', fontsize=18)
    
    plt.show()


# In[ ]:


plot_perc_Season(men_delta, women_delta, 'delta_Seed')


# On this line of exploration, we notice that having the advantage in Field Goals Made during the regular season is becoming less and less important for the Men's tourney

# In[ ]:


plot_perc_Season(men_delta, women_delta, 'delta_FGM')


# While precision on the perimeter during the regular season is becoming more and more relevant in the Women's Tourney

# In[ ]:


plot_perc_Season(men_delta, women_delta, 'delta_FGM3_perc')


# Another surprising trend is how having an advantage in Free Throws made during the regular season is becoming less and less important for the victory in the Men's Tourney

# In[ ]:


plot_perc_Season(men_delta, women_delta, 'delta_FTM')


# Furthermore, we notice how winning teams in the Women's tournament are taking better and better care of the ball during the regular season 
# 
# (*TO, as well as Seed, is a statistic that you want to be as low as possible, but here we reversed the sign to make it more similar to the other stats*)

# In[ ]:


plot_perc_Season(men_delta, women_delta, 'delta_TO')


# # Adversarial validation
# 
# Here the idea is to train a model to distinguish between the Men's and the Women's tournaments and further identify important features.

# In[ ]:


kfolds = KFold(n_splits=7, shuffle=True, random_state=345)

men = men_train.copy()
women = women_train.copy()
# target variable we want to predict
men['tourney'] = 1
women['tourney'] = 0
tot_adv = pd.concat([men, women], ignore_index=True)
tot_adv.drop(['ID', 'DayNum', 'Team1', 'Team2', 'Season'], axis=1, inplace=True)

train = tot_adv.drop(['tourney'] + 
                     [col for col in tot_adv if '_rating' in col or 'possessions' in col], axis=1) # dropping some very obvious features
y_train = tot_adv['tourney']

pipe = Pipeline([('scl', df_p.df_scaler())] + [('forest', RandomForestClassifier(n_estimators=500, 
                                                                                min_samples_split=40, 
                                                                                min_samples_leaf=20, 
                                                                                max_features='sqrt', 
                                                                                n_jobs=4))])
oof, imp_coef = cv_score(train, y_train, kfolds, pipe, imp_coef=True)
print(f'ROC AUC score: {round(roc_auc_score(y_true=y_train, y_score=oof),4)}')
plt.figure(figsize=(14, 12))
sns.barplot(x="mean", y="feat", 
            data=imp_coef.head(50).reset_index(), 
            xerr=imp_coef.head(50)['std'])
plt.show()


# We see that the model is very good in finding out if a game has been played in the Women's tournament or in the Men's one. The most important features for this prediction are the number of rebounds, of field goals attempted, the precision on these field goals, and the number of tournovers.
# 
# Moreover, we can also see how different the 2 tournaments are by using t-SNE, which shows how only a few games can cross the border betweem the Men's and Women's tournaments.

# In[ ]:


tsne = TSNE(n_components=2, init='pca', random_state=51, perplexity=100, learning_rate=100)

green = tot_adv['tourney'] == 1
red = tot_adv['tourney'] == 0

y_total = tsne.fit_transform(tot_adv.drop(['target_points', 'tourney', 'target'], axis=1))                         
                           
fig, ax = plt.subplots(1, figsize=(15,8))

ax.scatter(y_total[red, 0], y_total[red, 1], c="r", alpha=0.7, label='Women')
ax.scatter(y_total[green, 0], y_total[green, 1], c="g", alpha=0.7, label='Men')
ax.legend()
plt.show()


# At last, comparing the embeddings for the Men's and Women's tournaments separately, we notice once more how the two tournaments differ with a more net split between wins and losses in the Women's tournament.

# In[ ]:


tsne = TSNE(n_components=2, init='pca', random_state=51, perplexity=50, learning_rate=300)

red_m = men_train['target'] == 1
green_m = men_train['target'] == 0
red_w = women_train['target'] == 1
green_w = women_train['target'] == 0

y_men = tsne.fit_transform(men_train.drop(['ID', 'DayNum', 'Team1', 'Team2', 'Season'] + 
                                          ['target'], axis=1))
y_women = tsne.fit_transform(women_train.drop(['ID', 'DayNum', 'Team1', 'Team2', 'Season'] + 
                                          ['target'], axis=1))
                           
fig, ax = plt.subplots(1,2, figsize=(15,7))

ax[0].scatter(y_men[red_m, 0], y_men[red_m, 1], c="orange", alpha=0.8, label='Win')
ax[0].scatter(y_men[green_m, 0], y_men[green_m, 1], c="b", alpha=0.5, label='Loss')
ax[1].scatter(y_women[red_w, 0], y_women[red_w, 1], c="orange", alpha=0.8, label='Win')
ax[1].scatter(y_women[green_w, 0], y_women[green_w, 1], c="b", alpha=0.5, label='Loss')
ax[0].legend()
ax[1].legend()
ax[0].set_title("Men's Tournament", fontsize=16)
ax[1].set_title("Women's Tournament", fontsize=16)
plt.show()


# Given these results, it might not be a good idea to train one model for both competitions.
