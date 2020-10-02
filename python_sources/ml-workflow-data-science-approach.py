#!/usr/bin/env python
# coding: utf-8

# # ML Workflow - Data Science Approach

# The aim of this notebook differs slightly from your traditional kaggle kernel, where submissions tend to be about more computational power, latest NN implementations, and fishing for data leaks. 
# 
# Here we approach the matter at hand with a business/data science methodology, trying to understand what the problem is and how we should model it.
# 
# No more youtube hack videos, hail the power of data and become your best player!

# #### TABLE OF CONTENT
# 
# * [Introduction](#Introduction:)
# * [Back to School](#Back-to-School)
# * [Testing our Assumptions](#Testing-our-Assumptions)
# * [Transforming our Dataset](#Transforming-our-Dataset)
# * [Feature Engineering](#Feature-Engineering)
# * [Machine Learning - Training Grounds](#Machine-Learning---Training-Grounds)
# * [Interpretation](#Interpretation)
# * [Next Steps](#Next-Steps)
# 
# 
# 

# ![pubg](http://media.comicbook.com/2018/01/pubg-4-1070817.jpeg)

# # Introduction
# [Return to top](#TABLE-OF-CONTENT)

# **Context:**
# 
# *You are given over 65,000 games' worth of anonymized player data, split into training and testing sets, and asked to predict final placement from final in-game stats and initial player ratings.*
# 
# *What's the best strategy to win in PUBG? Should you sit in one spot and hide your way into victory, or do you need to be the top shot? Let's let the data do the talking!*

# **Challenge:**
# 
# *In a PUBG game, up to 100 players start in each match (matchId). Players can be on teams (groupId) which get ranked at the end of the game (winPlacePerc) based on how many other teams are still alive when they are eliminated. In game, players can pick up different munitions, revive downed-but-not-out (knocked) teammates, drive vehicles, swim, run, shoot, and experience all of the consequences -- such as falling too far or running themselves over and eliminating themselves.*
# 
# *You are provided with a large number of anonymized PUBG game stats, formatted so that each row contains one player's post-game stats. The data comes from matches of all types: solos, duos, squads, and custom; there is no guarantee of there being 100 players per match, nor at most 4 player per group.*
# 
# *You must create a model which predicts players' finishing placement based on their final stats, on a scale from 1 (first place) to 0 (last place).*
# 

# **Data Dictionary:**
# - **DBNOs** - Number of enemy players knocked.
# - **assists** - Number of enemy players this player damaged that were killed by teammates.
# - **boosts** - Number of boost items used.
# - **damageDealt** - Total damage dealt. Note: Self inflicted damage is subtracted.
# - **headshotKills** - Number of enemy players killed with headshots.
# - **heal** - Number of healing items used.
# - **killPlace** - Ranking in match of number of enemy players killed.
# - **killPoints** - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.)
# - **killStreaks** - Max number of enemy players killed in a short amount of time.
# - **kills** - Number of enemy players killed.
# - **longestKill** - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
# - **matchId** - Integer ID to identify match. There are no matches that are in both the training and testing set.
# - **revives** - Number of times this player revived teammates.
# - **rideDistance** - Total distance traveled in vehicles measured in meters.
# - **roadKills** - Number of kills while in a vehicle.
# - **swimDistance** - Total distance traveled by swimming measured in meters.
# - **teamKills** - Number of times this player killed a teammate.
# - **vehicleDestroys** - Number of vehicles destroyed.
# - **walkDistance** - Total distance traveled on foot measured in meters.
# - **weaponsAcquired** - Number of weapons picked up.
# - **winPoints** - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.)
# - **groupId** - Integer ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
# - **numGroups** - Number of groups we have data for in the match.
# - **maxPlace** - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
# - **winPlacePerc** - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.

# # Back to School
# [Return to top](#TABLE-OF-CONTENT)
# 
# ![school](https://zilliongamer.com/uploads/pubg-mobile/map/all-maps/school-and-apartment.PNG)

# The first question we must ask ourselves is what are we trying to optimize and what is the math?
# Let's look at our problem statement and our target
# 
# > **Problem Statement:** *You must create a model which predicts **players' finishing placement** based on their final stats, on a scale from 1 (first place) to 0 (last place)*
# 
# > **Target variable:** *winPlacePerc - This is a **percentile winning placement**, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.*
# 
# 
# So our goal is to **rank** players and our target variable is a **percentile**. 
# 
# These two bits are critical pieces of information: the goal tells us what kind of **ML technique** we want to use while the target definition tells us about its **distribution**.

# **ML Technique:** What is the metric?
# 
# 
# Ideally, we would like to use a ranking algorithm like lambdarank/lambdamart.
# 
# Learning to Rank (LTR) is a class of techniques that apply supervised machine learning to solve ranking problems. Essentially, the ranking is transformed into a pairwise regression problem. The algorithm compares pairs of items and comes up with the optimal ordering for that pair, iterating through the different pairs to extrapolate with the final ranking of all items.
# 
# The business metric is the rank, not the MAE. Unfortunately, I've found the LGBM lambdarank implementation quite confusing and was unsuccessful at using it. As such we will optimize our algorithm using an MAE objective.
# 
# Limitations using MAE:
# - The predictions are not bound, as such we can have some predictions falling above/below our percentile range
# - The predictions are not going to be unique, you could have several players/teams assigned same scores
# 
# 
# **Target variable:** Percentile
# 
# In a  game of PUBG, each team gets assigned a percentile value so there there should approximatly be the same cout of 0s, 0.5s, 1s etc. (there might be irregularities in the distribution due imbalance in team sizes or number of teams). We should expect to see a uniform distribution for all percentile scores value, and a gaussian distribution with mean 0.5 for the average percentile score per match. We want our target to mimic a uniform distribution as much as possible.
# 
# 

# # Testing our Assumptions
# [Return to top](#TABLE-OF-CONTENT)
# 
# ![assumptions](https://i.ytimg.com/vi/gdvxapI9b_0/maxresdefault.jpg)

# **Importing Libraries:**

# In[ ]:


import os
import sys

import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import  init_notebook_mode, iplot
init_notebook_mode()

import shap

DATA_DIR = '../input'
RANDOM_STATE = 212

pd.options.display.max_columns = 60
pd.options.display.float_format = '{0:.2f}'.format

sns.set_style('darkgrid') 

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def shape(df):
    return '{:,} rows - {:,} columns'.format(df.shape[0], df.shape[1])


# **Reading Data:**

# In[ ]:


data = pd.read_csv('{}/{}'.format(DATA_DIR, 'train.csv'))
data.columns = data.columns.str.lower()
shape(data)


# In[ ]:


data.head()


# **Looking at target distribution:**

# In[ ]:


def plot_hist(x, title):
    
    fig, ax = plt.subplots(figsize=(13,7))
    formatter = plt.FuncFormatter(lambda x, y: '{:,.2f}'.format(x))
    
    ax.yaxis.set_major_formatter(formatter=formatter)
    ax.xaxis.set_major_formatter(formatter=formatter)

    ax.set_title(title)
    sns.distplot(x, bins=50, kde=False, ax=ax);


# In[ ]:


print('The average winning percentile is {:.3f}, the median is {:.3f}'.format(data.winplaceperc.mean(), data.winplaceperc.median()))


# In[ ]:


plot_hist(data.winplaceperc, title='Histogram of winning percentiles')


# From the above, we can see that our percentile distribution isn't exactly as uniform as we expected.There seems to be more cases occuring at low percentiles. This indicates that overall, we have more losers than winners, why is that? 
# 
# The higher counts at the percentile extremes is to be expected. There is **always** some someone getting the percentiles scores of zero and 1, the rest of the scores varying across different matches.
# 
# Let's consider the average winning percentile per match and look at the distribution. As mentioned earlier, we should expect a normal distirbution with mean 0.5

# In[ ]:


data = data.assign(match_mean=data.groupby('matchid').winplaceperc.transform('mean'))
data = data.assign(match_median = data.groupby('matchid').winplaceperc.transform('median'))


# In[ ]:


print('The average match winning percentile is {:.2f}, the median is {:,.2f}'.format(data.winplaceperc.mean(), data.winplaceperc.median()))


# In[ ]:


plot_hist(data.match_mean, title='Histogram of average match winning percentiles');
plot_hist(data.match_median, title='Histogram of median match winning percentiles');


# Interrestingly, the mean distribution is somewhat right-tailed. Why is it that certain games have lower average percentiles, is there anything odd about these games?

# **Looking at team and match sizes:**
# 
# Team size and game sizes are not given metrics but they can be easily derived from the groupid and matchid.

# In[ ]:


data = data.assign(team_size=data.groupby('groupid').groupid.transform('count'))
data = data.assign(max_team_size=data.groupby('matchid').team_size.transform('max'))
data= data.assign(match_size=data.groupby('matchid').id.transform('nunique'))


# In[ ]:


plot_hist(data.match_size, title='Distribution of players per game')


# Luckily this graph turns out as expected. We see the frequency is highest when approaching 100 players which is our maximum size. As such, we wouldn't expect the earlier graphs to be due to a lack of players in a game. 
# 
# It is probably due to team imbalance.

# In[ ]:


print('The largest team has {} team members'.format(data.max_team_size.max()))


# In[ ]:


plot_hist(data.team_size, title='Histogram of team sizes')
plt.xlim(0,12);


# Interestingly we can see that there are teams bigger than what your traditional squad can contain. It is possible that due to this team imbalance, the percentiles scores get distorted. 
# 
# For example, if a team of 50 ends up first or last (assuming the other teams are smaller), the average percentile ranking will be twisted towards the extremes. As such our original assumptions won't apply.
# 
# Reading through some of the kernels and discussions, it's been established that a good proportion of the games are custom games where team imbalance is more pronounced.
# 
# **Here is the hypothesis:** the regular games will be much more statistically driven than the custom games. If we can isolate regular games from custom ones, the data should lend itself to modelling better and the feature importances should be much more meaningful in ranking your day-to-day player!

# # Transforming our Dataset
# [Return to top](#TABLE-OF-CONTENT)

# The goal of this machine learning exercise is to identify what features make or break player ranks. We need to focus our attention on games that are consistent across time and mitigate irregularities.
# 
# In due order, we must first find out if we are in a regular game or a custom game.

# My first approach was to define the game mode using maximum team size:
#     - solo if max_team_size == 1 
#     - duo if max_team_size == 2
#     - squad if max_team_size == 4
#     - custom if max_team_size > 4
#     
# However this approach was somewhat inconclusive as some games would have only a single occurrence of a larger team distorting the labels.

# In[ ]:


data[data.max_team_size == 2].team_size.value_counts()


# In[ ]:


plot_hist(data.max_team_size, title='Distribution of maximum team size')
plt.xlim(0,20);


# My **big assumption** here is that you can have regular games where a player disconnects while still in the lobby, getting replaced by a second player who will inherit the groupid under a different user id
# 
# My second approach was to look at the prevalence of team sizes across a singular game. The logic here would be that in a for each mode, the majority of participants will be concentraded in team sizes of 1, 2, 3-4, and 5+ as follows:
# 
#     - solo if majority of the game's team_size is 1
#     - duo if majority of the game's team_size is 2
#     - squad if majority of the game's team_size is 3 or 4
#     - custom if the majority of the game's team_size is more than 4
#     
#     
# Another approach worth considering would be looking at number of teams per game.

# In[ ]:


data =  data.assign(team_indicator = data.team_size.apply(lambda x: 5 if x>= 5 else x))


# We generate match statistics per game, returning the percentage split of team sizes
# 
# The new fields represent the team size densities per game.
# 

# In[ ]:


data = pd.get_dummies(data, columns=['team_indicator'])
dummy_cols = ['team_indicator_{}'.format(i) for i in np.arange(1,6)]
data[dummy_cols] = data.groupby('matchid')[dummy_cols].transform('mean')
data.head()


# In[ ]:


plot_hist(data[data.max_team_size==2].team_indicator_1, title='Distribution of solo teams density where maximum team size is 2')


# The above graph seems to corroborate the fact that max_team_size is not very indicative of game mode. A recorded max_team_size of 2 consists mostly of solo games!
# 
# Consequently, we probably want to use the team size densities in order to define our game modes. The following filters are not statistically derived but should be reasonnable. Anything that falls outside the filter's scope is considered custom.
# 
# **Note:** Ideally we would like to use unsupervised learning in this case trying to derive clusters of game modes. Improving how you well you define regular games will improve your model's accuracy. Would love to hear some ideas!

# In[ ]:


data.loc[data.team_indicator_1 >= 0.7, 'game_mode'] = 'solo'
data.loc[data.team_indicator_2 >= 0.6, 'game_mode'] = 'duo'
data.loc[(data.team_indicator_3 + data.team_indicator_4) >= 0.5, 'game_mode'] = 'squad'

data.game_mode = np.where((data.team_indicator_5 >= 0.2), 'custom', data.game_mode)


# In[ ]:


data.game_mode = data.game_mode.fillna('custom')


# In[ ]:


data[dummy_cols+['game_mode']].sample(15)


# The sample results classification looks reasonnable.
# 
# Let's isolate the regular games and compare the winning percentile distribution against custom games!

# In[ ]:


print('The average winning percentile for regular games is {:.3f}, the median is {:.3f}'.format(data[data.game_mode!='custom'].winplaceperc.mean(), data[data.game_mode!='custom'].winplaceperc.median()))
print('The average winning percentile for custom games is {:.3f}, the median is {:.3f}'.format(data[data.game_mode=='custom'].winplaceperc.mean(), data[data.game_mode=='custom'].winplaceperc.median()))

plot_hist(data[data.game_mode != 'custom'].winplaceperc, title = 'Histogram of winning percentiles scores for regular games')
plot_hist(data[data.game_mode == 'custom'].winplaceperc, title = 'Histogram of winning percentiles scores for custom games')


# While we never got to the hoped-for 50th percentile average, we can clearly see that statistical difference between the regular and custom games. We were successful in segmenting the game modes, obtaining a more uniform distribution. Our earlier assumptions about regular games seem to hold quite well!

# # Feature Engineering
# [Return to top](#TABLE-OF-CONTENT)

# We're almost ready to start modelling, we'll create a few additional features to help LGBM find interactions more easily.

# In[ ]:


data['max_possible_kills'] = data.match_size - data.team_size
data['total_distance'] = data.ridedistance + data.swimdistance + data.walkdistance
data['total_items_acquired'] = data.boosts + data.heals + data.weaponsacquired
data['items_per_distance'] =  data.total_items_acquired/data.total_distance
data['items_per_distance'] =  data.total_items_acquired/data.total_distance
data['kills_per_distance'] = data.kills/data.total_distance
data['knocked_per_distance'] = data.dbnos/data.total_distance
data['damage_per_distance'] = data.damagedealt/data.total_distance
data['headshot_kill_rate'] = data.headshotkills/data.kills
data['max_kills_by_team'] = data.groupby('groupid').kills.transform('max')
data['total_team_damage'] = data.groupby('groupid').damagedealt.transform('sum')
data['total_team_kills'] =  data.groupby('groupid').kills.transform('sum')
data['total_team_items'] = data.groupby('groupid').total_items_acquired.transform('sum')
data['pct_killed'] = data.kills/data.max_possible_kills
data['pct_knocked'] = data.dbnos/data.max_possible_kills
data['pct_team_killed'] = data.total_team_kills/data.max_possible_kills
data['team_kill_points'] = data.groupby('groupid').killpoints.transform('sum')
data['team_kill_rank'] = data.groupby('groupid').killplace.transform('mean')
data['max_kills_match'] = data.groupby('matchid').kills.transform('max')
data['total_kills_match'] = data.groupby('matchid').kills.transform('sum')
data['total_distance_match'] = data.groupby('matchid').total_distance.sum()
data['map_has_sea'] =  data.groupby('matchid').swimdistance.transform('sum').apply(lambda x: 1 if x>0 else 0)
data.fillna(0, inplace=True)


# In[ ]:


def plot_interactions(df, feature_list, hue_labels=None, sample_size=10000):
    
    '''
    Target to decile should be first
    '''
    sample_df = df.sample(sample_size)
    sample_df.team_size = sample_df.team_size.apply(lambda x: 5 if x>= 5 else x)
    
    colors = pd.qcut(sample_df[feature_list[0]], q=10, labels=np.arange(1,11)).astype(int)
    colorscale = 'RdBu'
    
    trace = [go.Parcoords(
        line = dict(color=colors, colorscale = colorscale),
        dimensions = list([dict(range = [np.round(sample_df[i].quantile(0.01)*0.9, decimals=1),
                                         np.round(sample_df[i].quantile(0.99)*1.1, decimals=1)],
                                label = str(i),
                                values = sample_df[i]) for i in feature_list]))]

    fig = go.Figure(data=trace)
    iplot(fig)


# **How do I get the dub?**
# 
# The below chart is interactive, feel free to try to play around and identify trends! Maybe you can find the key to winning!
# 
# Feel free to fork the notebook and try different features!

# In[ ]:


features = ['winplaceperc', 'walkdistance', 'damagedealt', 'boosts', 'total_items_acquired', 'revives']
plot_interactions(df=data, feature_list=features)


# # Machine Learning - Training Grounds
# [Return to top](#TABLE-OF-CONTENT)
# 
# ![training_grounds](https://i.ytimg.com/vi/UZXM_JRwKpE/maxresdefault.jpg)
# 
# Time to train our model! 
# 
# In the interest of speed and given the rather large nature of our dataset, no cross-validation or hyper-parameter tuning is in this notebook (although I have iterated through a few parameters with early stopping). 
# 
# Ideally we would want to use hyper-parameter optimization using a custom stratified cross-validator (one that would make sure teams don't get split-up!).
# 

# **Configuration:**

# In[ ]:


EXCLUDE_COLS = ['id', 'match_mean', 'match_median', 'team_indicator_5', 'game_mode']
CATEGORICAL_COLS = ['matchid', 'groupid']
TARGET = 'winplaceperc'
TRAIN_SIZE = 0.9
EARLY_STOP_ROUNDS = 10


# In[ ]:


df = data[data.game_mode != 'custom'].drop(EXCLUDE_COLS, axis=1)
df[CATEGORICAL_COLS] =  df[CATEGORICAL_COLS].astype('category')
shape(df)


# **Generating train and validation sets:**

# In[ ]:


def train_validation(df, train_size=TRAIN_SIZE):
    
    unique_games = df.matchid.unique()
    train_index = round(int(unique_games.shape[0]*train_size))
    
    np.random.shuffle(unique_games)
    
    train_id = unique_games[:train_index]
    validation_id = unique_games[train_index:]
    
    train = df[df.matchid.isin(train_id)]
    validation = df[df.matchid.isin(validation_id)]
    
    return train, validation
    
train, validation = train_validation(df)


# **Defining our sample weights:**
# 
# This is quite important and I would love to hear suggestions and ideas.
# 
# Each player in a team will share the same percentile ranking, which we must account for. Having a team of 4 and a team of 3 in one game only accounts for 2 unique scores but 7 observations. Using the team size as a weight should help with that.
# 
# Another thing to consider is incorporating the fact that there are only so many teams in one game. I was thinking predictions should also be weighed by the number of teams or match size but haven't come up with a definite answer, would love some feedback.

# In[ ]:


train_weights = (1/train.team_size)
validation_weights = (1/validation.team_size)


# In[ ]:


X_train = train.drop(TARGET,axis=1)
X_test = validation.drop(TARGET, axis=1)

y_train = train[TARGET]
y_test = validation[TARGET]

shape(X_train), shape(X_test)


# **Defining and fitting our model:**

# In[ ]:


time_0 = datetime.datetime.now()

lgbm = LGBMRegressor(objective='mae', n_estimators=250,  
                     learning_rate=0.3, num_leaves=200, 
                     n_jobs=-1,  random_state=RANDOM_STATE, verbose=0)

lgbm.fit(X_train, y_train, sample_weight=train_weights,
         eval_set=[(X_test, y_test)], eval_sample_weight=[validation_weights], 
         eval_metric='mae', early_stopping_rounds=EARLY_STOP_ROUNDS, 
         verbose=0)

time_1  = datetime.datetime.now()

print('Training took {} seconds. Best iteration is {}'.format((time_1 - time_0).seconds, lgbm.best_iteration_))


# In[ ]:


print('Mean Absolute Error is {:.5f}'.format(mean_absolute_error(y_test, lgbm.predict(X_test, num_iteration=lgbm.best_iteration_), sample_weight=validation_weights)))
print('R2 score is {:.2%}'.format(r2_score(y_test, lgbm.predict(X_test, num_iteration=lgbm.best_iteration_), sample_weight=validation_weights)))


# In[ ]:


def plot_training(lgbm):
    
    fig, ax = plt.subplots(figsize=(13,7))
    losses = lgbm.evals_result_['valid_0']['l1']
    ax.set_ylim(np.max(losses), 0)
    ax.set_xlim(0,100)
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('Mean Asbolute Error')
    ax.set_title('Evolution of MAE over training iterations')
    ax.plot(losses, color='grey');
    
plot_training(lgbm)


# Training is quite fast and the results are quite strong. While the decrease in impurity considerably slows down after a few iterations iterations, I've noticed that the results do continuously improve as you grow more estimators. If you are looking for higher accuracy (but longer computation), try reducing the learning rate and the number of leaves (beware of over-fitting!)

# **Fixing our predictions:**
# 
# As mentioned earlier, the algorithm used optimizes for MAE and won't know the upper and lower bound inherent to percentiles. We most likely will have predictions below 0 and above 1, which we must correct (if we were to increase training time, this would more strongly mitigated). Furthermore each players in a team will most likely get different rankings while it should be identical.

# In[ ]:


results = validation.copy()
results = results.assign(predicted_player_rank=lgbm.predict(X_test, num_iteration=lgbm.best_iteration_))
print('The minimum predicted ranking is {}, the maximum is {}'.format(results.predicted_player_rank.min(), results.predicted_player_rank.max()))


# In[ ]:


results.predicted_player_rank = results.predicted_player_rank.clip(0, 1)
print('The minimum predicted ranking is {}, the maximum is {}'.format(results.predicted_player_rank.min(), results.predicted_player_rank.max()))


# In[ ]:


print('R2 score is {:.2%}'.format(r2_score(y_test, results.predicted_player_rank, sample_weight=validation_weights)))
print('Mean Absolute Error is {:.5f}'.format(mean_absolute_error(y_test, results.predicted_player_rank, sample_weight=validation_weights)))


# Let's homogenize our team ratings

# In[ ]:


results = results.assign(predicted_team_rank_max=results.groupby('groupid').predicted_player_rank.transform('max'))
results = results.assign(predicted_team_rank_mean=results.groupby('groupid').predicted_player_rank.transform('mean'))

print('Using team maximum predicted ranking:')
print('R2 score is {:.2%}'.format(r2_score(y_test, results.predicted_team_rank_max.clip(0, 1), sample_weight=validation_weights)))
print('Mean Absolute Error is {:.5f}'.format(mean_absolute_error(y_test, results.predicted_team_rank_max, sample_weight=validation_weights)))

print('\nUsing team average predicted ranking:')
print('R2 score is {:.2%}'.format(r2_score(y_test, results.predicted_team_rank_mean.clip(0, 1), sample_weight=validation_weights)))
print('Mean Absolute Error is {:.5f}'.format(mean_absolute_error(y_test, results.predicted_team_rank_mean, sample_weight=validation_weights)))


# In[ ]:


sns.jointplot(y_test, results.predicted_team_rank_mean,
              kind='reg', height=12,
              xlim=(-0.1, 1.1), ylim=(-0.1, 1.1),
              color='darkred', scatter_kws={'edgecolor':'w'}, line_kws={'color':'black'});
plt.title('Actual Ranking vs Predicted Ranking');


# The results are quite strongs and our predicted distribution matches the actual distribution quite well.
# 
# Interestingly, the model is missing out on a lot of winners. The good news is: some players in the above the 60th percentile share the same characteristics as players in the top percentile - you might just be the chosen one!

# # Interpretation
# [Return to top](#TABLE-OF-CONTENT)

# In[ ]:


shap.initjs()

SAMPLE_SIZE = 10000
SAMPLE_INDEX = np.random.randint(0, X_test.shape[0], SAMPLE_SIZE)

X = X_test.iloc[SAMPLE_INDEX]

explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X)


# **Feature Importance:**

# In[ ]:


shap.summary_plot(shap_values, X, plot_type='bar', color='lightblue')


#  Distance, end game kill rank and team kill rank are the features that have the largest mean decrease in impurity. This seems very reasonnable.
#  
#  Let's look at feature values and how they correlate to the shap values. This will gives us an idea of what type of values contributes to a higher/lower ranking

# In[ ]:


shap.summary_plot(shap_values, X)


# We can see that higher distance, more boosts, lower kill rank, lower team kill rank, and lower kills per distance travelled lead to higher ranking! Nothing surprising here
# 
# More interesting fields include number of kills, number of kills as a percentage of game size and whether the map has a sea or not.
# - Individual kills don't seem to be particularily correlated with higher rankings, there is more higher ranked players/teams with lower kill counts
# - The lower your number of kills as a percentage of the match size, the higher ranked you can get.
# - If the doesn't have a sea you are more likely to be ranked lower. You shall not swim away!
# 
# 

# In[ ]:


interactions = ['assists', 'boosts', 'damagedealt', 'heals', 'longestkill', 'walkdistance', 'revives']
features = ['team_kill_rank'] * len(interactions)

for i, j in zip(features, interactions):
    shap.dependence_plot(i, shap_values, X, interaction_index=j);


# **Conclusion: The PUBG Commandments**
# 
# - You shall assist your teammates
# - You shall boost yourself
# - You shall deal a lot of damage
# - You shall heal yourself
# - You shall kill from afar
# - You shall walk a lot
# - You shall revive the fallen ones
# 
# and victory shall be yours!

# # Next Steps
# [Return to top](#TABLE-OF-CONTENT)
# 
# - Unsupervised learning for game mode classification. This approach would be stronger than the rule-based one defined here
# - Stronger EDA analysis for a more powerful outlier treatment (i.e. look at trends across teams, identifying individuals farming xp, cheaters, zombies?)
# - Additional feature engineering. Team statistics and match statistics are so important! The algorithm won't necessarily identify the relationship between the teams and the overall game if you don't help it.
# - Build a cross-validator that ensures that no match is split across different folds. This will prevent temporal leaks and inflated accuracy metrics.
# - Try the lambdarank/lambdamart implementations in XGBoost, try quantile regression.
# - Second model built for the custom game mode. I'm thinking fitting a separate model for the custom games, or using some form of stacking including outputs from the "regular games" model.
# 
# Thank you and looking forward to some discussion!
# 
