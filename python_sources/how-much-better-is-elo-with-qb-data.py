#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Addition of QB Data
# 
# In 2019, the 538 Elo prediction added QB data as well as some additional features. However, it appears that the retrofitted their model back to 1950 with QB. This analysis will do a simple statistical comparision to find out whether QB data improves predictions.
# 

# In[ ]:





# ## Quick Look at the Data

# In[ ]:


all_elo = pd.read_csv('/kaggle/input/nfl-elo-ratings-from-538/nfl_elo.csv')


# In[ ]:


all_elo.head(15)


# In[ ]:


all_elo[~all_elo.qb1_game_value.isna()][['qb1','qb2','elo1_pre','elo2_pre','qbelo1_pre','qbelo2_pre','qb1_game_value','qb2_game_value']]


# ## Background
# 
# An obvious first thing we need is a team identification table. The team identifiers will not be recognized by any modern day NFL fan. The first two seasons were from a professional league called the American Professional Football Association, which became the National Football Leaguge (NFL) in 1922. It grew from the Ohio League and the New Your Pro Football League.
# 
# The very first game between Rock Island Independents (RII) and the St. Paul Ideals (STP) was won by Rock Island 48-0.   
# 
# In the first season, the league title was won by the Akron Pros, who were undefeated (with several ties). 
# 
# In the modern NFL, only two teams survive, the Chicago Cardinals (now Arizona Cardinals) and the Decatur Staleys, now the Chicago Bears. The Muncie Flyers had a short season, with a record of 0-1. They did play a few other games that year against local teams; however these games did not count in the APFA standings.

# ## The Elo Rating as a Predictor
# 
# How good is Elo as a predictor? We need a few enhancements to the dataset.
# 
# For this analysis, I will discard tie games, unless Elo predicts a tie.
# 
# 
# 
# 

# In[ ]:


len(all_elo)


# In[ ]:


#First, remove the NaN values (only in season 2019)
all_elo = all_elo[~all_elo.score1.isna()==True]
all_elo ['predicted_tie'] = all_elo.elo1_pre == all_elo.elo2_pre
all_elo ['actual_tie'] = all_elo.score1 == all_elo.score2


# In[ ]:


all_elo [all_elo.predicted_tie == True]


# There are only 7 games with predicted ties. These were all in league start-up seasons when there was no prior history. 

# In[ ]:


all_elo [all_elo.actual_tie == True]


# There have been 315 tie games since the start of the NFL. This has fallen off dramatically in the last 45 years, with the introduction of overtime in 1974. The overtime rules have been tweaked several times in the last 10 years. 

# In[ ]:


import plotly.express as px
ties_by_season = pd.DataFrame(all_elo.groupby(['season'])['actual_tie'].sum())
ties_by_season = ties_by_season.reset_index(drop=False)
print(ties_by_season.columns)
#fig, ax = plt.subplots()
fig = px.bar(ties_by_season, x='season', y='actual_tie',width = 600,height = 400, orientation = 'v')
fig.show()


# ## Check on Prediction
# 
# We first eliminate the tie predictions and actual ties. 

# In[ ]:


non_ties = all_elo[~all_elo.actual_tie == True]
non_ties = non_ties[~non_ties.predicted_tie == True]


# In[ ]:


non_ties


# In[ ]:


non_ties ['team1_predicted'] = non_ties.elo1_pre > non_ties.elo2_pre
non_ties ['team1_won'] = non_ties.score1 > non_ties.score2


# In[ ]:




non_ties['correct_prediction']=non_ties.team1_predicted == non_ties.team1_won


# In[ ]:


non_ties[['team1_predicted','team1_won','correct_prediction']]


# In[ ]:


non_ties.tail()


# In[ ]:


non_ties.correct_prediction.describe().freq/len(non_ties)


# An overall prediction percentage of 64.8% seems quite good. How well does it compare to industry experts?
# 
# This year, Trey Wingo of ESPN has predicted 158 of 240 NFL games so far. This is 65.8%, almost the same as Elo. How did Elo do in 2019?

# In[ ]:


non_ties_2019 = non_ties[non_ties.season==2019]
non_ties_2019.correct_prediction.describe().freq/len(non_ties_2019)


# In[ ]:


non_ties_2019.correct_prediction.describe()


# For 2019, Elo performs slighly lower with 64.0% correct. That would still place Elo in the top 6 of professional football predictors in the sports media!
# 
# ## How Does Elo Do Against Point Spread?
# 
# The Elo predictions are design such that the prediction difference divided by 25 is an effective point spread. With this known, we now examine Elo's prediction against its own point spreads.

# In[ ]:


non_ties['point_spread'] = (non_ties.elo2_pre - non_ties.elo1_pre)/25.0


# In[ ]:


non_ties.point_spread.describe()


# Based on the mean of the point spread of -0.15, we can assume team1 is the home team, with a slight advantage in point spread.
# 

# In[ ]:


non_ties[non_ties.point_spread <=-20]


# In[ ]:


non_ties['score1_adj'] = non_ties.score1 + non_ties.point_spread


# In[ ]:


non_ties['team1_won_adj'] = non_ties.score1_adj > non_ties.score2


# In[ ]:


non_ties[non_ties.team1_won != non_ties.team1_won_adj]

There were over 2000 games where the point spread changed the outcome, i.e. a team beat the spread. How that change the prediction?

# In[ ]:


non_ties['correct_spread_prediction']=non_ties.team1_predicted == non_ties.team1_won_adj


# In[ ]:


non_ties.correct_spread_prediction.describe().freq/len(non_ties)


# The point spread predictions are better than 50% at 52.4%. That is probably not enough to beat the line in Las Vegas, due to the house advantage of 10%. You might make some money in your office pool!

# ## Finally - Prediction With and Without QB Information
# 
# We know that Elo is a decent predictor overall with correct predictions of 64.0%. We need a statistical test to compare two ratios - win percentage before and after 1950.
# 
# One test we can use is a z statistic to compare two proportions. 
# 
# 

# In[ ]:


from statsmodels.stats.proportion import proportions_ztest
before_1950 = non_ties[non_ties.season<1950]
since_1950 = non_ties[non_ties.season>=1950]


# In[ ]:


print('Probability of correct prediction before 1950: ' +str(before_1950.correct_prediction.describe().freq/len(before_1950)))
print('Probability of correct since 1950:             ' + str(since_1950.correct_prediction.describe().freq/len(since_1950)))


# This is interesting. Why are the predictions better before the addition of the quarterback and home field data? 
# 
# We now have a null hypothesis to test:
# 
# * H0 - Prediction probability of games before 1950 and games since 1950 are the same; P(before) = P(since)
# * H1 - Prediction probability of games before > games since 1950; P(before) > P(since)
# * Test is at 
# 
# 

# In[ ]:


# Get sample sizes for each
n_pre = len(before_1950)
n_post = len(since_1950)
correct_pre = before_1950.correct_prediction.describe().freq
correct_post = since_1950.correct_prediction.describe().freq


# In[ ]:


correct = np.array([correct_pre, correct_post])
overall = np.array([n_pre, n_post])
z, pval = proportions_ztest(correct, overall)


# In[ ]:


print ('The results of the z proportion test are: ')
print ('  z = ' + str(round(z,2)) )
print ('  p = ' + str((pval)) )


# The P-value is the probability that the z-score is greater than than 2.13. Therefore, with a z value of 7.04, we reject the null hypothesis and conclude that the addition of QB ratings did not improve the Elo prediction algorithm.
# 
# This is a curious finding. It seems that, in general, addition of predictive factors should improve the prediction function.
