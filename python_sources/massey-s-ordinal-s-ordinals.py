#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Massey Ordinals is a compilation of ranking systems. In this notebook we compare those systems in terms of their performance. In other words, we get a ranking of rankings. 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (15, 9)

season_df = pd.read_csv('../input/datafiles/RegularSeasonCompactResults.csv')
tourney_df = pd.read_csv('../input/datafiles/NCAATourneyCompactResults.csv')
ordinals_df = pd.read_csv('../input/masseyordinals/MasseyOrdinals.csv')                .rename(columns={'RankingDayNum':'DayNum'})
# Get the last available data from each system previous to the tournament
ordinals_df = ordinals_df.groupby(['SystemName','Season','TeamID']).last().reset_index()
ordinals_df.head()


# ## Merge game and ordinal data

# In[ ]:


# Add winner's ordinals
games_df = tourney_df.merge(ordinals_df,left_on=['Season','WTeamID'],
                          right_on=['Season','TeamID'])
games_df.head()
# Then add losser's ordinals
games_df = games_df.merge(ordinals_df,left_on=['Season','LTeamID','SystemName'],
                          right_on=['Season','TeamID','SystemName'],
                          suffixes = ['W','L'])
games_df.head()


# ## How accurate are the different systems?
# Here we evaluate the accuracy of each system (i.e. the percentage of times in which the result agrees with the ranking difference) and plot the top 50 systems. 

# In[ ]:


## Add column with 1 if result is correct
games_df = games_df.drop(labels=['TeamIDW','TeamIDL'],axis=1)
games_df['prediction'] = (games_df.OrdinalRankW<games_df.OrdinalRankL).astype(int)
results_by_system = games_df.groupby('SystemName').agg({'prediction':('mean','count')})
results_by_system['prediction']['mean'].sort_values(ascending=False)[:50].plot.bar(ylim=[.7,.8])


# This figure does not take into account the consistency of the accuracy, i.e. the length of the track record evaluated. To account for that, we show the accuracy together with the sample size for each system

# In[ ]:


results_by_system['prediction'].sort_values('mean',ascending=False)[:50]


# ## Evaluation of predictions based on the competition's metrics
# Ok, we have measured the accuracy, but how are the systems ranked according to the log-loss?
# 
# To do that,  we transform the rank into an elo-like rating given the following formula (see this [post](https://www.kaggle.com/c/march-machine-learning-mania-2014/discussion/6777) by Jeff Sonas):
# 
# Rating = 100 - 4*LN(rank+1) - rank/22
# 
# And then compute the probability of team A beating team B is:
# 
# prob = 1/(1+POWER(10,-RatingDiff/15))
# 
# We focus on the four last seasons, so that the metrics computed should be in agreement with the leaderboard score.

# In[ ]:


games_df['Wrating'] = 100-4*np.log(games_df['OrdinalRankW']+1)-games_df['OrdinalRankW']/22
games_df['Lrating'] = 100-4*np.log(games_df['OrdinalRankL']+1)-games_df['OrdinalRankL']/22
games_df['prob'] = 1/(1+10**((games_df['Lrating']-games_df['Wrating'])/15))
loss_results = games_df[games_df.Season>=2015].groupby('SystemName')['prob'].agg([('loss',lambda p: -np.mean(np.log(p))),('count','count')])
loss_results['loss'].sort_values()[:50].plot.bar(ylim=[.4,.6])


# ## Let us choose a system and generate a submission

# In[ ]:


ref_system = 'POM'
ordinals_df['Rating']= 100-4*np.log(ordinals_df['OrdinalRank']+1)-ordinals_df['OrdinalRank']/22
ordinals_df = ordinals_df[ordinals_df.SystemName==ref_system]
# Get submission file
submission_df = pd.read_csv('../input/SampleSubmissionStage1.csv')

submission_df['Season'] = submission_df['ID'].map(lambda x: int(x.split('_')[0]))
submission_df['Team1'] = submission_df['ID'].map(lambda x: int(x.split('_')[1]))
submission_df['Team2'] = submission_df['ID'].map(lambda x: int(x.split('_')[2]))
submission_df = submission_df.merge(ordinals_df[['Season','TeamID','Rating']],how='left',
                                    left_on = ['Season','Team1'], right_on = ['Season','TeamID'])
submission_df = submission_df.merge(ordinals_df[['Season','TeamID','Rating']],how='left',
                                    left_on = ['Season','Team2'], right_on = ['Season','TeamID'],
                                   suffixes=['W','L'])
submission_df['Pred'] = 1/(1+10**((submission_df['RatingL']-submission_df['RatingW'])/15))
submission_df[['ID', 'Pred']].to_csv('submission.csv', index=False)
submission_df[['ID', 'Pred']].head()


# In[ ]:


submission_df.tail()

