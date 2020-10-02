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


# In this kernel, I try to find good measurements from league parameters to evaluate the final tourney win count. This count can show us the who wins the tourney at the end. However, when I analyze the league, I see that general statistics of team is not related with final performance. Final performance has something inclusive inside. There can be several methods that can be created. Nonetheless, while I am improvasing my method, I finally thought that again a total metric score for final measurement of wins could be benefical. Nonetheless, I took mostly samples from data and compare it with tourney win results. Then ranking the score and win rate can give us the tournement result in the end. I do not know about NCAA or its process. However, we can say that some parameters of team performance in some time period could be the projector of final tourney performance as a win rate. Thus, this analysis is based on these assumptions.

# In[ ]:


import seaborn as sns


# In[ ]:


women_players = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WPlayers.csv')
man_teams = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MTeams.csv')
woman_comp_results = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WRegularSeasonCompactResults.csv')
woman_detailed_results = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WRegularSeasonDetailedResults.csv')


# In[ ]:


woman_detailed_results


# In[ ]:


woman_comp_results


# # Analysis
# 
# I want to find how the difference between home and away lose and win ratio. Another thing that I am curios is that how much the score difference between home and away team.

# In[ ]:


woman_comp_results['score_diff'] = woman_comp_results['WScore'] - woman_comp_results['LScore']
woman_detailed_results['score_diff'] = woman_detailed_results['WScore'] - woman_detailed_results['LScore']


# In[ ]:


ax = sns.barplot(x="WLoc", y="score_diff", data=woman_comp_results)


# In[ ]:


ax = sns.barplot(x="WLoc", y="score_diff", data=woman_detailed_results)


# In[ ]:


ax = sns.scatterplot(x="WLoc", y="score_diff", data=woman_comp_results)


# In[ ]:


the_best_scoring_teams = woman_comp_results.groupby('WTeamID').mean().sort_values('score_diff',ascending =False)


# In[ ]:


the_best_scoring_teams


# In[ ]:


woman_detailed_results.corr().sort_values('score_diff',ascending =False)


# In[ ]:


the_total_score = woman_detailed_results.groupby('WTeamID').mean().sort_values('score_diff',ascending =False) 


# In[ ]:


the_total_score['total_score'] = the_total_score['score_diff'] +  the_total_score['WFGM']* 0.5 + the_total_score['WAst'] * 0.5+ the_total_score['WStl'] * 0.4 


# In[ ]:


the_total_score


# In[ ]:


woman_detailed_results.groupby('WTeamID').count()


# In[ ]:


the_final_table = pd.merge(woman_detailed_results.groupby('WTeamID').count()['Season'],the_total_score['total_score'],left_index = True,right_index = True)


# In[ ]:


the_final_table.columns = ['win_count','total_score']


# In[ ]:


the_final_table.sort_values('win_count',ascending =False)


# In[ ]:


the_final_table.corr().iloc[1][0]


# I found that there is highly correlation with win count and score of important features. That is said, There are lots of analysis to make for total understanding

# In[ ]:


tourney_detailed_results = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WNCAATourneyDetailedResults.csv')


# In[ ]:


tourney_detailed_results.groupby('WTeamID').count()


# We have 81 different teams on the final dataset. So I will get some samples from league and control how it match with final season matches.

# In[ ]:


tourney_detailed_results['score_diff'] = tourney_detailed_results['WScore'] - tourney_detailed_results['LScore']


# In[ ]:


tourney_total_score = tourney_detailed_results.groupby('WTeamID').mean().sort_values('score_diff',ascending =False) 


# In[ ]:


tourney_total_score.columns


# In[ ]:


tourney_total_score.corr().sort_values('win',ascending = False)


# In[ ]:


tourney_total_score['total_score_tourney'] = tourney_total_score['score_diff'] + tourney_total_score['DayNum'] * 0.6 + tourney_total_score['WFGM']* 0.5 + tourney_total_score['WAst'] * 0.3 


# In[ ]:


the_final_table_tourney = pd.merge(tourney_detailed_results.groupby('WTeamID').count()['Season'],tourney_total_score['total_score_tourney'],left_index = True,right_index = True)


# In[ ]:


the_final_table_tourney.columns = ['win_count_tourney','total_score_tourney']


# In[ ]:





# In[ ]:


def take_samples(number):
    the_list_of_corr = []
    sample = woman_detailed_results.loc[515*number:515*(number+1)]
    the_total_score = sample.groupby('WTeamID').mean().sort_values('score_diff',ascending =False)
    the_total_score['total_score'] = the_total_score['score_diff']  + the_total_score['WScore'] * 0.5 + the_total_score['WFGM']* 0.5 + the_total_score['WAst'] * 0.3+ the_total_score['WDR'] * 0.2 + the_total_score['WBlk'] * 0.2
    the_final_table = pd.merge(sample.groupby('WTeamID').count()['Season'],the_total_score['total_score'],left_index = True,right_index = True)
    the_final_table.columns = ['win_count_league','total_score_league']
    the_final_table = pd.merge(the_final_table_tourney,the_final_table,left_index = True,right_index = True)
    the_final_table['win_count_tourney'] = the_final_table['win_count_tourney'].rank(pct=True)
    the_final_table['total_score_league'] = the_final_table['total_score_league'].rank(pct=True)
    the_list_of_corr.append(the_final_table.corr().iloc[0][3])
    return the_list_of_corr


# In[ ]:


the_list_of_similarity = [take_samples(i)[0] for i in range(100)]


# In[ ]:


take_samples(1)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(the_list_of_similarity)
plt.ylabel('similartiy correlations')
plt.show()


# We take 515 number of samples from league data and collect the score correlation changes from samples to final win rate at tourney. There is no significant relation in the graph and performance has a trend increases and decreases over time. Maybe if we can test all the performances of different teams, with time series analysis, with performance metric, we can predict the final performance.

# # How to optimize this correlation to close the 1?

# Maybe we cannot optimize this metric because it is a different mathematic issue but we can predict final win number with machine learning

# In[ ]:


the_final_table_tourney_league_match = pd.merge(woman_detailed_results,the_final_table_tourney,left_index = True,right_index = True)


# In[ ]:


the_final_table_tourney_league_match['total_score'] = the_final_table_tourney_league_match['score_diff']  + the_final_table_tourney_league_match['WScore'] * 0.5 + the_final_table_tourney_league_match['WFGM']* 0.5 + the_final_table_tourney_league_match['WAst'] * 0.3+ the_final_table_tourney_league_match['WDR'] * 0.2 + the_final_table_tourney_league_match['WBlk'] * 0.2


# In[ ]:


the_final_table_tourney_league_match.loc[the_final_table_tourney_league_match.WLoc == "H","WLoc"] = 1
the_final_table_tourney_league_match.loc[the_final_table_tourney_league_match.WLoc == "A","WLoc"] = 2
the_final_table_tourney_league_match.loc[the_final_table_tourney_league_match.WLoc == "N","WLoc"] = 3


# In[ ]:


from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
import xgboost as xgb
from sklearn.ensemble import BaggingRegressor
import numpy as np 
import pandas as pd 
import random
from sklearn.decomposition import PCA


# In[ ]:


X = the_final_table_tourney_league_match.drop('win_count_tourney',axis = 1)
y = the_final_table_tourney_league_match['win_count_tourney']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


regr = RandomForestRegressor()
regr.fit(X_train, y_train)

predictions = regr.predict(X_test)


# In[ ]:



pca = PCA(n_components=3)
principalComponents_train = pca.fit_transform(X)
sum(pca.explained_variance_ratio_)


# In[ ]:


X['component_1'] = [i[0] for i in principalComponents_train]
X['component_2'] = [i[1] for i in principalComponents_train]
X['component_3'] = [i[2] for i in principalComponents_train]


# In[ ]:


X = the_final_table_tourney_league_match.drop('win_count_tourney',axis = 1)
y = the_final_table_tourney_league_match['win_count_tourney']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


regr = RandomForestRegressor(n_estimators = 400,min_samples_split = 2,min_samples_leaf = 1,max_features= 'sqrt',max_depth =None,bootstrap= False)
regr.fit(X_train, y_train)

predictions = regr.predict(X_test)


# In[ ]:


mean_squared_error(predictions.round(), y_test)


# In[ ]:


ml_df = pd.DataFrame(predictions.round(),y_test).reset_index()


# In[ ]:


ml_df.corr()


# We see that machine learning in this case is futile attempt and old score is better than this

# # Creating Metric for Final Win Rate

# In[ ]:


the_final_table_tourney_league_match.corr().sort_values('win_count_tourney',ascending = False)


# There is no sufficient correlation values for final win rate so I will try different method

# In[ ]:


def take_samples(number):
    the_list_of_corr = []
    the_list_of_parameters = []
    sample = woman_detailed_results.loc[200*number:200*(number+1)]
    random_num_list = [random.randint(-100,100) for i in range(11)]
    the_total_score = sample.groupby('WTeamID').mean().sort_values('score_diff',ascending =False)
    the_total_score['total_score'] = the_total_score['score_diff'] * random_num_list[0] + the_total_score['WScore'] * random_num_list[1]+ the_total_score['WFGM']* random_num_list[2] + the_total_score['WAst'] * random_num_list[3] + the_total_score['WDR'] * random_num_list[4] + the_total_score['WBlk'] * random_num_list[5] + the_total_score['WFGM3'] * random_num_list[6] + the_total_score['WFGA3'] * random_num_list[7] + the_total_score['WFTA'] * random_num_list[8] + the_total_score['WOR'] * random_num_list[9] + the_total_score['WFTM'] * random_num_list[10]
   
    the_final_table = pd.merge(sample.groupby('WTeamID').count()['Season'],the_total_score['total_score'],left_index = True,right_index = True)
    the_final_table.columns = ['win_count_league','total_score_league']
    the_final_table = pd.merge(the_final_table_tourney,the_final_table,left_index = True,right_index = True)
    the_final_table['win_count_tourney'] = the_final_table['win_count_tourney'].rank(pct=True)
    the_final_table['total_score_league'] = the_final_table['total_score_league'].rank(pct=True)
    the_list_of_corr.append(the_final_table.corr().iloc[0][3])
    the_list_of_parameters.append(random_num_list)
    return the_list_of_corr,the_list_of_parameters


# In[ ]:


take_samples(0)


# In[ ]:


the_list_of_similarity_parameters = [take_samples(i) for i in range(100)]
the_similarity_list = [i[0] for i in the_list_of_similarity_parameters]


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(the_similarity_list)
plt.ylabel('similartiy correlations')
plt.show()


# In[ ]:


the_max_value = 0
parameter_list = []
for i in range(100):
    the_list_of_similarity_parameters = [take_samples(i) for i in range(200)]
    the_similarity_list = [i[0] for i in the_list_of_similarity_parameters]
    the_parameter_list  = [i[1] for i in the_list_of_similarity_parameters]
    if max(the_similarity_list)[0] > the_max_value:
        the_max_value = max(the_similarity_list)[0]
        parameter_list = the_parameter_list[the_similarity_list.index(max(the_similarity_list)[0])]
        print(the_max_value)
        print(parameter_list)
        
    


# This parameter list shows the coefficients of the equation that ==> the_total_score['score_diff'] * random_num_list[0] + the_total_score['WScore'] * random_num_list[1]+ the_total_score['WFGM']* random_num_list[2] + the_total_score['WAst'] * random_num_list[3] + the_total_score['WDR'] * random_num_list[4] + the_total_score['WBlk'] * random_num_list[5] + the_total_score['WFGM3'] * random_num_list[6] + the_total_score['WFGA3'] * random_num_list[7] + the_total_score['WFTA'] * random_num_list[8] + the_total_score['WOR'] * random_num_list[9] + the_total_score['WFTM'] * random_num_list[10]

# I took the parameters from table that includes league statistics and final win rate

# You can change the parameters of equation that can increase the correlation factor. Good luck

# In[ ]:





# In[ ]:




