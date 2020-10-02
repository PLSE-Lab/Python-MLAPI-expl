#!/usr/bin/env python
# coding: utf-8

# # Exploration of How  Winning Metrics and Social Media Predict Salary

# ## Objective
# 
# In this notebook, we explore how winning metrics of NBA players in year 2016-2017 and social media data can be used to predict their salary.
# 
# We first normalized all variables, and then built multiple models to predict salary. Based on the means of variable coefficients , we selected the three variables that have biggest prediction effect on salary. Because salary is based upon a lot of factors other than historical performance, this notebook is not aimed at predicting reasonable salary for purchasing players, rather, it's suggesting top factors to take into consideration.

# In[ ]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
color = sns.color_palette()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing


# In[ ]:


attendance_valuation_elo_df = pd.read_csv("../input/nba_2017_att_val_elo.csv");attendance_valuation_elo_df.head()


# In[ ]:


salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()


# In[ ]:


pie_df = pd.read_csv("../input/nba_2017_pie.csv");pie_df.head()


# In[ ]:


plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");plus_minus_df.head()


# In[ ]:


br_stats_df = pd.read_csv("../input/nba_2017_br.csv");br_stats_df.head()


# In[ ]:



plus_minus_df.rename(columns={"NAME":"PLAYER", "WINS": "WINS_RPM"}, inplace=True)
players = []
for player in plus_minus_df["PLAYER"]:
    plyr, _ = player.split(",")
    players.append(plyr)
plus_minus_df.drop(["PLAYER"], inplace=True, axis=1)
plus_minus_df["PLAYER"] = players
plus_minus_df.head()


# In[ ]:



nba_players_df = br_stats_df.copy()
nba_players_df.rename(columns={'Player': 'PLAYER','Pos':'POSITION', 'Tm': "TEAM", 'Age': 'AGE', "PS/G": "POINTS"}, inplace=True)
nba_players_df.drop(["G", "GS", "TEAM"], inplace=True, axis=1)
nba_players_df = nba_players_df.merge(plus_minus_df, how="inner", on="PLAYER")
nba_players_df.head()


# In[ ]:



pie_df_subset = pie_df[["PLAYER", "PIE", "PACE", "W"]].copy()
nba_players_df = nba_players_df.merge(pie_df_subset, how="inner", on="PLAYER")
nba_players_df.head()


# In[ ]:


salary_df.rename(columns={'NAME': 'PLAYER'}, inplace=True)
salary_df["SALARY_MILLIONS"] = round(salary_df["SALARY"]/1000000, 2)
salary_df.drop(["POSITION","TEAM", "SALARY"], inplace=True, axis=1)
salary_df.head()


# In[ ]:


nba_players_with_salary_df = nba_players_df.merge(salary_df); 


# In[ ]:


# import wiki data
wiki_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv");wiki_df.head()
wiki_df.rename(columns={'names': 'PLAYER', "pageviews": "PAGEVIEWS"}, inplace=True)


# In[ ]:


median_wiki_df = wiki_df.groupby("PLAYER").median()
median_wiki_df_small = median_wiki_df[["PAGEVIEWS"]]
median_wiki_df_small = median_wiki_df_small.reset_index()
# merge with wikipageviews
nba_players_with_salary_wiki_df = nba_players_with_salary_df.merge(median_wiki_df_small)
nba_players_with_salary_wiki_df.head()


# In[ ]:


# import twitter data
twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twitter_df.head()


# In[ ]:


# merge with twitter data
nba_players_with_salary_wiki_twitter_df = nba_players_with_salary_wiki_df.merge(twitter_df)
nba_players_with_salary_wiki_twitter_df.head()


# In[ ]:


# drop strings
nba_players_with_salary_wiki_twitter_df.drop(['Rk','PLAYER','POSITION','TEAM'],inplace=True, axis=1)
nba_players_with_salary_wiki_twitter_df.head()


# In[ ]:


# exam missing values
nba_players_with_salary_wiki_twitter_df.describe()


# In[ ]:


# drop columns with missing values
nba_players_with_salary_wiki_twitter_df.dropna(axis=0,inplace=True)
nba_players_with_salary_wiki_twitter_df.head()


# In[ ]:


# define training features and training labels
train_labels = nba_players_with_salary_wiki_twitter_df['SALARY_MILLIONS']
train_features = nba_players_with_salary_wiki_twitter_df.drop(['SALARY_MILLIONS'], axis=1)
# Store the column/feature names into a list "colnames"
colnames = train_features.columns


# In[ ]:


# Normalize features
train_features = pd.DataFrame(preprocessing.normalize(train_features,axis=0),columns= colnames)
train_features.head()


# In[ ]:


# exam shapes
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
#print('Testing Features Shape:', test_features.shape)
#print('Testing Labels Shape:', test_labels.shape)


# ## Model Training

# In[ ]:


# Define dictionary to store our rankings
ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


# ##  Linear Model Feature Ranking

# In[ ]:


# Using Linear Regression
lr = LinearRegression(normalize=True)
lr.fit(train_features, train_labels)
ranks["LinReg"] = ranking(np.abs(lr.coef_), colnames)

# Using Ridge 
ridge = Ridge(alpha = 7)
ridge.fit(train_features, train_labels)
ranks['Ridge'] = ranking(np.abs(ridge.coef_), colnames)

# Using Lasso
lasso = Lasso(alpha=.05)
lasso.fit(train_features, train_labels)
ranks["Lasso"] = ranking(np.abs(lasso.coef_), colnames)


# In[ ]:


# random forest feature ranking
rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
rf.fit(train_features, train_labels)
ranks["RF"] = ranking(rf.feature_importances_, colnames);


# In[ ]:


# Create empty dictionary to store the mean value calculated from all the scores
r = {}
for name in colnames:
    r[name] = round(np.mean([ranks[method][name] 
                             for method in ranks.keys()]), 2)
 
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
 
print("\t%s" % "\t".join(methods))
for name in colnames:
    print("%s\t%s" % (name, "\t".join(map(str, 
                         [ranks[method][name] for method in methods]))))


# In[ ]:


# Put the mean scores into a Pandas dataframe
meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])

# Sort the dataframe
meanplot = meanplot.sort_values('Mean Ranking', ascending=False)


# In[ ]:


# Let's plot the ranking of the features
sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar", 
               size=14, aspect=1.9, palette='coolwarm')

