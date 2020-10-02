#!/usr/bin/env python
# coding: utf-8

# In[51]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Now we want to read in the data and see the heads of each of the data sets

# In[52]:


data_dir = '../input/'
df_seeds = pd.read_csv(data_dir + 'datafiles/NCAATourneySeeds.csv')
df_tour = pd.read_csv(data_dir + 'datafiles/NCAATourneyCompactResults.csv')

df_teams = pd.read_csv(data_dir + 'datafiles/Teams.csv')
df_massey = pd.read_csv(data_dir + 'masseyordinals_thru_2019_day_128/MasseyOrdinals_thru_2019_day_128.csv')

# filter massey according to last day of season
# get the most predictive rankings
df_massey = df_massey[df_massey['RankingDayNum'] == 128]
df_massey = df_massey[df_massey['SystemName'].isin(['POM', 'SAG', 'TRP', 'TRK', 'DOK'])]


# In[ ]:


df_seeds.head()


# In[ ]:


df_tour.head()


# In[ ]:


df_teams.head()


# In[ ]:


df_massey.head()


# Replace the seed with its cooresponding int

# In[ ]:


def seed_to_int(seed):
    # Get just the digits from the seeding. Return as int
    s_int = int(seed[1:3])
    return s_int

df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(labels=['Seed'], inplace=True, axis=1)  # This is the string label
df_seeds.head()


# Split into two data frames, the wins and then the losing seeds

# In[ ]:


df_tour.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
df_tour.head()


# In[ ]:


#  summarizes wins & losses along with their corresponding seed differences
df_winseeds = df_seeds.rename(columns={'TeamID': 'WTeamID', 'seed_int': 'WSeed'})
df_lossseeds = df_seeds.rename(columns={'TeamID': 'LTeamID', 'seed_int': 'LSeed'})

df_winseeds_mass = df_massey.rename(columns={'TeamID': 'WTeamID'})
df_lossseeds_mass = df_massey.rename(columns={'TeamID': 'LTeamID'})


# In[ ]:


df_lossseeds.head()


# In[ ]:


df_winseeds.head()


# Split the massey ordinals into winning id and losing id

# In[ ]:


df_winseeds_mass = df_massey.rename(columns={'TeamID': 'WTeamID'})
df_lossseeds_mass = df_massey.rename(columns={'TeamID': 'LTeamID'})


# Now, we want to aggregate all the ordinals associated with a winning team and a losing team

# In[ ]:


# Merge POM
df_temp = df_winseeds_mass[(df_winseeds_mass.SystemName == 'POM')]
df_concat = pd.merge(df_tour, df_temp[['Season', 'WTeamID', 'OrdinalRank']], how='left', on=['Season', 'WTeamID'])

df_temp = df_lossseeds_mass[(df_lossseeds_mass.SystemName == 'POM')]
df_concat = pd.merge(df_concat, df_temp[['Season', 'LTeamID', 'OrdinalRank']], on=['Season', 'LTeamID'])

df_concat = df_concat.rename(columns={'OrdinalRank_x': 'POM_W', 'OrdinalRank_y': 'POM_L'})

# Merge SAG
df_temp = df_winseeds_mass[(df_winseeds_mass.SystemName == 'SAG')]
df_concat = pd.merge(df_concat, df_temp[['Season', 'WTeamID', 'OrdinalRank']], how='left', on=['Season', 'WTeamID'])

df_temp = df_lossseeds_mass[(df_lossseeds_mass.SystemName == 'SAG')]
df_concat = pd.merge(df_concat, df_temp[['Season', 'LTeamID', 'OrdinalRank']], on=['Season', 'LTeamID'])

df_concat = df_concat.rename(columns={'OrdinalRank_x': 'SAG_W', 'OrdinalRank_y': 'SAG_L'})

# Merge TRP
df_temp = df_winseeds_mass[(df_winseeds_mass.SystemName == 'TRP')]
df_concat = pd.merge(df_concat, df_temp[['Season', 'WTeamID', 'OrdinalRank']], how='left', on=['Season', 'WTeamID'])

df_temp = df_lossseeds_mass[(df_lossseeds_mass.SystemName == 'TRP')]
df_concat = pd.merge(df_concat, df_temp[['Season', 'LTeamID', 'OrdinalRank']], on=['Season', 'LTeamID'])

df_concat = df_concat.rename(columns={'OrdinalRank_x': 'TRP_W', 'OrdinalRank_y': 'TRP_L'})

# Merge TRK
df_temp = df_winseeds_mass[(df_winseeds_mass.SystemName == 'TRK')]
df_concat = pd.merge(df_concat, df_temp[['Season', 'WTeamID', 'OrdinalRank']], how='left', on=['Season', 'WTeamID'])

df_temp = df_lossseeds_mass[(df_lossseeds_mass.SystemName == 'TRK')]
df_concat = pd.merge(df_concat, df_temp[['Season', 'LTeamID', 'OrdinalRank']], on=['Season', 'LTeamID'])

df_concat = df_concat.rename(columns={'OrdinalRank_x': 'TRK_W', 'OrdinalRank_y': 'TRK_L'})

# Merge DOK
df_temp = df_winseeds_mass[(df_winseeds_mass.SystemName == 'DOK')]
df_concat = pd.merge(df_concat, df_temp[['Season', 'WTeamID', 'OrdinalRank']], how='left', on=['Season', 'WTeamID'])

df_temp = df_lossseeds_mass[(df_lossseeds_mass.SystemName == 'DOK')]
df_concat = pd.merge(df_concat, df_temp[['Season', 'LTeamID', 'OrdinalRank']], on=['Season', 'LTeamID'])

df_concat = df_concat.rename(columns={'OrdinalRank_x': 'DOK_W', 'OrdinalRank_y': 'DOK_L'})

df_concat = pd.merge(left=df_concat, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_concat, right=df_lossseeds, on=['Season', 'LTeamID'])

df_concat['SeedDiff'] = df_concat.WSeed - df_concat.LSeed
df_concat.head(10)


# We now want to create our training set

# In[ ]:


df_wins = pd.DataFrame()
df_wins['POM_W'] = df_concat['POM_W']
df_wins['POM_L'] = df_concat['POM_L']
df_wins['SAG_W'] = df_concat['SAG_W']
df_wins['SAG_L'] = df_concat['SAG_L']
df_wins['TRK_W'] = df_concat['TRK_W']
df_wins['TRK_L'] = df_concat['TRK_L']
df_wins['TRP_W'] = df_concat['TRP_W']
df_wins['TRP_L'] = df_concat['TRP_L']
df_wins['DOK_W'] = df_concat['DOK_W']
df_wins['DOK_L'] = df_concat['DOK_L']

df_wins['Result'] = 1

df_losses = pd.DataFrame()
df_losses['POM_W'] = df_concat['POM_L']
df_losses['POM_L'] = df_concat['POM_W']
df_losses['SAG_W'] = df_concat['SAG_L']
df_losses['SAG_L'] = df_concat['SAG_W']
df_losses['TRK_W'] = df_concat['TRK_L']
df_losses['TRK_L'] = df_concat['TRK_W']
df_losses['TRP_W'] = df_concat['TRP_W']
df_losses['TRP_L'] = df_concat['TRP_L']
df_losses['DOK_W'] = df_concat['DOK_W']
df_losses['DOK_L'] = df_concat['DOK_L']
df_losses['Result'] = 0

df_predictions = pd.concat((df_wins, df_losses))


# In[ ]:


df_predictions.head(10)


# In[ ]:


df_predictions.tail(10)


# Train the logistic regression model

# In[ ]:


X_train = df_predictions[['POM_W', 'POM_L', 'SAG_W', 'SAG_L', 'TRK_W', 'TRK_L', 'TRP_W', 'TRP_L', 'DOK_W', 'DOK_L']].values.reshape(-1, 10)
y_train = df_predictions.Result.values # train according to a 0 or 1 -> 1 winning and 0 losing
X_train, y_train = shuffle(X_train, y_train) # shuffle the training data

logreg = LogisticRegression()

# use grid serach to identify the params for the regularization of paramaters
# use log loss to score the grid search because we are using logistic regression to evaluate a binary outcome
params = {'C': np.logspace(start=-5, stop=3, num=9)}
clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)

# train the funcation
clf.fit(X_train, y_train)
print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))


# Now, lets see the results

# > > 
