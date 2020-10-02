#!/usr/bin/env python
# coding: utf-8

# # Australia Tour of India 2019 ODI Predictions

# > In this notebook we will attempt to predict the outcomes of the Australia tour of India 2019 ODI matches. First we will try to predict the match by match outcome. Then we shall try to find the players with most runs scored , highest sixes and highest fours. Then we shall try to find the highest wicket taker.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/cricinfo-statsguru-data/"))

# Any results you write to the current directory are saved as output.


# > Reading the dataset for prediction of matches

# In[ ]:


data = pd.read_csv('../input/odi-cricket-matches-19712017/ContinousDataset.csv')

display(data.head())


# In[ ]:


#Filter dataset for india and australia

dataset_filtered = data[ (data['Team 1'].isin(['Australia','India'])) & (data['Team 2'].isin(['Australia','India']))]


# Converting categorical values to encoded values
x = dataset_filtered.drop(['Winner','Unnamed: 0','Scorecard','Match Date','Host_Country'],axis =1)
y = dataset_filtered['Winner']
x_filtered = pd.get_dummies(x,drop_first = True)
y_filtered = pd.get_dummies(y,drop_first = True)


# ## Match Predictions

# In[ ]:



# Splitting data into training and test set
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_filtered,y_filtered,test_size = 0.3,random_state = 0)


# In[ ]:


display(x_train.head())
display(y_train.head())


# In[ ]:


# Model building
from xgboost import XGBClassifier

classifier = XGBClassifier(learning_rate=0.25,n_estimators=500,objective='binary:logistic')

classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)



# In[ ]:


# Model accuracy
from sklearn.metrics import roc_auc_score

display(roc_auc_score(y_test,y_pred))

display(x_filtered.head())


# In[ ]:


# Filtering data for current venue

x_current = x_filtered[(x_filtered['Team 1_India']==1) & (x_filtered['Venue_Team1_Home']==1) &
         (x_filtered['Ground_Visakhapatnam']==1)]

display('Visag',classifier.predict(x_current).mean())

#  (x_filtered['Ground_Visakhapatnam']==1) | (x_filtered['Ground_Bengaluru']==1) |
#           (x_filtered['Ground_Hyderabad']==1) | (x_filtered['Ground_Nagpur']==1) |
#           (x_filtered['Ground_New Delhi']==1)]


x_current = x_filtered[(x_filtered['Team 1_India']==1) & (x_filtered['Venue_Team1_Home']==1) &
         (x_filtered['Ground_Bengaluru']==1)]

display('Bangalore',classifier.predict(x_current).mean())

x_current = x_filtered[(x_filtered['Team 1_India']==1) & (x_filtered['Venue_Team1_Home']==1) &
         (x_filtered['Ground_Hyderabad']==1)]

display('Hydrabad',classifier.predict(x_current).mean())

x_current = x_filtered[(x_filtered['Team 1_India']==1) & (x_filtered['Venue_Team1_Home']==1) &
         (x_filtered['Ground_Nagpur']==1)]

display('Nagpur',classifier.predict(x_current).mean())

x_current = x_filtered[(x_filtered['Team 1_India']==1) & (x_filtered['Venue_Team1_Home']==1) &
         (x_filtered['Ground_New Delhi']==1) | (x_filtered['Ground_Delhi']==1)]

display('New Delhi',classifier.predict(x_current).mean())




# In[ ]:


#There is 53% chance that india will win the competition
# The score will be 3/2
# Data for Ranchi venue is not found


# ## Top Scorer Predictions

# In[ ]:


display(x_filtered.head())


# In[ ]:


# Player info
data_player = pd.read_csv('../input/cricinfo-statsguru-data/ODIs - Batting.csv')
data_player_filtered = data_player[(data_player['Country'] == 'Australia') | (data_player['Country'] == 'India')]


# In[ ]:


# Analysing data for the indian team

data_player_filtered = data_player_filtered.replace('-', data_player_filtered.replace(['-'], [None]))
data_player_filtered = data_player_filtered.dropna()

playing_team = data_player_filtered[data_player_filtered['Career End'] == 2018]

playing_team_ind = playing_team[playing_team['Player'].str.contains('Kohli|Sharma|Dhawan|Rayudu|Rahul|Pant|Dhoni|Yadav|Shankar|Kumar|Siraj|Chahal|Jadeja|Bumrah')]
data_ind = playing_team_ind.iloc[:,[2,3,4,5,6,7,9,10,11,12,13,14,15]].astype(np.float)
data_ind['Player'] = playing_team_ind['Player']
data_ind['Player capacity'] = (data_ind['Batting Strike Rate'] * data_ind['Batting Avg']/100.)
data_ind['Player intensity'] = (2 * data_ind['Hundreds Scored'] + data_ind['Scores Of Fifty Or More']) + 0.1
data_ind['Player trait'] =  data_ind['Player intensity']*data_ind['Player capacity'] 
data_ind = data_ind.sort_values(by = ['Player trait'],ascending = [0])

display(data_ind)


# In[ ]:


import seaborn as sns

sns.scatterplot(y = 'Player capacity',x = 'Player intensity',data = data_ind)



# In[ ]:


# Analysing data for the Australian Team

data_player_filtered = data_player_filtered.replace('-', data_player_filtered.replace(['-'], [None]))
data_player_filtered = data_player_filtered.dropna()

playing_team_aus = data_player_filtered[data_player_filtered['Career End'] == 2018]

playing_team_aus = playing_team_aus[playing_team['Player'].str.contains('Paine|Marsh|Hazelwood|Cummins|Finch|Hadscomb|Harris|Head|Khawaja|Lyon|Siddle|Starc|Tremain')]
data_aus = playing_team_aus.iloc[:,[2,3,4,5,6,7,9,10,11,12,13,14,15]].astype(np.float)
data_aus['Player'] = playing_team['Player']
data_aus['Player capacity'] = (data_aus['Batting Strike Rate'] * data_aus['Batting Avg']/100.)
data_aus['Player intensity'] = (2 * data_aus['Hundreds Scored'] + data_aus['Scores Of Fifty Or More']) + 0.1
data_aus['Player trait'] =  data_aus['Player intensity']*data_aus['Player capacity'] 
data_aus = data_aus.sort_values(by = ['Player trait'],ascending = [0])

display(data_aus)


# In[ ]:


sns.scatterplot(y = 'Player capacity',x = 'Player intensity',data = data_aus)


# > Comparing the above two results we find that **Virat Kohli** has the highest chance of being the top scorer as he has the highest **trait** parameter

# In[ ]:


data_ind['Country'] = pd.Series('India' for x in range(len(data_ind))).values
data_aus['Country'] = pd.Series('Australia' for x in range(len(data_aus))).values
dataset_combined = pd.concat([data_ind,data_aus])
display(dataset_combined.sort_values(by = ['Player trait'],ascending = [0]))

sns.scatterplot(y = 'Player capacity',x = 'Player intensity',data = dataset_combined,hue = 'Country')


# > The above table also ranks the players based on their expected performance or the highest scorer of the match.
# >  The above scatter plot compares both the teams. We can easily predict that the indian players are far ahead of their australian counterparts.

# ## Highest Fours and Sixes

# >  Ranking player based on strike rate this will provide us with the highest number of fours and sixes

# In[ ]:


display(dataset_combined.sort_values(by = ['Batting Strike Rate'],ascending = [0]))


# > Based on the strike rate we find credible evidence that a player with higher strike rate will hit more sixes and fours. Thus the above table ranks the highest number of sixes or fours scorers

# ## Bowling predictions

# In[ ]:


data_bowling = pd.read_csv("../input/cricinfo-statsguru-data/ODIs - Bowling.csv")
display(data_bowling.head())


# In[ ]:


data_bowling_sorted = data_bowling[data_bowling['Player'].str.contains('Paine|Marsh|Hazelwood|Cummins|Finch|Hadscomb|Harris|Head|Khawaja|Lyon|Siddle|Starc|Tremain|Kohli|Sharma|Dhawan|Rayudu|Rahul|Pant|Dhoni|Yadav|Shankar|Kumar|Siraj|Chahal|Jadeja|Bumrah')]
data_bowling_sorted = data_bowling_sorted[['Player','Bowling Strike Rate']]
data_bowling_sorted = data_bowling_sorted.replace('-', 0.0)
data_bowling_sorted['Bowling Strike Rate'] = data_bowling_sorted['Bowling Strike Rate'].astype(np.float)
data_bowling_sorted = data_bowling_sorted[(data_bowling_sorted['Bowling Strike Rate'] != 0) & (data_bowling_sorted['Player'].str.contains('-2018'))]
display(data_bowling_sorted.sort_values(by = ['Bowling Strike Rate'],ascending = [1]))


# >  Based on the bowling strike rate we can predict the highest wicket takers of the match. The above table represents the same
