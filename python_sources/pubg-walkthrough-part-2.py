#!/usr/bin/env python
# coding: utf-8

# Now we are on part 2. Read [Part 1](https://www.kaggle.com/mjenkins1/pubg-walkthrough) in case you missed it where I explore some individual features. This part will focus on exploring some features as a group rather than as an individual. Finally Part 3 will involve doing the actual modeling and prediction. 

# In[ ]:


# Import necessary libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import warnings

warnings.filterwarnings("ignore")


# In[ ]:


# For plot sizes
plt.rcParams["figure.figsize"] = (18,8)
sns.set(rc={'figure.figsize':(18,8)})


# In[ ]:


os.listdir('../input')


# In[ ]:


# Load Part 1 data
data = pd.read_csv('../input/Training_Data_New.csv')
print("Done loading data from part 1")


# In[ ]:


# Let's review the features
data.columns


# Since the goal of this notebook is to explore and create group features, let's take a look at the matchType feature

# In[ ]:


data['matchType'].value_counts().plot(kind='bar');


# Top 6 matchType are significatly more represented in this dataset than the other modes. Therefore, we want to be careful of marking sure our model doesn't get biased due to unbalanced data.
# 
# Now it is time to group teams together

# In[ ]:


df_groups = (data.groupby('groupId', as_index=False).agg({'Id':'count', 'matchId':'count', 'assists':'sum', 'boosts':'sum',
                                'damageDealt':['sum', 'mean', 'max', 'min'], 'DBNOs':'sum', 'headshotKills':'sum',
                                'heals':['sum', 'mean'], 'killPlace':['mean', 'max', 'min'], 'killPoints':['mean', 'max', 'min'],
                                'kills':['sum', 'mean', 'max', 'min'],
                                'killStreaks':'mean', 'longestKill':'mean', 'matchDuration':['mean', 'min', 'max', 'sum'],
                                'maxPlace':['mean', 'min', 'max'], 'numGroups':['count','sum', 'mean', 'max', 'min'],
                                'revives':'sum', 'rideDistance':'max', 'roadKills':'sum', 'swimDistance':'max',
                                'teamKills':['sum', 'mean', 'max', 'min'], 'vehicleDestroys':'sum', 'walkDistance':['sum', 'mean', 'max', 'min'],
                                'weaponsAcquired':'sum','winPoints':['sum', 'mean', 'max', 'min'], 'winPlacePerc':'mean',
                                'killsPerMeter': 'mean', 'healsPerMeter': 'mean', 'killsPerHeal': 'mean',
                                'killsPerSecond': 'max', 'TotalHealsPerTotalDistance': 'max',
                                'killPlacePerMaxPlace': 'max'}).rename(columns={'Id':'teamSize'}).reset_index())


# In[ ]:


# Show changes
df_groups.head(5)


# In[ ]:


df_groups['teamSize'].describe()


# In[ ]:


df_groups['teamSize']['count'].value_counts()


# I find it strange that there is some matches with more than 4 people in a group. Therefore I want to see what type of matchType it is for a group of 74 

# In[ ]:


df_groups[df_groups['teamSize']['count'] == 74]


# # Some visualizations

# In[ ]:


df_groups.columns


# In[ ]:


sns.distplot(df_groups['kills']['mean']);


# In[ ]:


sns.distplot(df_groups['weaponsAcquired']['sum'], color='red');


# In[ ]:


sns.distplot(df_groups['damageDealt']['mean'], color='purple');


# In[ ]:


sns.distplot(df_groups['damageDealt']['sum'], color='purple');


# In[ ]:


sns.jointplot(df_groups['teamSize']['count'], df_groups['winPlacePerc']['mean'], height = 12, ratio = 4, color='darkorange');


# In[ ]:


sns.jointplot(df_groups['winPoints']['mean'], df_groups['winPlacePerc']['mean'], height = 12, ratio = 4, color='mediumseagreen');


# In[ ]:


sns.jointplot(df_groups['killPoints']['mean'], df_groups['winPlacePerc']['mean'], height = 12, ratio = 4, color='darkblue');


# In[ ]:


sns.jointplot(df_groups['killPoints']['max'], df_groups['winPlacePerc']['mean'], height = 12, ratio = 4, color='darkred');


# In[ ]:


# Save new grouped data
df_groups.to_csv(r'Training_Data_New_Groups.csv')


# In[ ]:




