#!/usr/bin/env python
# coding: utf-8

# # PUBG Finish Placement Prediction
# 
# ## Content
# - Overview the data
#     - Load the data
#     - Columns dtypes
#     
#     
# - EDA
#     - Distributions and Box plots
#     - Correlations
#     - Other analysis
#         
# 
# - Feature engineering
#     - Count players in each game
#     - Normalize attributes
#     - Sum boosts and distance
#     - Boosts per distance
#     - Kills per distance
#     - Teams
#     

# In[ ]:


# Data and arrays handling
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Interactive plotting
from plotly.offline import init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go

init_notebook_mode(connected=True)

# Inline plots
get_ipython().run_line_magic('matplotlib', 'inline')

# Ignore warns
import warnings
warnings.filterwarnings('ignore')


# ## 1. Overview the data
# ### 1.1 Load data

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain = pd.read_csv('../input/train.csv')\ntest = pd.read_csv('../input/test.csv')")


# ### 1.2 Columns dtypes

# In[ ]:


train.head(3).T


# In[ ]:


train.info()


# ## 2. EDA
# So, our target column is `winPlacePerc`. Let's go through all the columns and try to find out any interconnections.
# 
# ### 2.1 Distributions and Box plots
# In this step we will plot the distribution of each attribute. After that we will make some box plots to see the impact of different features on winPlacePerc - our target.

# #### Assists
# *Description:* Number of enemy players this player damaged that were killed by teammates.  
# 
# Let's look though assist distribution

# In[ ]:


plt.figure(figsize=(8,6))
sns.distplot(train['assists'], kde=False)
plt.show()


# Now we will try to find out difference between people end with 0 assist and people end with 1 or more

# In[ ]:


assist_df = train[['assists', 'winPlacePerc']]
assist_df['assists'] = assist_df['assists'].apply(lambda x: 'zero' if x == 0 else '1 or more')


# In[ ]:


assist_df['assists'].value_counts(normalize=True)


# In[ ]:


plt.figure(figsize=(8,7))
sns.boxplot(x='assists', y='winPlacePerc', data=assist_df)
plt.title('Zero assists vs. one or more assists')
plt.show()


# We can see the pattern!

# #### Boosts
# *Description:* Number of boost items used.

# In[ ]:


plt.figure(figsize=(8,6))
sns.distplot(train['boosts'], kde=False)
plt.show()


# In[ ]:


boosts_df = train[['boosts', 'winPlacePerc']]
boosts_df['boosts'] = boosts_df['boosts'].apply(lambda x: 'zero' if x == 0 else '1 or more')
boosts_df['boosts'].value_counts()


# In[ ]:


plt.figure(figsize=(8,7))
sns.boxplot(x='boosts', y='winPlacePerc', data=boosts_df, order=['zero', '1 or more'])
plt.show()


# So boosted players are always winning!

# #### Another attributes
# 
# Let's deal with other attrbutes the same way.

# In[ ]:


attrs_for_boxplots = ['damageDealt', 'DBNOs',
                       'headshotKills', 'heals', 'kills',
                       'killStreaks', 'longestKill', 'revives',
                       'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
                       'vehicleDestroys', 'walkDistance', 'weaponsAcquired']

other_attr = ['killPlace', 'killPoints', 'maxPlace', 'winPoints', 'numGroups']

target = 'winPlacePerc'


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Number of columns in our big picture\ncolumns = 2\n\n# Number of rows\nrows = len(attrs_for_boxplots)\n\n# Position index\nplot_position_in_grid = 1\n\n# Iterate through all attributes\nfor attr in attrs_for_boxplots:\n    \n    # Set figure size\n    plt.figure(figsize=(12, 4 * rows))\n    \n    # fix the subplot position\n    # plot the distribution\n    plt.subplot(rows, columns, plot_position_in_grid)\n    sns.distplot(train[attr], kde=False)\n    \n    # Create compare df\n    temp_df = train[[attr, target]]\n    temp_df[attr] = temp_df[attr].apply(lambda x: 'zero' if x == 0 else 'more')\n\n    # fix the subplot position\n    # plot the boxplot\n    plt.subplot(rows, columns, plot_position_in_grid+1)\n    sns.boxplot(x=attr, y=target, data=temp_df, order=['zero', 'more'])\n\n    plot_position_in_grid += 2")


# ### 2.2 Correlations
# 
# #### Correlation matrix

# In[ ]:


get_ipython().run_cell_magic('time', '', "corr_matrix = train.corr()\n\nplt.figure(figsize=(24,23))\nsns.set(font_scale=1.3)\nsns.heatmap(corr_matrix, annot=True, fmt='.1f')\nplt.show()")


# #### Correlations with target

# In[ ]:


corr_with_target = train.drop(['Id', 'matchId', 'groupId', 'winPlacePerc'], 1).corrwith(train['winPlacePerc'])


# In[ ]:


# Set our dots
trace0 = go.Scatter(
    x = corr_with_target.index,
    y = corr_with_target.values,
    name = 'corrs',
    mode='markers',
    marker = {
        'size' : 20,
        'color' : corr_with_target.values,
        'colorscale' : 'Jet',
        'showscale' : True,
        'symbol' : 202,
        'opacity' : .76
    }
)

# Create data array and layout
data = [trace0]
layout = {'title': 'Correlation between winPlacePerc and other attributes',
          'yaxis' : {'title' : 'winPlacePerc'},
          'xaxis' : {'tickangle' : 45}}

# Display it
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)


# #### Correlations within attributes (no target)
# 
# As we can see at heatmap - there is lot of attributes higly correlated with each-other. Let's find out who are they.

# - killPlace vs damageDealt

# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(train['killPlace'], train['damageDealt'], alpha=.8, c='orange')
plt.xlabel('killPlace')
plt.ylabel('damageDealth')
plt.title('killPlace vs damageDealt scatter plot')
plt.show()


# - DBNOs vs damageDealth  
# It's obvious why them are correlated. Maybe it will be better to keep only one of this attributes. So we will see it later

# - Heals vs boosts

# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(train['heals'], train['boosts'], alpha=.8, c='green')
plt.xlabel('heals')
plt.ylabel('boosts')
plt.title('Heals vs boots scatter plot')
plt.show()


# #### Loop it
# We can set the threshold for significant correlation and create scatter plots in loop

# In[ ]:


def get_subplot_row_columns(df, corr_threshold):
    corr = df.corr()
    n = corr[(corr > corr_threshold) & (corr != 1)].dropna(axis=0, how='all')                                            .dropna(axis=1, how='all')                                            .notnull().sum().sum()
    cols = 3

    return cols, int(np.ceil(n / cols))


# In[ ]:


get_ipython().run_cell_magic('time', '', "# It takes some time to display all graphs\n\n# Feel free to play with this value\nthreshold = .5\n\n# All attributes\nattrs = train.columns.values\n\n# Array for pairs already checked\nseen_pairs = []\n\n# Params for subplotting\nsubplot_number = 0\ncols, rows = get_subplot_row_columns(train, threshold)\n\nfig, axs = plt.subplots(rows, cols, figsize=(18, 5 * rows))\nplt.subplots_adjust(wspace=0.45, hspace=0.35)\naxs = axs.ravel()\n\nprint(f'{rows} x {cols}')\n\nfor first_attr in attrs:\n    for second_attr in attrs:       \n        # Skip same\n        if first_attr == second_attr:\n            continue\n        \n        # Skip swap attributes\n        if (first_attr, second_attr) in seen_pairs:\n            continue\n        else:\n            seen_pairs.append((second_attr, first_attr))\n        \n        # Check threshold\n        current_corr = train[[first_attr]].corrwith(train[second_attr])[first_attr]\n        \n        if current_corr > threshold:  \n            print(f'{subplot_number} - {first_attr} x {second_attr}')\n            axs[subplot_number].set_title(f'{first_attr} vs {second_attr}\\n(correlation = {current_corr})')\n            axs[subplot_number].scatter(train[first_attr], train[second_attr], alpha=.8, c='green')\n            axs[subplot_number].set_xlabel(first_attr)\n            axs[subplot_number].set_ylabel(second_attr)\n            \n            \n            subplot_number += 1\n\n            \nfor i in range(subplot_number, cols*rows):\n    fig.delaxes(axs[i])\n    \nprint(f'{subplot_number} graphs plotted')")


# #### 2.3 Other analysis
# Thanks Dimitrios Effrosynidis for his great kernel. I take some analysis and feature engineering from his work.  
# 
# Check it out: [Eda is fun!](https://www.kaggle.com/deffro/eda-is-fun)

# #### Vechicle destroy vs winPlacePerc

# In[ ]:


plt.figure(figsize=(8,6))
sns.lineplot(x='vehicleDestroys', y='winPlacePerc',
             data=train)
plt.title('vehicleDestroys impact on winPlacePerc')


# #### Team play

# In[ ]:


solos = train[train['numGroups']>50]
duos = train[(train['numGroups']>25) & (train['numGroups']<=50)]
squads = train[train['numGroups']<=25]
games_count = train.shape[0]


game_types_df = pd.DataFrame({'Games count' : [solos.shape[0],
                                         duos.shape[0],
                                         squads.shape[0]],
                              'Normalized' : [solos.shape[0] / games_count,
                                         duos.shape[0] / games_count,
                                         squads.shape[0] / games_count]},
                              index=['solos', 'duos', 'squads'])


# In[ ]:


game_types_df


# #### Team kills

# In[ ]:


plt.figure(figsize=(18,8))

sns.pointplot(x='kills', y='winPlacePerc', data=solos, color='green')
sns.pointplot(x='kills', y='winPlacePerc', data=duos, color='red')
sns.pointplot(x='kills', y='winPlacePerc', data=squads, color='black')

plt.text(14,0.5,'Solos',color='green',fontsize = 17,style = 'italic')
plt.text(14,0.45,'Duos',color='red',fontsize = 17,style = 'italic')
plt.text(14,0.40,'Squads',color='black',fontsize = 17,style = 'italic')

plt.title('Kills in different play modes impact pn winPlacePerc')

plt.grid()


# ## 3. Feature Engineering
# Also from Dimitrios Effrosynidis [kernel](https://www.kaggle.com/deffro/eda-is-fun).

# ### 3.1 Count players in each game

# In[ ]:


train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')


# In[ ]:


data = train.copy()
data = data[data['playersJoined']>49]
plt.figure(figsize=(18,9))
sns.countplot(data['playersJoined'])
plt.title("Players Joined",fontsize=15)
plt.show()


# ### 3.2 Normalize attributes

# In[ ]:


train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)

train[['playersJoined', 'kills', 'killsNorm', 'damageDealt', 'damageDealtNorm']].head(4)


# ### 3.3 Sum boosts and distance

# In[ ]:


train['healsAndBoosts'] = train['heals']+train['boosts']
train['totalDistance'] = train['walkDistance']+train['rideDistance']+train['swimDistance']


# ### 3.4 Boosts per distance

# In[ ]:


train['boostsPerWalkDistance'] = train['boosts']/(train['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where boosts>0 and walkDistance=0. Strange.
train['boostsPerWalkDistance'].fillna(0, inplace=True)
train['healsPerWalkDistance'] = train['heals']/(train['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where heals>0 and walkDistance=0. Strange.
train['healsPerWalkDistance'].fillna(0, inplace=True)
train['healsAndBoostsPerWalkDistance'] = train['healsAndBoosts']/(train['walkDistance']+1) #The +1 is to avoid infinity.
train['healsAndBoostsPerWalkDistance'].fillna(0, inplace=True)
train[['walkDistance', 'boosts', 'boostsPerWalkDistance' ,'heals',  'healsPerWalkDistance', 'healsAndBoosts', 'healsAndBoostsPerWalkDistance']][40:45]


# ### 3.5 Kills per distance

# In[ ]:


train['killsPerWalkDistance'] = train['kills']/(train['walkDistance']+1) #The +1 is to avoid infinity, because there are entries where kills>0 and walkDistance=0. Strange.
train['killsPerWalkDistance'].fillna(0, inplace=True)
train[['kills', 'walkDistance', 'rideDistance', 'killsPerWalkDistance', 'winPlacePerc']].sort_values(by='killsPerWalkDistance').tail(10)


# ### 3.6 Teams

# In[ ]:


train['team'] = [1 if i>50 else 2 if (i>25 & i<=50) else 4 for i in train['numGroups']]

