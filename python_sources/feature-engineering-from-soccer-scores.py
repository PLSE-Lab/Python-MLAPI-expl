#!/usr/bin/env python
# coding: utf-8

# # Soccer Matches Feature Engineering
# Using only the scores and teams, how much information can we get?

# # Data Loading

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


import pandas as pd
import os
print(os.listdir("../input"))
data = pd.read_csv('../input/results.csv')


# In[ ]:


data.head()


# In[ ]:


data.home_team.value_counts().head(10)


# In[ ]:


data.away_team.value_counts().head(10)


# In[ ]:


data.tournament.value_counts().head(10)


# # Data Analysis & Visualization

# ## Countries

# #### Let's visualize which countries win, lose, or participate the most.

# In[ ]:


data['home_win'] = data['home_score'] #filler values for now

for i in range(data.shape[0]): #for each row
    
    home_score = data.at[i,'home_score'] #get home score
    away_score = data.at[i,'away_score'] #get away score
    
    if home_score > away_score: #home score win
        data.at[i,'home_win'] = 1
    elif away_score > home_score: #away score win
        data.at[i,'home_win'] = 0
    else:
        data.at[i,'home_win'] = 0.5 #tie


# In[ ]:


data['home_win'].head()


# Let's create a new DataFrame with country stats.

# In[ ]:


unique_countries_home = list(data.home_team.unique())
unique_countries_away = list(data.away_team.unique())
unique_countries = unique_countries_home + unique_countries_away
#Only get unique values by converting to set, then list
unique_countries = list(set(unique_countries))


# In[ ]:


country_stats = pd.DataFrame({"country":unique_countries})


# In[ ]:


country_stats.country.value_counts()


# In[ ]:


data.head(3)


# Good! Now, let's add four columns, away_win, away_lose, home_win, home_lose.

# In[ ]:


#Initialize 4 columns
col_names = ['away_win','away_lose','home_win','home_lose']
for name in col_names:
    country_stats[name] = 0 #placeholder value

#Counting process
for i in range(data.shape[0]):
    
    #Get indexes in country_stat for home and away country
    home_index = country_stats[country_stats['country']==data.at[i,'home_team']].index.values.astype(int)[0]
    away_index = country_stats[country_stats['country']==data.at[i,'away_team']].index.values.astype(int)[0]
    
    #Add 1 to either away win or away lose to away team, and home win or home lose to home team
    if data.at[i,'home_win'] == 1: #The home team has won
        country_stats.at[home_index,'home_win'] += 1
        country_stats.at[away_index,'away_lose'] += 1
    elif data.at[i,'home_win'] == 0: #The home team has lost
        country_stats.at[home_index,'home_lose'] += 1
        country_stats.at[away_index,'away_win'] += 1
    else: #tie. We will just do nothing, since no one really won or lost.
        pass


# In[ ]:


country_stats.head()


# Great! Now, using these four columns, we can generate some statistics.

# In[ ]:


#Get total losses and wins
country_stats['lose'] = country_stats['away_lose'] + country_stats['home_lose']
country_stats['win'] = country_stats['away_win'] + country_stats['home_win']

#Get total games played away and at home, and in general
country_stats['home'] = country_stats['home_win'] + country_stats['home_lose']
country_stats['away'] = country_stats['away_win'] + country_stats['away_lose']
country_stats['games'] = country_stats['home'] + country_stats['away']

#Win-to-lose ratio (the bigger the better)
country_stats['win_lose_ratio'] = country_stats['win']/country_stats['lose']


# In[ ]:


country_stats.head()


# In[ ]:


country_stats[country_stats.lose == 0]


# In[ ]:


#Dealing with two pesky infinity cases, we can just delete them as they are
#not significant
country_stats.drop(list(country_stats[country_stats.lose == 0].index.values.astype(int)),inplace=True)


# In[ ]:


country_stats[country_stats.lose == 0]


# Good! Let's check out our country_stats DataFrame:

# In[ ]:


country_stats.head()


# Let's get a visual for how countries fare in a win-to-lose ratio by creating a scatterplot.

# In[ ]:


grid = sns.JointGrid(country_stats.lose, country_stats.win, space=0, size=7, ratio=5)
grid.plot_joint(plt.scatter, color="b")
plt.plot([0, 0], [700, 700], linewidth=100)
plt.title('Win to Lose Ratio Scatterplot')


# As expected, most of the points are cluttered around the lower left area because most countries haven't played very many games, gradually thining out the more up/right the scatterplot goes.
# Let's try to plot the ratio out with seaborn's handy distplot.

# In[ ]:


sns.distplot(country_stats.win_lose_ratio,rug=True)
plt.title("Country Win to Lose Ratio Distribution Plot")


# Most teams have lost a little more than they have won (a win-lose ratio just under 1). 

# Let's get some additional metrics:
# - away win-to-lose ratio
# - home win-to-lose ratio
# - difference between home and away win-to-lose ratios
#      - Does the team win more at home (positive value) or away (negative value), and to what degree (absolute value of value)?

# In[ ]:


country_stats['away_win_lose_ratio'] = country_stats['away_win']/country_stats['away_lose']
country_stats['home_win_lose_ratio'] = country_stats['home_win']/country_stats['home_lose']

country_stats['home_away_degree'] = country_stats['home_win_lose_ratio'] - country_stats['away_win_lose_ratio']


# In[ ]:


country_stats.head()


# Let's see which countries have the highest home_away_degree values.

# In[ ]:


country_stats['home_away_degree'].nlargest(5)


# Uh oh, we've come across pesky infinity again! Let's check out which countries these are and if they are significant.

# In[ ]:


country_stats.loc[list(country_stats[country_stats.home_away_degree == float('inf')].index.values.astype(int))]


# These teams have all played 2 or over games, so let's try to take care of this by setting them to nan.

# In[ ]:


problem_indexes = list(country_stats[country_stats.home_away_degree == float('inf')].index.values.astype(int))
bad_columns = ['home_win_lose_ratio','home_away_degree']
for column in bad_columns:
    for index in problem_indexes:
        country_stats.at[index,column] = np.nan


# Our new leaderboard:

# In[ ]:


country_stats['home_away_degree'].nlargest(5)


# Let's get these countries' rows for further analysis.

# In[ ]:


country_stats.loc[country_stats['home_away_degree'].nlargest(5).index.values.astype(int)]


# All of these teams have played at least 5 teams.
# What is particularly interesting is the Jersey Team - they've played 82 games and still have the 2nd highest difference between their home and away win-lose ratios. Let's focus on this team.

# In[ ]:


country_stats[country_stats.country=='Jersey']


# Jersey has played a substantial amount of home and away games (48 and 34 respectively), which means that the numbers are secure.
# This suggests that Jersey has an advantage when playing at home - a big advantage.

# Let's look at the smallest ones (the most negative)

# In[ ]:


country_stats['home_away_degree'].nsmallest(5)


# Pesky infinity again! We know the drill.

# In[ ]:


country_stats.loc[country_stats['home_away_degree'].nsmallest(3).index.values.astype(int)]


# In[ ]:


problem_indexes = country_stats['home_away_degree'].nsmallest(3).index.values.astype(int)
bad_columns = ['away_win_lose_ratio','home_away_degree']
for column in bad_columns:
    for index in problem_indexes:
        country_stats.at[index,column] = np.nan


# In[ ]:


country_stats['home_away_degree'].nsmallest(5)


# Let's check these out:

# In[ ]:


country_stats.loc[country_stats['home_away_degree'].nsmallest(5).index.values.astype(int)]


# Woah! Basque county has played a substantial amount of games at home and away, but perform a lot better away than at home - a bit larger than 2.75x better.

# Now, let's create a dataframe that gets the top performers in each category.

# In[ ]:


top_cs = country_stats
top_cs = top_cs.iloc[0:0] #clearing out all data
top_cs.drop('country',axis=1,inplace=True)


# In[ ]:


top_cs


# In[ ]:


for i in range(5):
    top_cs.loc[i] = 0


# In[ ]:


top_cs


# And the ranking code:

# In[ ]:


top_cs.reset_index()

def get_country(index): #function to get country by index
    return country_stats.loc[index]['country']
    
#For each column
for column in top_cs.columns:
    
    placement_index = 0
    
    #For each index in a list of top indexes by column value
    for index in list(country_stats[column].nlargest(5).index.values.astype(int)):
        
        #Assign value to country
        top_cs.loc[placement_index,column] = get_country(index)
        #Next index
        placement_index += 1


# Viola!

# In[ ]:


top_cs


# Above is a nice little chart with each of the features we managed to extract, and which countries top in them.
