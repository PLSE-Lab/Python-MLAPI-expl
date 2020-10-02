#!/usr/bin/env python
# coding: utf-8

# # Pandas Exercise: NHL Game Data - Homefield Advantage?
# 
# Welcome! I created the following notebook to serve as both an exploratory data analysis (EDA) and a Pandas tutorial that covers some important Pandas functions and syntax.
# 
# In the world of sports, it has been a long-standing debate as to whether or not "homefield advantage" is real in professional sports. In other words, does your team have an advantage over another team when playing at your home venue as opposed to the opponent's home venue? This question is tough to answer in professional sports because the level of talent and competition is so high.
# 
# In this exercise, you will utilize game data from the National Hockey League (NHL) over the course of the 2010's decade to gain more insight on the homefield advantage question, at least as it relates to hockey. 
# 
# Coded solutions are posted in the proper order with comments. If you are looking for a challenge, try following the analysis wihtout looking at the solution code. Enjoy!
# 
# Start by running the first cell below:

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


# **1) Load the CSV file 'game.csv' into a DataFrame called 'games' using the read_csv function in the Pandas library
# (Use the file path from running the previous cell as the file name for this cell.)**
# 
# **2) Output the number of NHL games played in the 2010's using the 'shape' data attribute of all DataFrames**

# In[ ]:


# Format: data_frame = pd.read_csv(file)

games = pd.read_csv('/kaggle/input/nhl-game-data/game.csv')

# Format: data_frame.shape. This returns a pair of numbers; the first is the number of rows while the second is the
# number of columns. Because each row is a game in the data set, and because we want the the total number of games played,
# we only want the number of rows. To do this, we will use shape[0] (shape[1] would return the number of columns).
games.shape[0]


# **3) Display the first 5 rows of the data set using the .head() function.**

# In[ ]:


# Format: data_frame.head(# of rows to print - default 5)

games.head()


# **4) Create a function that takes in the name of the data frame and checks if the home team won or not. It should return 1 if the home team won and 0 if the home team did not win.**
# 
# **5) Create a new column in the data frame called 'home_team_win' that is set to 1 or 0 for each row depending on whether or not the home team won using the previously created function. The column can be created using square brackets[], and the function can be applied using the .apply() function.**
# 
# **6) Output the first 5 rows of the updated data frame in the same way as before.**

# In[ ]:


# Function decides if home team won by checking the goals scored by home and away team and returns appropriate result

def home_team_win(df):
    if df['home_goals'] > df['away_goals']:
        return 1
    else:
        return 0
    
# Format for creating new column in data frame: data_frame[new_column] = ...

# Format for applying function to rows of data frame: data_frame.apply(function, axis=1). If you would like to
# learn more about the additional input arguments of the .apply() function, you can look up its documentation online.

games['home_team_win'] = games.apply(home_team_win, axis=1)

games.head()


# **7) Filter out unneeded columns from the data frame. Select only the 'season', 'type', 'home_goals', 'away_goals', 'venue', and 'home_team_win' columns and store this updated data frame back into 'games'.**
# 
# **8) Once again output the first 5 rows in order to confirm that the filtering has worked properly.**

# In[ ]:


# Format for choosing only a single column: data_frame[column]
# Format for choosing a subset of columns: data_frame[[column1, column2, ...]]

games = games[['season', 'type', 'home_goals', 'away_goals', 'venue', 'home_team_win']]

games.head()


# **9) Now that the data has been properly set up, let's do some analysis. First, let's find the most impressive victory by a home team. To do this, we must:
#     a. Sort games by the most goals scored by the home team using the .sort_values() function.
#     b. Further sort the games by the least goals scored by the visiting team using the same .sort_values() function.
#     c. Select the first game (row) on the sorted data frame using the .iloc[] function.**

# In[ ]:


# Format for sorting by one column: data_frame.sort_values(column, ascending = True/False). The ascending value should 
# be set to true for ascending and false for descending.

# Format for sorting by multiple columns: data_frame.sort_values([column1,...], ascending = (column1 val, ...)). This
# format is used below.

# Format for selecting rows using .iloc: data_frame[starting row index(inclusive): ending row index(exclusive)]. It
# generally follows the same formatting as standard Python list indexing.

games.sort_values(['home_goals', 'away_goals'], ascending = (False, True)).iloc[0]


# **10) Using a similar structure to the last cell, let's now find the most impressive victory by the away (visiting) team. Again, the sorting priority should still be the most away goals scored followed by the least home goals scored. The .iloc[] function should be used again for the selection.**

# In[ ]:


games.sort_values(['away_goals', 'home_goals'], ascending = (False, True)).iloc[0]


# The 'games' data frame includes data from both regular season games and playoff games. Playoff games only include a subset of the teams in the league each year. In order to get a more even distribution of the teams for assessing home ice advantage, we will look only at the regular season games in the data set for now (we will revisit the playoff games later.)
# 
# **11) Create a new data frame called 'regularGames' that only contains the regular season games from 'games' using conditional selection. The regular season games are marked with an 'R' in the 'type' column of the 'games' data frame.**
# 
# **12) Once again, display the first 5 rows of this new data frame.**

# In[ ]:


# Format for conditional selection: data_frame[condition]

regularGames = games[games.type == 'R']

regularGames.head()


# **13) Next, let's look at the number of goals scored by each team to get an initial sense of whether home ice advantage is real. Begin by finding the average number of goals scored by the home team in the regularGames data frame using the .mean() function with the 'home_goals' column.**

# In[ ]:


# Format: data_frame.column.mean()

regularGames.home_goals.mean()


# **14) Let's now do the same thing for the average number of goals scored by the away team in regular season games.**

# In[ ]:


regularGames.away_goals.mean()


# We can see that the goals scored by home teams is slightly higher on average, but it might not be convincing enough. Let's now move to the actual number of wins for home teams as opposed to away teams in regular season games.
# 
# **15) Display the number of games won by home and away teams using the .value_counts() function. The value_counts function displays the frequency of each separate value of a particular column in the data frame.**

# In[ ]:


# Format: data_frame.column.value_counts()

regularGames.home_team_win.value_counts()


# While this percentage does tell us that the home team has won many more games than the visiting team, let's see what the win ratio of the home team is (wins / total games). Because the home_team_win column is either a 1 or a 0, this ratio can be found by simply computing the average of this column.
# 
# **16) Compute the win ratio of the home team using the .mean() function on the home_team_win column.**

# In[ ]:


regularGames.home_team_win.mean()


# Another good way to visualize the findings being presented is through visuals. Let's make one for the wins vs. losses for the home teams in regular season games.
# 
# **17) Create a pie chart that shows the relative frequencies of wins and losses for home teams in regular season games. Use the .plot() function to do so on the two outcomes of the 'home_team_win' column.**

# In[ ]:


# Format for line plot (default plot): data_frame.column.value_counts().plot()
# Format for pie chart: data_frame.column.value_counts().plot.pie()
# The 1 in the graph represents wins while the 0 represents losses.

regularGames.home_team_win.value_counts().plot.pie()


# The home team appears not not only score more goals but also win more games than the away team does. Using the actual proportion from the graph, running a 1-sample z-test eventually reveals that the results are indeed statistically significant. This indicates that having home ice is indeed an advantage for regular season hockey games. Let's dig into these results more by looking at the venues that actually create the greatest advantage for their teams (for various reasons).
# 
# **18) Display the various venues and their associated ratios for regular season games using the .groupby() function. There are also these following additions: a. The venues must have at least been used for two full NHL seasons (41 home games per season) b. The win ratios must be in descending order c. Only display the venues that have a home win ratio greater than the average home win ratio across all regular season games.**

# In[ ]:


# Format for .groupby(): data_frame.groupby(column). This function groups the values of certain columns based on the
# different categorical values of another column. In this case, we are grouping the mean of the home_team_win column
# (win ratio) based on the different values of the venue column (aka the different venues).

# Additional detail (a): The venue must appear in at least 82 entries in the data frame.
# Additional detail (b): The values must be sorted in descending order at the end.
# Additional detail (c): The home_team_win mean for a venue must be greater than the home_team_win mean for the data frame.

regularGames.groupby(regularGames.venue).home_team_win.mean() [(regularGames.venue.value_counts() >= 82)&  (regularGames.groupby(regularGames.venue).home_team_win.mean()>regularGames.home_team_win.mean())] .sort_values(ascending = False)


# We can now see some of the top venues in terms of home team win ratios. Let's now see if some of these top venues are also leaders in the average number of goals scored by the home team. Let's use a visual to see some of these top venues.
# 
# **19) Create a horizontal bar plot of the top 10 venues in most goals scored by the home team on average. Once again, let's only include venues that have been used for at least 82 games. Furthermore, let's sort the graph in descending order so that the top value in the graph is the highest value.**

# In[ ]:


# Format: Use groupby to once again categorize results by venue.
# The value we are looking to compare is the average of home_goals scored.
# Sort the results
# Apply the restriction to only include venues that have been used at least 82 times.
# .tail() works exactly like .head() except it filters results on the opposite end of the data frame.
# To make a horizontal bar graph, do plot.barh() instead of the plot.pie() from the earlier pie chart.

regularGames.groupby(regularGames.venue).home_goals .mean() .sort_values(ascending = True) [regularGames.venue.value_counts() >= 82].tail(10) .plot.barh()

# Note: If you tried the same thing with 'ascending = False' and .head(10), the bar chart results would be in the same
# general order except flipped. This is due to the way that the bar chart is created. To get the proper direction with
# the highest value on top, you must use 'ascending = True' and .tail(10).


# We can see that many of the venues on the two lists above are common names. While part of this might be attributed to higher team quality (aka ability of players), the venue can certainly still have an impact on helping the home team win. After all, home ice advantage is highly likely given the previously mentioned z-test on the data.
# 
# Moving on, let's look at some of the similar metrics in playoff games. Playoff games are notable for having an increased level of intensity, both from the players as well as from the fans in the venue.
# 
# **20) Create a new data frame called 'playoffGames' and assign it to the filtered set of games with type 'P'. This should be done very similarly to how the 'regularGames' data frame was created.**
# 
# **21) Display the first 5 rows of the 'playoffGames' data frame.**

# In[ ]:


playoffGames = games[games.type == 'P']

playoffGames.head()


# **22) Similarly to before, use the .mean() function to calculate the average number of goals scored by the home team in playoff games.**

# In[ ]:


playoffGames.home_goals.mean()


# **23) Calculate the average number of goals scored by the away team in playoff games.**

# In[ ]:


playoffGames.away_goals.mean()


# **24) Calculate the win ratio of the home team in playoff games.**

# In[ ]:


playoffGames.home_team_win.mean()


# As we can see, the three metrics just calculated are quite similar to those of the regular season, meaning that they can are likely to have similar results. Home ice advantage is likely an advantage for the host team, whether in the regular season or in the playoffs.
# 
# Finally, let's examine the impact of home ice advantage over time across all NHL games (regular season and playoffs). To do this, let's look at the original 'games' data frame and create a line chart visualization.
# 
# **25) Create a line graph showing the home team win ratio over each season in the 'games' data frame. Use the .groupby() function to aggregate win ratio data over each season.**

# In[ ]:


# Format: Use groupby with the 'season' column in the 'games' data frame
# Find the mean of the 'home_team_win' column to find the win ratio of each season
# Use .plot() to create a line graph since .plot() defaults to a line graph

games.groupby(games.season).home_team_win.mean().plot()


# # In Closing
# 
# After spiking in 2012, the effect of home ice advantage appears to be trending downward. However, the win ratio of home teams has continued to stay consistently over 50%, indicating that it is likely a real phenomenon. It will be interesting to see how this evolves over the next decade and onward. 
# 
# I hope you enjoyed working with this notebook, and I also hope that you might have learned a thing or two from the exercise!
