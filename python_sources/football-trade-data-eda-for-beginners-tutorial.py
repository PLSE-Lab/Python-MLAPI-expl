#!/usr/bin/env python
# coding: utf-8

# ### **Please consider upvoting if you liked the post :)**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # **Read in Data**

# In[ ]:


# Store data in df
df = pd.read_csv("../input/top250-00-19.csv")


# # **Simple Exploration of Data**

# In[ ]:


# Look at first five rows of data
df.head()


# In[ ]:


# Look at how many rows and columns the data have
df.shape


# In[ ]:


# Basic statistics about the data and its variables
df.describe()


# # **EDA (Explatoratory Data Analysis)**

# EDA is important because it allows us to better understand the data before modelling (although there is no specific modelling involved for this data... but of course, predicting the "transfer fee" of a player can be one potentially interesting project). Also, we can preliminary insights about the data through EDA  even before using anything fancy like Machine Learning or Deep Learning for modelling!

# In[ ]:


plt.style.use('ggplot')


# In[ ]:


plt.rcParams['figure.figsize'] = (9, 9)


# ## Distribution of variables

# In[ ]:


# Histogram
df.hist()
plt.tight_layout()


# Age seems to be pretty normally distributed but market_value and transfer_fee seem very skeweed to the right with very little players traded at very high prices. This is fair becuase we only have "few" star players who are traded for exorbitant prices

# ## Which "position" has been traded the most?

# In[ ]:


sns.countplot(x='Position',data=df)
plt.xticks(rotation=90)


# Center Foward(CF) traded the most followed by Center Back and multiple midfielder positions. The **Center positions** are traded the most!

# ## Which Season was football trade the most active?

# In[ ]:


sns.pointplot(x='index',y='Season',data=pd.DataFrame(df.Season.value_counts()).reset_index().sort_values('index'))
plt.xticks(rotation=90)
plt.xlabel('Season')
plt.ylabel('Number of Trade')


# Multiple seasons had the most number of trade (e.g. 2013-2014, 2016-2017) and the number of trades doesn't seem to display any kind of seasonal trend over time!

# ## Which league spent the most money buying players for every season? & Which league sold the most players in terms of average transfer fees?

# Groupby is a very useful operation for finding answers to these kinds of questions! Groupby allows us to perform calculations on every group!

# In[ ]:


# Order by mean transfer fee from highest to lowest
df.groupby(['League_to']).Transfer_fee.describe().sort_values('mean', ascending=False)


# The league that spent the most money, on average, to buy players for the past 13 years or so was Spanish LaLiga and British Premiere League. One unexpected league that was ranked 5th place was the Super League of Greece which is less known to global soccer fans. But it has definitely spent a lot of money bringing expensive players to the league!

# In[ ]:


# Order by mean transfer fee from highest to lowest
df.groupby('League_from').Transfer_fee.describe().sort_values('mean', ascending=False)


# Leagues that sold players the most in terms of average transfer fee are not that different from the previous list of leagues which spent the most money on average to buy players. But Ligue1, the top division of French men's soccer league, is topping the list. However, we can see there were only two trades that happened (as we can see from the "count" column with the number 2) but the average of those two trades were significant enough to make Ligue1 rank 1st for this list. Let's look at which two players from Ligue1 were sold to other teams

# In[ ]:


# Two players from Ligue1 sold to other teams
df[df.League_from == 'Rel. Ligue 1']


# ## Which Teams spent the most money (total sum) in buying players for the past 13 years?

# In[ ]:


# Top10 Teams that spend the most money in terms of total amount in buying players for the past 13 years
df.groupby("Team_to").Transfer_fee.sum().sort_values(ascending=False).head(10)


# Big teams in premeire league, La Liga, Serie A and Ligue1 spent the most money in buying star players!

# ## Comparison of players sold from Premeire League, La Liga and Serie A

# In[ ]:


df_from_top3_league = df[df.League_from.isin(['Serie A','Premier League','LaLiga'])]


# We can use "Parallel Coordinates" to compare players sold from the three leagues!

# In[ ]:


from pandas.plotting import parallel_coordinates
parallel_coordinates(df_from_top3_league.drop(['Name','Position','Team_from',
                             'Team_to','League_to','Season'], axis=1), 'League_from',colormap='rainbow')


# - We can observe the three outliers (the three strands of data whose transfer fee are way higher than the market value and also way up in the graph compared to the cluster of strands below) let's look at which players they are
# - Most strands seem to be clustered around at the bottom with the green(Serie A) and orange(Premier League) having been valued at slightly higher market value and transferred for higher fees than LaLiga players

# In[ ]:


df_from_top3_league.sort_values('Transfer_fee',ascending=False).head(3)


# Those three outlier players were the big shots! Christiano Ronaldo, Coutinho and Neymar lol

# In[ ]:


sns.FacetGrid(df_from_top3_league, hue="League_from", height=10).map(plt.scatter, "Market_value", "Transfer_fee",alpha=0.6).add_legend()


# We can also fit a linear regression line on each of the league cluster!

# In[ ]:


sns.lmplot("Market_value", "Transfer_fee", data=df_from_top3_league, hue='League_from',height=10)


# - Players sold from Premier League have less variability compared to the other two leagues. Majority of the points are clustered at the lower hand left corner grid, so the transfer
# - The slopes of the linear regression lines are steepest in the order of LaLiga, Premier League and Serie A. This shows us that for one increment of market value of a player, it is more expensive to purchase a player from LaLiga than Premier League

# ## Correlation amongst variables

# In[ ]:


# Correlation Heatmap
sns.heatmap(df.corr(),annot=True)


# - Correlation between market value and transfer fee is very high. This means the more expensive the player is evaluated in the market, the more likely it is going to be sold at a expensive price.
# - The correlation value betwen age and transfer free is negative although the strenght of the correlation isn't that strong. This at least suggests that the older the age, the less value it has in the market. Of course, I expect the returns to age is a reversed bell shaped curve because as a player gets older, he gets more experience and hence becomes more valuable as a player. But after a certain threshold, the player is considered to be old and is valued lowly in the market.

# ### **Please consider upvoting if you liked the post :)**
