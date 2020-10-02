#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (20,10)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


transfers_dataframe = pd.read_csv('../input/top250-00-19.csv')
transfers_dataframe.tail()


# > **Exploratory Data Analysis**
# > The dataset consists of football transfers over major leagues from the year 2000 - 2019.
# The major attributes are -
# * Name of the Player
# * Position
# * Age
# * Team_from - Team from which the player moved
# * League_from - League from which the player moved
# * Team_to - Team from which the player moved
# * League__to - League from which the player moved
# * Market_value
# * Transfer Fee
# 
# Lets take a deep dive into the dataset now
# 

# In[ ]:


print('The total number of players transferred till date  : ' + str(transfers_dataframe['Name'].nunique()))
print('The total number of unique player positions transferred till date  : ' + str(transfers_dataframe['Position'].nunique()))
print('The total number of unique teams that have transferred its players till date  : ' + str(transfers_dataframe['Team_from'].nunique()))
print('The total number of unique teams that have brought in players till date  : ' + str(transfers_dataframe['Team_to'].nunique()))
print('There is a good difference in count of selling clubs and buying clubs this means that some clubs are more interested in buying from other rather than growing academy players.')
print('Top Transfer based on Transfer Value is : ')
print(''+str(transfers_dataframe.iloc[transfers_dataframe['Transfer_fee'].idxmax()]))

Lets just go through the basic analysis as of now and have a look at the basic distribution
# In[ ]:


top_10_transferred_players = pd.DataFrame(transfers_dataframe['Name'].value_counts())
# Based on the Number of Transfers
top_10_selling_clubs = pd.DataFrame(transfers_dataframe['Team_from'].value_counts())
top_10_selling_leagues = pd.DataFrame(transfers_dataframe['League_from'].value_counts())
top_10_buying_clubs = pd.DataFrame(transfers_dataframe['Team_to'].value_counts())
top_10_buying_leagues = pd.DataFrame(transfers_dataframe['League_to'].value_counts())


# In[ ]:


# Top Leagues and Teams as per Transfer value
top_10_selling_clubs_money = transfers_dataframe.groupby('Team_from',as_index = False)['Transfer_fee'].sum().sort_values(by = 'Transfer_fee',ascending = False)
top_10_selling_leagues_money =  transfers_dataframe.groupby('League_from',as_index = False)['Transfer_fee'].sum().sort_values(by = 'Transfer_fee',ascending = False)
top_10_buying_clubs_money =  transfers_dataframe.groupby('Team_to',as_index = False)['Transfer_fee'].sum().sort_values(by = 'Transfer_fee',ascending = False)
top_10_buying_leagues_money = transfers_dataframe.groupby('League_to',as_index = False)['Transfer_fee'].sum().sort_values(by = 'Transfer_fee',ascending = False)


# In[ ]:


# Lets inspect the above dataframes
top_10_buying_clubs_money.head(10)


# **Lets Just Visualise the above data to get a good grasp of things**
# 

# # Based on the number of Transfers

# In[ ]:


plt.subplot(2,2,1)
plt.bar(x = top_10_selling_clubs.head(10).index , height =top_10_selling_clubs['Team_from'].head(10))
plt.xticks(rotation = 45)
plt.ylabel('Number of Players')
plt.title('Top Selling Clubs')

plt.subplot(2,2,2)
plt.bar(x = top_10_buying_clubs.head(10).index , height =top_10_buying_clubs['Team_to'].head(10))
plt.xticks(rotation = 45)
plt.ylabel('Number of Players')
plt.title('Top Buying Clubs')

plt.subplot(2,2,3)
plt.bar(x = top_10_selling_leagues.head(10).index , height =top_10_selling_leagues['League_from'].head(10))
plt.xticks(rotation = 45)
plt.ylabel('Number of Players')
plt.title('Top Selling Leagues')

plt.subplot(2,2,4)
plt.bar(x = top_10_buying_leagues.head(10).index , height =top_10_buying_leagues['League_to'].head(10))
plt.xticks(rotation = 45)
plt.ylabel('Number of Players')
plt.title('Top Buying Leagues')
plt.tight_layout()

plt.show()


# **Gives a  lot of short and interesting stories**
# Food for Thought -- 
# 1. Porto ,Benfica and Udinese Calcio are amongst the top sellers but not in the top buyers 
# 2. Inter seems to be the squad that has changed a lot as it is on top for buying and selling list.
# 
# Coming on the League Perspective Now -- 
# 1. A lot of players have left Serie A and Ligue 1 and most probably joined Premier League .
# 2. It also signifies that  these leagues maybe now have more homegrown players with each passing season.

# # Based on the Amount spent in transfers

# In[ ]:


plt.subplot(2,2,1)
plt.bar(x = top_10_selling_clubs_money['Team_from'].head(10) , height =top_10_selling_clubs_money['Transfer_fee'].head(10),color = 'green')
plt.xticks(rotation = 45)
plt.ylabel('Amount')
plt.title('Top Selling Clubs Based on Money')

plt.subplot(2,2,2)
plt.bar(x = top_10_buying_clubs_money['Team_to'].head(10) , height =top_10_selling_clubs_money['Transfer_fee'].head(10),color = 'green')
plt.xticks(rotation = 45)
plt.ylabel('Amount')
plt.title('Top Buying Clubs Based on Money')

plt.subplot(2,2,3)
plt.bar(x = top_10_selling_leagues_money['League_from'].head(10) , height =top_10_selling_leagues_money['Transfer_fee'].head(10),color = 'orange')
plt.xticks(rotation = 45)
plt.ylabel('Amount')
plt.title('Top Selling Leagues Clubs Based on Money')



plt.subplot(2,2,4)
plt.bar(x = top_10_buying_leagues_money['League_to'].head(10) , height =top_10_buying_leagues_money['Transfer_fee'].head(10),color = 'orange')
plt.xticks(rotation = 45)
plt.ylabel('Amount')
plt.title('Top Buying Leagues Clubs Based on Money')

plt.tight_layout()

plt.show()


# **Gives a  lot of short and interesting stories**
# Food for Thought -- 
# 1. Monaco,Porto and Benfica are amongst top sellers 
# 2. Paris SG and Spurs are the teams which are clearly buying more players than selling.
# (Building their squad)
# 
# Coming on the League Perspective Now -- 
# 1. Premier League is clearly the league where most of the valuable in and out transfers take place
# 2. Serie A is clearly losing more value players than gaining.

# In[ ]:





# 

# 

# # Scatter Plot (Players In Vs Player Out)

# In[ ]:


#scatter_league = pd.merge(top_10_selling_clubs,top_10_buying_clubs,left_on =top_10_selling_clubs.index,right_on = top_10_buying_clubs 
scatter_clubs = top_10_selling_clubs.join(top_10_buying_clubs)
scatter_league = top_10_selling_leagues.join(top_10_buying_leagues)
plt.subplot(2,1,1)
plt.scatter(scatter_clubs['Team_from'],scatter_clubs['Team_to'])
plt.xlabel('Transfers Out')
plt.ylabel('Transfers In')
plt.title('Team Perspective')

plt.subplot(2,1,2)
plt.scatter(scatter_league['League_from'],scatter_league['League_to'], color  = 'red')
plt.xlabel('Transfers Out')
plt.ylabel('Transfers In')
plt.title('League Perspective')
plt.tight_layout()
plt.show()


# **There seems to be a positive correlation between Transfers In and Transfers Out, Lets Just validate the fact**

# In[ ]:


print('Team Transfer Correlation')
print(str(scatter_clubs.corr()))


# In[ ]:


print('League Transfer Correlation ')
print(str(scatter_league.corr()))


# **Lets analyse profitability of various groups**

# In[ ]:





# **Lets take a look at the age of players that were transferred**

# In[ ]:


transfers_dataframe['Age'].hist(bins = 35,color = 'green')


# # Players between the age of 20 - 30 are transferred the most

# In[ ]:




