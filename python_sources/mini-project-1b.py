#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Mini Project 1B: Summarizing an NBA dataset
# 
# **Overview**
# 
# For this project, I plan to answer multiple questions I am interested in through a series of visualizations.
# 
# These are some of the questions I would like to answer:
# * How does age affect a players performance? What age on average is a normal players' "prime"? What about an all star player? 
# * At what height and weight do players score the most points on average? 
# * What countries' players perform the best? 
# * What draft class from 1996 to 2019 performed the best? 
# * With the advancement of 3 point shooting and efficient scoring, how are star players points per game being affected? 
# 
# 
# **Data Profile**
# 
# The data set features multiple entities that may be extracted. For example, height, weight, player name, college, country, points per game, rebounds per game etc. Most of the data I decide to extract will depend on the questions I hope to answer.
# 
# Some possible limitations of using the data is that some of the data contains strings and integers in the same columns. For example, some players were undrafted instead of given a draft position (1, 2, 34, 29 etc.). This isnt necessarily a limitation but it does remind me that I have to be careful with converting the data when I do my analysis. 

# **Analysis**
# 
# We will start by taking a look at the data:

# In[ ]:


#add in matplotlib library so we can create visualizations.
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

players = pd.read_csv('../input/nba-players-data/all_seasons.csv') #set dataframe of csv as "players"
players.tail(5) #preview the end of the dataset to see attributes


# ***How does age affect a players performance? What age on average is a normal players' "prime"? **
# *
# 
# Here we attempt to create a scatterplot using age vs. points to visually see where players tend to peak and how the decline looks like.

# In[ ]:


age=players[['age']]   #define age and points per game in dataframe
pts=players[['pts']]

plt.style.use('seaborn-poster') #set style to seaborn-poster so that plots are large and easier to read
plt.style.use('fivethirtyeight') #set style to fivethirtyeight so that there are guidelines

plt.scatter(age,pts, alpha=0.5, c='blue') #scatter plot showing age on the x-axis and pts on the y.


# We can see here that players generally peak in points per game at around 26-29 years old, then they start to decline as they get older.

# We will now try to see this effect on one player. We will use Kobe Bryant as an example.

# In[ ]:


kobe = players[players.player_name == 'Kobe Bryant'] #create new dataframe with the values of kobe
kobe #preview the new kobe dataframe


# In[ ]:


kb_age=kobe[['age']] #define age and points per game column in kobe dataframe
kb_pts=kobe[['pts']]

plt.scatter(kb_age,kb_pts, alpha=0.5, c='purple')  #scatterplot using kobe attributes


# Here we can see a similar pattern to the scatterplot above. Kobe's points per game peaked around 26 years old at 35+ PPG, then started to decline. 

# ***With the advancement of 3 point shooting and efficient scoring, how are players points per game being affected? ***
# 
# Here we will create a bar graph to visualize the averages of points per game from 1996-2019 seasons. We will first take the averages of all the points per game stats on each player. Then we will plot these averages per season from 1996-97 to 2019-20 seasons. This will show us if there is any trend of increase or decrease in scoring.

# In[ ]:


#Here we will set "season" as a new dataframe that makes season, the y axis, and averages all of the stats in the dataframe.
season = players.groupby(['season']).mean()

season['pts'].plot(kind='barh',legend=True)   #plot points per season in a horizontal bar graph using the season dataframe we just created.


# Here we can see that there is a fluctuation in scoring per season. There seems to be an increase in 2008-2010, but then a decline until another peak in 2018-2020. With the increase in efficient scoring and 3 point shooting in modern basketball, we expect to see an increase, however this is a bit slower of a pace than expected.  

# ***At what height and weight do players score the most points on average?***
# 
# Now we will look at heights and weights of all players and compare this with points per game to see at what height or weight are players scoring the most points.

# In[ ]:


height=players['player_height']/30.48  #define height in dataframe and convert cm to feet
weight=players['player_weight']*2.205  #define weight and convert kg to lb

plt.scatter(height,pts, alpha=0.5, c='blue') #scatterplot using height as x-axis and points as y-axis


# We can see that at around 6 and half feet tall is where most high scoring happens. Over 7 feet and below 6 feet, there are not too many high scoring players.

# In[ ]:


plt.scatter(weight,pts, alpha=0.5, c='red')  #scatterplot using height as x-axis and points as y-axis


# It seems that at 190 to 240 lbs players score the most points. There are quite a few exceptions however with players over 300 lbs sscoring just as many points. This scatterplot does not give us an accurate representation as most players are around the 180-250 lb range anyway.

# ***What draft class from 1996 to 2019 performed the best?***
# 
# Now we will try to see what draft class performed the best by determining each draft classes average points, rebounds, and assists. We will visualize this in a bar graph.
# 
# First, we will exclude undrafted players as they do not count. We are also only including 1996-2019 drafts since the dataset only includes stats for those seasons and not any seasons before. Then, we will create a new dataframe and find the averages of all of these stats. Finally, we will plot them in a segmented bar graph.
# 

# In[ ]:


draft_players= pd.read_csv('../input/nba-players-data/all_seasons.csv') #create new df to manipulate

#creating new column with only drafted players. We want to make sure these are only storing the drafted years and not undrafted.
draft_players["drafted"]= draft_players["draft_year"].str.isdigit() 

#creating an object that will store drafted players
was_drafted = draft_players['drafted'] == True

#creating a filter that will only include drafted players
draft_players.where(was_drafted, inplace = True) #where function finds players that are drafted

after_95 = draft_players["draft_year"].astype(float)>=1996  #this will only find players that were drafted 1996 and later
draft_players.where(after_95, inplace = True) 

#set "draft_class" as a new dataframe that makes draft_year, the row axis, and averages all of the stats in the dataframe.
draft_class = draft_players.groupby(['draft_year']).mean()



draft_class.plot.bar(y=['pts','reb','ast']) #plot a segmented bar graph containing points, rebounds, and assists from the draft_class dataframe


# Here we can see each draft class from 1996 to 2019 and each of their main stats on average. We see that 1996 is the highest scoring draft class. This makes sense since high scoring players such as Kobe Bryant and Allen Iverson were drafted in 1996. Note that the later year draft classes are relatively low since those players have not hit their prime yet. 

# ***What countries' players perform the best?***
# 
# Finally, we will look at each countries average stats in each country from the dataset.
# 

# In[ ]:


#set "country" as a new dataframe that makes country, the row axis, and averages all of the stats in the dataframe.
country = players.groupby(['country']).mean()

country.plot.bar(y=['pts','reb','ast']) #plot a segmented bar graph containing points, rebounds, and assists from the country dataframe


# The graph above is a bit difficult to read as there are so many NBA players from different countries. I was not able to make it larger using the figure function. 
# 
# From reading the graph though, we see that the US Virgin Islands, Bahamas, and Germany all have the highest points per game average. The data can be skewed to look like this as there are only a few players in those countries and they just happened to be high scoring players. Since there are so many players from the USA, the average is lower because we are factoring in low scoring players. Virgin Islands is also very high because the dataset has a typo error in which there are two. One includes Tim Duncan which is a high scoring and high rebounding player.
# 
# If we take a look at rebounds, we can see that Congo and  is very high. This is because a lot of Centers and Power Forwards are from the Congo. Such as Dikembe Mutombo. 
# 
# Taking a look at assists, we see that Spain is the highest. This may be because of players such as Jose Calderon and Ricky Rubio which are well-known as Spanish point guards with great passing ability. 

# **Conclusion/Directions for Future Work**
# 
# All in all, this project was both exciting and challenging for me as it did answer some interesting questions about the NBA. Being an NBA fan, I was able to compare different relationships such as players points per game and how it changes out of their prime. I was also able to relate certain outcomes to my knowledge of the NBA. The challenging part of it was definitely sorting through the data to set up the plots. Once I figured out how to use functions such as "where" and "groupby", the analysis became easier to create.
# 
# Future work would consist of diving more deep into the weight and height analysis. I simply made a scatterplot showing the relationship but it is hard to tell what weight and height perform the best as there is such a large group of players with similar heights/weights. A regression analysis can also be done for the age vs pts plots, we can use it to predict a current players points per game throughout their entire career.  
