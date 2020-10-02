#!/usr/bin/env python
# coding: utf-8

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


# **Impact of Map Pick on Game Win**
# 
# In CS:GO, there is generally a map selection process between the two teams playing a match to determine what map(s) will be played. This usually involves a process of both picking maps to play, and banning away maps such that the other team cannot then pick them. 
# 
# A core belief is that a team should have a higher chance than the opponent to win their map pick. 
# 
# I wanted to quantify how often teams win their own pick. 
# 
# 
# I started with a data set from kaggle.com with CS:GO professional matches scraped from HLTV.org spanning 11/2015 - 3/2020. This data was read into pandas, a data analysis library for Python.
# 
# The data can be found here: https://www.kaggle.com/mateusdmachado/csgo-professional-matches

# In[ ]:


d_picks = pd.read_csv("/kaggle/input/csgo-professional-matches/picks.csv")
d_economy = pd.read_csv("/kaggle/input/csgo-professional-matches/economy.csv")
d_results = pd.read_csv("/kaggle/input/csgo-professional-matches/results.csv")
d_players = pd.read_csv("/kaggle/input/csgo-professional-matches/players.csv")


# **Filtering the Data**
# 
# This dataset is quite broad as it includes all HLTV matches from the end of 2015 to March 2020. For a more applicable snapshot, I filtered the data to include only the following:
# 
# * **Only include the first two games within a best of 3 Series.**
# 
# Matches are most often played as best of 1's, or best of 3's. For this analysis, I will only focus on best of 3's, as most analysts generally look at pick/ban in terms of a best of 3 series. Additionally, there is generally no "pick" in a best of 1 as the common methodology is for both teams to ban 3 maps until there is a surviving map to play.
# 
# * **Only include games between two teams within the top 20 (at the time the game was played).**
# 
# The scene is generally focused around the top twenty, and the pick/ban phase is considered quite important at this level. Imposing this filter also removes wins that top teams gain by farming weaker teams not near their skill level. 
# 
# * **Remove teams which did not pick more than 20 games**
# 
# This filter gives some strength to the analysis such that teams with only a few map picks do not generate outlier win/loss percentages.  
# 
# * **Only Include games after 12/31/2016**
# 
# I added this filter as I wanted to create a time-span of data of about 2 years. I felt this time span was large enough to gain a good snapshot, but still small enough that it hopefully curbs part of the issue generating from player changes within a team (see disclaimer at the bottom).

# **Python Code**
# 
# The below python code cleans up the data and generates a final table that with three columns:
# 
# * Team Name
# * Win Percentage on Picked Maps
# * Number of Games Played

# In[ ]:


d_picks.rename(columns={"team_1":"team_1_pick","team_2":"team_2_pick" }, inplace=True)

new_df = pd.merge(d_results, d_picks, left_on="match_id", right_on="match_id")

#filter for best of 3's
new_df = new_df[new_df["best_of"] == "3"]

#add new column for winning team name
new_df["winning_team_name"] = ""
new_df.loc[new_df["map_winner"]==1, "winning_team_name"] = new_df["team_1"]
new_df.loc[new_df["map_winner"]==2, "winning_team_name"] = new_df["team_2"]

#create column for name of team who picked the map
new_df["map_picker"] = ""
new_df.loc[new_df["_map"] == new_df["t1_picked_1"], "map_picker"] = new_df["team_1_pick"]
new_df.loc[new_df["_map"] == new_df["t2_picked_1"], "map_picker"] = new_df["team_2_pick"]
new_df.loc[new_df["_map"] == new_df["left_over"], "map_picker"] = "left_over"

#remove decider games
new_df = new_df[new_df["map_picker"] != "left_over"]

#create column for pick win
new_df["pick_win"] = new_df["winning_team_name"] == new_df["map_picker"]

#create column for rank of winning team
new_df["winning_rank"] = 0
new_df.loc[new_df["map_winner"]==1, "winning_rank"] = new_df["rank_1"]
new_df.loc[new_df["map_winner"]==2, "winning_rank"] = new_df["rank_2"]

#create column for rank of losing team
new_df["losing_rank"] = 0
new_df.loc[new_df["map_winner"]==1, "losing_rank"] = new_df["rank_2"]
new_df.loc[new_df["map_winner"]==2, "losing_rank"] = new_df["rank_1"]

#filter for only teams in the top twenty
new_df = new_df[new_df['winning_rank']< 21]
new_df = new_df[new_df['losing_rank']< 21]

#filter for after a certain date
new_df['date_x'] = pd.to_datetime(new_df['date_x'])
new_df = new_df[new_df['date_x'] > "12-31-2016"]

#groupby team
grouper = new_df.groupby("map_picker")["pick_win"].value_counts(normalize=True)
grouper = grouper[grouper.index.isin([True], level=1)]

#Remove redundant index and sort
grouper = grouper.reset_index(level=1, drop=True)
grouper = grouper.sort_values(ascending = False)
grouper = grouper.to_frame()
grouper = grouper.reset_index()

#Add column based on number of games
filter_series = new_df["map_picker"].value_counts()
filter_series = filter_series.to_frame()
filter_series = filter_series.reset_index()
grouper = pd.merge(grouper, filter_series, left_on="map_picker", right_on="index")
grouper = grouper.drop("index", 1)
grouper.rename(columns={"map_picker_x":"Team","pick_win": "Win Percentage", "map_picker_y":"Number of Games" }, inplace=True)
grouper = grouper[grouper["Number of Games"] > 20]


# In[ ]:


print(grouper)


# **Analysis**
# 
# The generated information was quite interesting to me for a number of reasons:
# 
# * **Astralis** - It provides just more evidence that Astralis is the best core of all time. Even other strong teams above (FaZe, Liquid, Evil Geniuses) are over 10% below that of Astralis. I imagine this percentage is boosted heavily by their Nuke streak, as well as generally being unbeatible (on any map) in 2018. 
# 
# * **Ence** - Ence being second feeds into the general narrative that Ence was a good team because of tactics and guile, using aspects such as map pick/ban to gain wins over better teams. It impresses me they were able to have the second highest win percentage on their own pick, while still being so far below the other elite teams in terms of skill. It is also a further indictment of their choice to kick Aleksib. I don't think they could ever have reached this percentage without them. 
# 
# * **Space Soldiers** - It is interesting that they are so high on the list. I imagine this is a function of farming low top 20 teams online, but I would have to look more into the data to determine the exact cause. 
# 
# * **General** - It appears to me that while your map pick can generate an increased chance to win a game, you still have to be a quality team in order to execute. Your map pick is by no means a free win, even if you are a tactical team. For example, BIG is second to last, and they are known to follow a strong gameplan. Additionally, neither North nor Optic, which were at times led by MSL with a fairly rigid system, have strong winrates (Optic is even last).  
# 
# **Disclaimer**
# 
# The data used for this project is not perfect as it includes online matches and does not account for player changes within a team. However, I thought this was a fun exercise in data science and a bit illuminating as to the impact of pick/ban. Please let me know if you see any issues with the data.
