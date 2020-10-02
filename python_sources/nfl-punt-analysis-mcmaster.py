#!/usr/bin/env python
# coding: utf-8

# The objective of the Punt Analysis is to determine a potential rule change that may reduce the occurrence of concussions during punt plays.
# 
# Below is an analysis of Punting Data from the 2016 and 2017 NFL Season.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


concussed_games = pd.read_csv("../input/video_review.csv")


# In[ ]:


play_data = pd.read_csv("../input/play_information.csv")


# In[ ]:


environmental_conditions = pd.read_csv("../input/game_data.csv")


# In[ ]:


player_data = pd.read_csv("../input/play_player_role_data.csv")


# In[ ]:


injured_players = pd.read_csv("../input/video_review.csv")


# In[ ]:


injured_players = injured_players.set_index("GSISID")


# In[ ]:


concussed_games = concussed_games.set_index("GameKey")


# In[ ]:


position_dictionary = pd.read_csv("../input/player_punt_data.csv")


# In[ ]:


position_dictionary = position_dictionary.set_index("GSISID")


# In[ ]:


player_data = player_data.set_index("GSISID")


# In[ ]:


concussion_details = concussed_games.merge(play_data, left_on=["Gamekey","PlayID"])


# In[ ]:


concussed_games.columns


# In[ ]:


concussed_games.drop(["Turnover_Related", "Friendly_Fire"], axis=1)


# In[ ]:


play_data.columns


# In[ ]:


play_data.drop(["Game_Date","Game_Clock","Play_Type","Score_Home_Visiting","PlayDescription"],axis=1)


# In[ ]:


combined = pd.merge(concussed_games,play_data, on=["GameKey","PlayID","Season_Year"] )


# In[ ]:


combined.columns


# 

# In[ ]:


combined.drop(["Score_Home_Visiting","PlayDescription"],axis=1)


# In[ ]:


Finished = pd.merge(combined, position_dictionary, on="GSISID")


# In[ ]:


position_dictionary.columns


# In[ ]:


Finished.drop("Number",axis=1)


# In[ ]:


Finished = Finished.drop(["Home_Team_Visit_Team","Score_Home_Visiting","PlayDescription"],axis=1)


# In[ ]:


Finished.columns


# In[ ]:


NFL_Punts=Finished


# The Data I want to analyze is assembled. Let us analyze it.

# In[ ]:


import matplotlib as plt


# In[ ]:


NFL_Punts.drop_duplicates(inplace=True)


# Here we see the number of Punts that Result in a Concussion in the Pre-Season is 12, where the number of Regular season punts is 25.
# 
# The 4 Weeks of Preseason games results in 32% of all Punt Plays that resulted in a concussed player. (Vs. 68% across 16 active weeks)
# 
# Note: There were no Punts that resulted in concussions during the 2016 or 2017 Post season.

# In[ ]:


NFL_Punts[NFL_Punts.Play_Type == "Punt"].Season_Type.value_counts()


# In[ ]:


NFL_Punts[NFL_Punts.Play_Type == "Punt"].Season_Type.value_counts(normalize = True)


# We can see here that the 5 players most likely to be concussed are:
# Tight End,
# Inside Linebacker,
# Corner Back,
# Outside Linebacker and 
# Wide Reciever

# In[ ]:


NFL_Punts[NFL_Punts.Play_Type == "Punt"].Position.value_counts()


# In[ ]:


print(NFL_Punts[NFL_Punts.Season_Type == "Pre"].Week)


# In[ ]:


NFL_Punts[NFL_Punts.Season_Type == "Pre"].Week.value_counts()


# Amongst the Regular Season Games, Weeks 11-16 represents 68% (17 of 25) of regular season games with Punt plays that result in concussion.

# In[ ]:


NFL_Punts[NFL_Punts.Season_Type == "Reg"].Week.value_counts()


# Punting plays that result in concussion are twice as likely during the Second and Third Quarters of the game. Rather than a generally standard deviation.

# In[ ]:


NFL_Punts[NFL_Punts.Play_Type == "Punt"].Quarter.value_counts()


# In[ ]:


play_data.Quarter.value_counts(normalize=True)


# Concussions during punt play are the result of one of these 4 activities, Blicking, Being Blocked, Tackling or Being Tackled.
# 
# In Tackling, Wide Recievers are most likely to suffer concussions during punts.
# In being Blocked, Tight Ends are most likely to suffer concussions during punts
# 

# In[ ]:


NFL_Punts[NFL_Punts.Play_Type == "Punt"].Primary_Partner_Activity_Derived.value_counts()


# In[ ]:


NFL_Punts[NFL_Punts.Primary_Partner_Activity_Derived == "Blocked"].Position.value_counts()


# In[ ]:


NFL_Punts[NFL_Punts.Primary_Partner_Activity_Derived == "Blocking"].Position.value_counts()


# In[ ]:


NFL_Punts[NFL_Punts.Primary_Partner_Activity_Derived == "Tackling"].Position.value_counts()


# In[ ]:


NFL_Punts[NFL_Punts.Primary_Partner_Activity_Derived == "Tackled"].Position.value_counts()


# The Two primary Impact Types that causes the most concussions during punt plays in the 2016 and 2017 seasons were Helmet ot Body and Helmet to Helmet impacts.
# 
# Among Helmet to Body Impacts, 10 of 17 happened during Tackling actions (Tackling or being Tackled)
# Among Helmet to Helmet Impacts, 9 of 17 happened during Blocking actions (Blocking or being Blocked)

# In[ ]:


NFL_Punts[NFL_Punts.Play_Type == "Punt"].Primary_Impact_Type.value_counts()


# In[ ]:


NFL_Punts.groupby(NFL_Punts.Primary_Impact_Type).Primary_Partner_Activity_Derived.value_counts()


# **Conclusion:****
# 
# 
# Based on the data above, I would like to suggest three preventable theories to injuries during Punt Plays:
# 
# 1. Players Selected to play during pre-season games are often less experienced than players selected to play during regular season games. As a result, they are less aware of how to protect themselves and suffer potentially avoidable concussions.
# 
# 2. Games including punt plays that result in a player suffering a concussion are more likely to occur toward the end of the regular season (does not extend into the Post Season). The 53-man roster adds strain on capable players as the season progresses and more player are added to the Injured reserve. The non IR players allowed to suit up is limited to 46 regardless of positions on IR
# 
# 3. Traditionally offensive players (i.e. WR) suffer injury during Tackling where they do not often Tackle in the Offensive Phase of the game. Also Traditionally Defensive players (i.e. CB and LBs) suffer injury during Blocking plays where they do not often block in the Defensive Phase of the game
# 
# 
# 

# ***Suggested Rule Changes:***
# 1. Pre-Season should be limited to 2 Games and the Regular season should be extended to 19 weeks.
# 2. The 53 Man roster should be increased to 56 player and the number of Suited players on Game Day should be increased to 49. In order to prevent Special teams injuries, Two new Positions should be added, Special Lineman and Designated Gunner.
# 3. The NFL should develop a training program for individuals eligible who participate in Special teams plays on Tackling and Blocking, with special emphasis on skill moves not often used in their normal phase of play.
