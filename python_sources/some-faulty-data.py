#!/usr/bin/env python
# coding: utf-8

# Here, I will list down all data anomalies I find and hopefully find an explanation for these.

# In[ ]:


import pandas as pd


# In[ ]:


games       = pd.read_csv("../input/games.csv")
game_events = pd.read_csv("../input/game_events.csv")


# ## Number of Players
# 
# Terra Mystica is a 2-5 player game so I don't know why there are 1, 6, and 7 player games in the dataset. Maybe they just wanted to test it out?

# In[ ]:


games["player_count"].value_counts()


# ## First Turn Temples and Strongholds
# 
# This shouldn't be possible as it would require a trading post to already be on the board.

# In[ ]:


game_events[
     game_events["event"].isin(["upgrade:SH", "upgrade:TE"]) &
    (game_events["round"] == 1) &
    (game_events["turn"]  == 1) 
].head()


# Looking closely at that first example there, both upgrading to trading post and temple got recorded as turn 1 moves. This continues as the upgrade to Santuary shows up as a turn 2 move.
# 
# No action (or any event) was recorded for turn 4 which is weird since the player doesn't pass until turn 6.

# In[ ]:


game_events[
    (game_events["game"]    == "0512") &
    (game_events["faction"] == "chaosmagicians") &
    (game_events["round"]    == 1)
]


# ## That's it so far
# 
# Will be updating as soon as I find more.
