#!/usr/bin/env python
# coding: utf-8

# A central dilemma in Terra Mystica is that of the choice between improving your economy versus gaining victory points. This dilemma shows up in different parts of the game: in power leeching where you give up victory points in order to gain power; in round score tiles where you build structures not just taking into account how it affects your economy but also how much vp you'll earn; in the bonus tiles which may not only give you resources but can also give you points.
# 
# This dilemma is also present in favor tiles. There are favor tiles that give you points and there are those that give you an income. In this notebook, we'll focus on the most popular favor tile: Earth1 (FAV11).
# 
# ![](https://cf.geekdo-images.com/original/img/YkzfHXct5LMiZ8OD8RJExU3i6Ls=/0x0/pic1509044.jpg)
# [Image Source](https://boardgamegeek.com/thread/900202/terra-mystica-game-1-onetonmee-play-forum)

# In[1]:


import numpy  as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams["figure.figsize"] = (20, 7)


# In[2]:


games          = pd.read_csv("../input/games.csv")
game_events    = pd.read_csv("../input/game_events.csv")
game_factions  = pd.read_csv("../input/game_factions.csv")
score_tiles    = pd.read_csv("../input/game_scoring_tiles.csv")


# In[26]:


FAV_SORT_ORDER  = [ "FAV" + str(num) for num in range(1, 13) ]

faction_palette = {
    "acolytes":       "#ff6103",
    "alchemists":     "black",
    "auren":          "green",
    "chaosmagicians": "red",
    "cultists":       "brown",
    "darklings":      "black",
    "dragonlords":    "#ff6103",
    "dwarves":        "gray",
    "engineers":      "gray",
    "fakirs":         "yellow",
    "giants":         "red",
    "halflings":      "brown",
    "icemaidens":     "#9adde9",
    "mermaids":       "blue",
    "nomads":         "yellow",
    "riverwalkers":   "#faebd7",
    "shapeshifters":  "#faebd7",
    "swarmlings":     "blue",
    "witches":        "green",
    "yetis":          "#9adde9"
}


# ## The Popularity of FAV11
# 
# FAV11 is the most taken favor tile in the game by a huge margin. 2nd to it is FAV10, another vp-giving favor tile. Their effectiveness at racking up points (particularly of FAV11) have long been the subject of discussions on boardgamegeek:
# 
# 
# * https://boardgamegeek.com/thread/1351214/are-fav10-and-fav11-too-strong
# * https://boardgamegeek.com/thread/1827131/possible-win-without-earth-1-favor-tile
# 

# In[4]:


favor_events = game_events[
     game_events["event"].str.startswith("favor:FAV") &
    (game_events["faction"] != "all")
].copy()

favor_events["tile"] = favor_events["event"].str[6:]
favor_events.drop(columns=["event", "num"], inplace=True)
favor_events.head()


# In[5]:


def barplot_favor_counts(series, title, ax=None):
    if ax is None:
        f, ax = plt.subplots(figsize=(20, 5))

    sns.set_color_codes("muted")
    sns.barplot(
        x=series.values,
        y=series.index,
        order=FAV_SORT_ORDER,
        color="b",
        ax=ax
    )
    ax.set_title(title)
    
favor_counts = favor_events["tile"].value_counts()
barplot_favor_counts(favor_counts, "Number of times each favor tile has been taken across all games")


# A common strategy is to get FAV11 as soon as possible in order to get points for all dwellings built in the game. Delaying FAV11 costs 2 points per dwelling built before acquiring it. As we see below, 51% of all favor tiles taken in the first round is FAV11 and 61% of the times it was taken was done on the first round.
# 
# Contrast this with FAV10 which is usually taken much later in the game.

# In[6]:


f, axs = plt.subplots(ncols=2)

fav_count_by_round =     favor_events.groupby(["tile", "round"])         .size()                                     .reset_index(name="count")                  .pivot("tile", "round", "count")            .reindex(index=FAV_SORT_ORDER)
        
fav_percent_by_round = fav_count_by_round.div(fav_count_by_round.sum(axis=0), axis=1)
fav_percent_by_favor = fav_count_by_round.div(fav_count_by_round.sum(axis=1), axis=0)

for ax, df, title in [
    (axs[0], fav_percent_by_round, "Distribution of tiles taken each round \n ie, each column sums up to 100%"),
    (axs[1], fav_percent_by_favor, "Distribution of rounds each tile was taken \n ie, each row sums up to 100%")
]:
    sns.heatmap(df, annot=True, fmt=".0%", ax=ax)
    ax.set(title=title, xlabel="Round", ylabel="")

plt.show()


# The round 0 data seen here is of course the Icemaiden, a faction that starts the game with 1 favor tile. Here, income tiles are more popular which makes sense since they get 1 more round to earn from it. With that said, FAV11 is still the number 1 choice when grabbing a favor tile in round 1.

# In[7]:


icemaiden_favor_counts =     favor_events[
        (favor_events["faction"] == "icemaidens") &
        (favor_events["round"]   == 1)
    ]["tile"].value_counts()

barplot_favor_counts(icemaiden_favor_counts, "Ice Maiden Round 1 Favor Tiles")


# ## The Race to Get FAV11
# 
# There are only 3 copies each of FAV5 to FAV12 and only 1 copy of FAV1 to FAV4, the 3-advance cult track favor tiles. This means that, in games with more than 3 players, these favor tiles can run out with 1 or 2 players (willingly or not) missing out on them.
# 
# The fear of missing out on FAV11 influences 4 and 5 player games so much that getting FAV11 as soon as possible (round 1 turn 2) is a play many make.

# In[36]:


fav11_round_counts = fav11_events.groupby(["game", "round"]).size().reset_index(name="count")
fav11_round_counts["cum_count"] = fav11_round_counts.groupby(["game"])["count"].apply(lambda x: x.cumsum())

fav11_gone = fav11_round_counts[fav11_round_counts["cum_count"] == 3]
fav11_gone_round_counts = fav11_gone["round"].value_counts()
fav11_gone_round_counts["Never"] = len(games) - sum(fav11_gone_round_counts)
fav11_gone_round_counts


# In[8]:


player_order = game_events[
    game_events["event"].str.startswith("order") &
    (game_events["faction"] != "all")
].copy()

player_order["player_order"] = player_order["event"].str[6:].apply(int)
player_order.drop(columns=["event", "num", "turn"], inplace=True)
player_order.head()


# In[9]:


def add_player_count_and_order(df):
    return pd.merge(
        pd.merge(df, games[["game", "player_count"]], on="game"),
        player_order,
        on=["game", "faction", "round"]
    )

fav11_events = favor_events[favor_events["tile"] == "FAV11"]
turn2_fav11  = fav11_events[(fav11_events["round"] == 1) & (fav11_events["turn"] == 2)]
turn2_fav11  = add_player_count_and_order(turn2_fav11)
turn2_fav11.head()


# In[10]:


total_by_player_count = games["player_count"].value_counts().reset_index(name="total_by_player_count")
total_by_player_count.rename(columns={ "index": "player_count" }, inplace=True)
total_by_player_count


# In[11]:


def calc_by_player_count_and_order(df):
    counts = df.groupby(["player_count", "player_order"])         .size()                                   .drop([1, 6, 7], errors="ignore")         .reset_index(name="count")
    
    counts = pd.merge(counts, total_by_player_count, on="player_count")
    counts["percent_by_player_count"] = counts["count"] / counts["total_by_player_count"]
    return counts

turn2_fav11_counts = calc_by_player_count_and_order(turn2_fav11)
turn2_fav11_counts


# In[12]:


def barplot_percentages_by_player_order(df, title, ax=None):
    if ax is None:
        ax = plt.subplot()

    sns.set_color_codes("muted")
    sns.barplot(
        data=df,
        x="player_count",
        y="percent_by_player_count",
        hue="player_order",
        ax=ax
    )
    ax.legend(loc="upper left", title="Player Order")
    ax.set(
        title=title,
        xlabel="Number of Players",
        ylabel="Percentage"
    )

barplot_percentages_by_player_order(
    turn2_fav11_counts,
    "Percentage of Players who went for Fav11 on Turn 2"
)


# That doesn't look right.
# 
# Firstly, higher percentage of players rush FAV11 on a 4-player game than on a 2 or 3 player game. That makes sense since there's a possibility of FAV11 running out for the former and there's enough for everyone in the latter. What doesn't make sense to me is the percentages drop for 5-players.
# 
# But what really baffles me here is how the first player always has a lower percentage. In a 4-player game, 28% of 2nd and 3rd players rush to secure FAV11 on turn 2 but only 12% of 1st players do. If FAV11 is all that great, why aren't more 1st players using their first-mover advantage to get FAV11 before anyone else does? They must be doing something much more valueable.

# In[13]:


turn2_favors = favor_events[
    (favor_events["round"] == 1) &
    (favor_events["turn"]  == 2)
]
turn2_favors = add_player_count_and_order(turn2_favors)
turn2_favors.head()


# In[14]:


turn2_temples = game_events[
    (game_events["round"] == 1) &
    (game_events["turn"]  == 2) &
    (game_events["event"]   == "upgrade:TE") &
    (game_events["faction"] != "all")
]
turn2_temples = add_player_count_and_order(turn2_temples)
turn2_temples.head()


# In[15]:


f, axs = plt.subplots(ncols=2)

turn2_favors_p1_counts = turn2_favors[
    turn2_favors["player_order"] == 1
]["tile"].value_counts()
barplot_favor_counts(turn2_favors_p1_counts, "Turn 2 Favor Tiles of Player 1", ax=axs[0])

barplot_percentages_by_player_order(
    calc_by_player_count_and_order(turn2_temples),
    "Percentage of Players who went for Temples on Turn 2",
    ax=axs[1]
)

plt.show()


# It's not that they don't get FAV11 when building temples - they do. It's that 1st players tend to not go for turn 2 temples. Why though?

# In[16]:


turn1_dwellings = game_events[
    (game_events["round"] == 1) &
    (game_events["turn"]  == 1) &
    (game_events["event"]   == "build:D") &
    (game_events["faction"] != "all")
]
turn1_dwellings = add_player_count_and_order(turn1_dwellings)

barplot_percentages_by_player_order(
    calc_by_player_count_and_order(turn1_dwellings),
    "Percentage of Players who went for Building Dwellings on Turn 1"
)


# This looks like it could explain it. While FAV11 is good, securing a location is more important for a lot of 1st players. 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ** wip **
