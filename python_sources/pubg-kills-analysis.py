#!/usr/bin/env python
# coding: utf-8

# # PUBG Kills Analysis Notebook 

# Player Unknown's Battlegrounds (PUBG), is a first person shooter game where the goal is to be the last player standing.  You are placed on a giant circular map that shrinks as the game goes on, and you must find weapons, armor, and other supplies in order to kill other players / teams and survive.  More can be read about the game [here](https://en.wikipedia.org/wiki/PlayerUnknown's_Battlegrounds#Mobile_versions).  In this notebook, I will be analyzing kill statistics in order to learn more about the game and make recommendations on winning strategies.  Leave a like and/or comment if you enjoy! :)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as r
import warnings
warnings.filterwarnings('ignore')


# ## Preparing the Data

# We have 5 total datasets that will later be concatinated, but for now, we will use this single dataset as a sample to avoid taking up a lot of memory.  This is approximately 20% of the data.

# In[ ]:


kills = pd.read_csv('../input/pubg-match-deaths/deaths/kill_match_stats_final_0.csv')


# In[ ]:


kills = kills.dropna(axis = 0, how = "any") #we will drop all rows with NA values


# In[ ]:


kills.head()


# In[ ]:


kills.info()


# ### Feature Engineering

# There are many additional variables that can be extracted here.

# #### Kill Distance

# We can use the pythagorean formula to get the distance between each killer and victim.

# In[ ]:


kills["kill_distance"] = ((kills["killer_position_x"] - kills["victim_position_x"]) ** 2 + (kills["killer_position_y"] - kills["victim_position_y"]) ** 2) ** (1/2)


# #### Placement Difference

# This variable will tell us the difference in placement between the killer and victim.  There are up to 100 players a game, so placement can range from 1st place to 100th place.
# 
# Note: This value can be negative because players are placed as a team.  For example, if player x kills player y, but player x's entire team is eliminated before player y's entire team, the placement difference for the given row will be negative.

# In[ ]:


kills["place_diff"] = kills["victim_placement"] - kills["killer_placement"]


# #### Winner Kills

# This variable will tell us the amount of kills achieved by the winner(s) of each game (1st place finishers), and will be constant along each match id.  There can multiple players in 1st place due to many of the games being team matches, so these kills are added together.

# In[ ]:


winners = dict(kills[kills["killer_placement"] == 1][["killer_name", "match_id"]].groupby("match_id").size())


# In[ ]:


kills["winner_kills"] = kills["match_id"].map(winners, kills["match_id"])


# We now have data that can be analyzed.

# In[ ]:


kills.head()


# ## Basic Distributions

# In[ ]:


def remove_outliers(kills, var):
    lower_bound = np.quantile(kills[var], 0.25) - (1.5 * (np.quantile(kills[var], 0.75) - np.quantile(kills[var], 0.25)))
    upper_bound = np.quantile(kills[var], 0.75) + (1.5 * (np.quantile(kills[var], 0.75) - np.quantile(kills[var], 0.25)))
    kills = kills[(kills[var] >= lower_bound) & (kills[var] <= upper_bound)]
    return kills

def distribution_cont(kills, var):
    if var == "kill_distance":
        kills = remove_outliers(kills, var)
    plt.hist(kills[var])
    plt.axvline(x = np.mean(kills[var]), c = "r")
    plt.text(0.7, 0.9, f"Mean: {round(np.mean(kills[var]), 2)}", transform = plt.gca().transAxes)
    plt.xlabel(f"{var}")
    plt.ylabel("Count")
    plt.title(f"Distribution of {var}")
    plt.show()
    
def basic_plot(kills, var):
    if var == "killed_by":
        grouped = kills.groupby(var).size().sort_values(ascending = False).head(10)
        plt.xticks(rotation = 90)
    else:
        grouped = kills.groupby(var).size().sort_values(ascending = False)
    plt.bar(grouped.index.values, grouped.values)
    plt.title(f"{var} Distribution")
    plt.show()


# ### Placement

# In[ ]:


[distribution_cont(kills, i) for i in ["killer_placement", "victim_placement", "place_diff"]]


# Killers are very often eliminated shortly after their victims (or sometimes they go on to win the game).

# ### Position and Distance

# In[ ]:


[distribution_cont(kills, i) for i in ["killer_position_x", "killer_position_y", "victim_position_x", "victim_position_y", "kill_distance"]]


# Outliers were removed from the kill distance data.

# ### Time

# In[ ]:


[distribution_cont(kills, i) for i in ["time"]]


# Note that this is a cumulative variable in seconds.

# ### Winner Kills

# In[ ]:


[distribution_cont(kills, i) for i in ["winner_kills"]]


# The average winning team kills about 10 players.

# ### Maps

# In[ ]:


basic_plot(kills, "map")


# Erangel is a far more common map than Miramar.

# ### Weapons / Causes of Death

# In[ ]:


basic_plot(kills, "killed_by")


# There are too many weapons/causes of death to represent on one graph, so only the 10 most common are shown.

# ## Mapping the Game

# In this section, we will plot sample points of killers and victims on both the Erangel and Miramar maps.  Points may completely overlap due to very close range kills.  This will help us understand the setting of the game, as we can visualize typical distances of kills.

# ### Erangel

# In[ ]:


erangel = kills[kills["map"] == "ERANGEL"]


# In[ ]:


erangel_sample = erangel.sample(n = 5, random_state = 1)


# In[ ]:


img = plt.imread("../input/pubg-match-deaths/erangel.jpg")
plt.subplots(figsize = (12,12))
plt.imshow(img,aspect='auto', extent=[0, 800000, 0, 800000])
plt.scatter(erangel_sample["killer_position_x"], erangel_sample["killer_position_y"], c = "r", s = 100)
plt.scatter(erangel_sample["victim_position_x"], erangel_sample["victim_position_y"], c = "g", s = 100)
plt.legend(["Killer", "Victim"], loc = "upper right")
plt.show()


# ### Miramar

# In[ ]:


miramar = kills[kills["map"] == "MIRAMAR"]


# In[ ]:


miramar_sample = miramar.sample(n = 5, random_state = 1)


# In[ ]:


img = plt.imread("../input/pubg-match-deaths/miramar.jpg")
plt.subplots(figsize = (12,12))
plt.imshow(img, aspect='auto', extent=[0, 800000, 0, 800000])
plt.scatter(miramar_sample["killer_position_x"], miramar_sample["killer_position_y"], c = "r", s = 100)
plt.scatter(miramar_sample["victim_position_x"], miramar_sample["victim_position_y"], c = "g", s = 100)
plt.legend(["Killer", "Victim"], loc = "upper right")
plt.show()


# ### Player Movements

# Now, let's track a given single player's journey through the game until their victory or point of death.  The function will take a player with atleast one kill as a parameter, as well as a match id to prevent different games with the same player from overlapping.

# In[ ]:


def get_player_movement(kills, match_id, player_name):
    game = kills[kills["match_id"] == match_id]
    player = game[(game["killer_name"] == player_name) | (game["victim_name"] == player_name)].sort_values(by = "time")
    m = player.iloc[0]["map"]
    place = int(player.iloc[0]["killer_placement"])
    kills = len(list(player[player["killer_name"] == player_name]["killer_name"]))
    killer_name = player.iloc[-1]["killer_name"]
    killed_by = player.iloc[-1]["killed_by"]
    
    if m == "ERANGEL":
        img = plt.imread("../input/pubg-match-deaths/erangel.jpg")
    else:
        img = plt.imread("../input/pubg-match-deaths/miramar.jpg")
    plt.subplots(figsize = (12,12))
    plt.imshow(img, aspect='auto', extent=[0, 800000, 0, 800000])
    plt.scatter(player["killer_position_x"].iloc[0:-2], player["killer_position_y"].iloc[0:-2], c = "g", s = 100)
    plt.scatter(player["victim_position_x"].iloc[-1], player["victim_position_y"].iloc[-1], c = "r", s = 100)
    if place != 1:
        plt.title(f"{player_name}'s Game (Place: #{place}, Kills: {kills}, Killed by {killer_name} with {killed_by})")
        plt.legend(["Kill", "Point of Death"], loc = "upper right")
    else:
        plt.title(f"{player_name}'s Game (Place: #{place}, Kills: {kills})")
        plt.legend(["Kill", "Last Kill"], loc = "upper right")


# In[ ]:


rand_match = kills.iloc[r.randint(0, 11908315)]


# In[ ]:


get_player_movement(kills, rand_match["match_id"], rand_match["killer_name"])


# In[ ]:


rand_match2 = kills.iloc[r.randint(0, 11908315)]


# In[ ]:


get_player_movement(kills, rand_match2["match_id"], rand_match2["killer_name"])


# In[ ]:


rand_match3 = kills.iloc[r.randint(0, 11908315)]


# In[ ]:


get_player_movement(kills, rand_match3["match_id"], rand_match3["killer_name"])


# ## Which Weapons are Most Common From Different Distances?

# This section will help us stategize which weapons a player should use when possible depending on the distance from their enemy.  We will get rid of outliers, many of which are likely due to glitches in the game.  
# 
# 

# In[ ]:


distances = remove_outliers(kills, "kill_distance")


# Here is another look at our kill distance distribution as a violin plot.  Most kills are very short distance, so these deaths will be split into two parts (very short distance and short distance).

# In[ ]:


sns.violinplot(data = distances, y = "kill_distance")
plt.title("Kill Distance Distribution")
plt.show()


# We will now create a function that plots the most lethal weapons in a given percentile range.  The first plot will show the raw frequency of weapons used within the range.  The second plot will show the difference between the overall proportion (all percentiles) of weapon usage and proportion for only the given percentile range.  
# 
# Note: The second plot is necessary as the first plot may reflect many weapons / causes of death that are common among the entirety of the data but not necessarily specific to the given percentile range.  The second plot tells us relative percentage which gives us an idea of the exclusively short, medium, or long range weapons.

# In[ ]:


def prop_list(lst):
    weapons = list(lst)
    weapon_props = []
    distinct_weapons = list(set(weapons))
    for i in distinct_weapons:
        counter = 0
        for j in weapons:
            if i == j:
                counter += 1
        weapon_props.append(counter / len(weapons))
    return dict(zip(distinct_weapons, weapon_props))


# In[ ]:


distances["weapon_prop"] = distances["killed_by"].map(prop_list(distances["killed_by"]), distances["killed_by"])


# In[ ]:


def get_weapons_range(distances, perc_low, perc_high):
    lower_quantile = np.quantile(distances["kill_distance"], perc_low)
    upper_quantile = np.quantile(distances["kill_distance"], perc_high)
    distances2 = distances[(distances["kill_distance"] >= lower_quantile) & (distances["kill_distance"] <= upper_quantile)]
    
    weapon_counts = distances2.groupby("killed_by").size().sort_values(ascending = False).head(10)
    
    distances2["weapon_prop_filtered"] = distances2["killed_by"].map(prop_list(distances2["killed_by"]), distances2["killed_by"])
    distances2["conditional_prob"] = (distances2["weapon_prop_filtered"] - distances2["weapon_prop"])
    distinct_props = distances2[["killed_by", "conditional_prob"]].drop_duplicates().sort_values(by = "conditional_prob", ascending = False).head(10)
    
    plt.subplots(figsize = (10,6))
    plt.subplot(1,2,1)
    sns.barplot(weapon_counts.index.values, weapon_counts.values)
    plt.xticks(rotation = 90)
    plt.title(f"By Frequency ({perc_low} to {perc_high} Percentile)")
    plt.subplot(1,2,2)
    sns.barplot(distinct_props["killed_by"], distinct_props["conditional_prob"])
    plt.xticks(rotation = 90)
    plt.title(f"By Proportion Diff ({perc_low} to {perc_high} Percentile)")
    plt.xlabel(None)
    plt.ylabel(None)
    plt.show()    


# #### Very Short Distance Kills

# In[ ]:


get_weapons_range(distances, 0, 0.05)


# #### Short Distance Kills

# In[ ]:


get_weapons_range(distances, 0.05, 0.25)


# #### Medium Distance Kills

# In[ ]:


get_weapons_range(distances, 0.25, 0.75)


# #### Long Distance Kills

# In[ ]:


get_weapons_range(distances, 0.75, 1)


# #### Insights

# * Very short distance weapons: punching, grenades, cars
# * Short distance weapons: S1897, S686, S12K, Micro UZI
# * Medium distance weapons: M416, UMP9, SCAR-L, AKM
# * Long distance weapons: Kar98k, Mini 14, SKS
# * Overall common weapons:  AKM, SCAR-L, M416, M16A4, UMP9 (Also depicted in previous section)
# * "Down and Out" is not a weapon, though it is the most common cause of death overall, especially at medium and long distance ranges.
# * The most common overall weapons are unsurprisingly medium distance weapons, as this covers the typical distance range of a kill.

# ## Which Weapons are Most Common in Different Points of the Game?

# This section will be similar to the previous one, though we will compare weapon use to the time variable instead of the distance variable.

# In[ ]:


kills.head()


# First, a closer look at the distribution of time (in seconds), which is also skewed towards lower time values.  This is due to every game starting at 0 and counting up.

# In[ ]:


sns.violinplot(data = kills, y = "time")
plt.title("Time Distribution")
plt.show()


# This function will graph the 10 most common weapons within a given time range.

# In[ ]:


def weapons_time(kills, min_time, max_time = max(list(kills["time"]))):
    
    time_range = kills[(kills["time"] >= min_time) & (kills["time"] <= max_time)]
    grouped_time = time_range.groupby("killed_by").size().sort_values(ascending = False)
    top_10 = grouped_time.head(10)
    deaths = sum(list(grouped_time.values))
    
    sns.barplot(top_10.index.values, top_10.values)
    plt.title(f"Top 10 Weapons ({min_time} to {max_time} sec.); {deaths} Total Deaths")
    plt.xticks(rotation = 90)
    plt.show()


# #### The First Minute

# In[ ]:


weapons_time(kills, 0, 60)


# There are hardly any deaths in the first minute of the game within the data, so lets expand our search to the first 3 minutes.

# #### The First 3 Minutes

# In[ ]:


weapons_time(kills, 0, 180)


# #### The Middle of the Game

# In[ ]:


weapons_time(kills, np.quantile(kills["time"], 0.25), np.quantile(kills["time"], 0.75))


# #### The End of the Game

# In[ ]:


weapons_time(kills, np.quantile(kills["time"], 0.9))


# #### Insights

# * Beginning of Game Weapons: Punching, M16A4, S1897
# * Middle of Game Weapons: M16, M16A4, AKM, SCAR-L
# * End of Game Weapons: M16, SCAR-L
# * The best weapons in the game are likely M16 and SCAR-L

# ### The Most Common Weapons on the Final Kills

# The time variable can only tell us so much about the later stages of the game because different games have various lengths.  This plot will give us an idea of the best weapons in the game.  They must be the best because players tend to accumlate better weapons as the game goes on and they kill more people.  The weapons used by 1st place finishers to take out 2nd place finishers must be quite powerful.

# In[ ]:


final_kills = kills[(kills["killer_placement"] == 1) & (kills["victim_placement"] == 2)]


# In[ ]:


grouped_final_kills = final_kills.groupby("killed_by").size().sort_values(ascending = False).head(10)


# In[ ]:


sns.barplot(grouped_final_kills.index.values, grouped_final_kills.values)
plt.xticks(rotation = 90)
plt.title("Top 10 Weapons For Final Kills")
plt.show()


# ## Do More Kills Improve Winning Chances?

# This question may seem self-explanatory as killing other players is positive, and being the last team standing is required to win the game.  However, it might not be that simple.  Many players in PUBG and similar video games try "camping", a strategy that involves hiding while other players take eachother out and lose health.  Campers usually stay in one spot, so they usually don't interact with many players.  It is a feasible idea that many players who play it safe ultimately come out on top, while many other players who are reckless and achieve many kills get themselves killed in the process.

# We can answer this question using our prop_list function from before, as each row represents a kill.

# In[ ]:


inc_wins = kills


# In[ ]:


place = prop_list(inc_wins["killer_placement"])


# In[ ]:


inc_wins["kill_prop"] = inc_wins["killer_placement"].map(place, inc_wins["killer_placement"])


# In[ ]:


place_df = inc_wins[["killer_placement", "kill_prop"]].drop_duplicates().sort_values(by = "killer_placement", ascending = False)


# In[ ]:


plt.scatter(place_df["kill_prop"], place_df["killer_placement"])
plt.plot(place_df["kill_prop"], place_df["killer_placement"])
plt.xlabel("Proportion of Kills")
plt.ylabel("Placement (1-100)")
plt.title("Overall Proportion of Kills by Placement")
plt.show()


# According to this very consistent pattern, more kills do strongly correlate with a higher placement, therefore camping is usually a bad strategy.  1st place finishers average twice the amount of kills as 2nd place finishers.  Interestingly, there exists a slight imperfection in the exponential pattern between about 10th place and 30th place.

# ### On Which Map is This Effect Greatest?

# In[ ]:


miramar_wins = inc_wins[inc_wins["map"] == "MIRAMAR"]
erangel_wins = inc_wins[inc_wins["map"] == "ERANGEL"]
place_m = prop_list(miramar_wins["killer_placement"])
place_e = prop_list(erangel_wins["killer_placement"])
miramar_wins["kill_prop"] = miramar_wins["killer_placement"].map(place_m, miramar_wins["killer_placement"])
erangel_wins["kill_prop"] = erangel_wins["killer_placement"].map(place_e, erangel_wins["killer_placement"])
place_df_m = miramar_wins[["killer_placement", "kill_prop"]].drop_duplicates().sort_values(by = "killer_placement", ascending = False)
place_df_e = erangel_wins[["killer_placement", "kill_prop"]].drop_duplicates().sort_values(by = "killer_placement", ascending = False)


# In[ ]:


plt.subplot(1,2,1)
plt.scatter(place_df_e["kill_prop"], place_df_e["killer_placement"])
plt.plot(place_df_e["kill_prop"], place_df_e["killer_placement"])
plt.xlabel("Proportion of Kills")
plt.title("Erangel")
plt.subplot(1,2,2)
plt.scatter(place_df_m["kill_prop"], place_df_m["killer_placement"])
plt.plot(place_df_m["kill_prop"], place_df_m["killer_placement"])
plt.xlabel("Proportion of Kills")
plt.title("Miramar")
plt.show()


# There is no significant difference here.

# ## To be continued? :)

# In[ ]:




