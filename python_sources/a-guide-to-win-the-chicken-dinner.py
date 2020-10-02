#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt 
import warnings, time, gc
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)

color = sns.color_palette("Set2")
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")

train = pd.read_csv("../input/train.csv")


# In[ ]:


import random

def generate_color():
    color = "#{:02x}{:02x}{:02x}".format(*map(lambda x: random.randint(0, 255), range(3)))
    return color


# ### Overview of Dataset

# In[ ]:


train.head()


# In[ ]:


train.describe()


# 

# In[ ]:


groups = train["groupId"].unique()
matches = train["matchId"].unique()


# * There are $1888732$ groups in the dataset
# * This dataset contains $47734$ matches 

# #### A Peak of a Match

# In[ ]:


sample_match = train[train["matchId"] == matches[0]]
sample_match.head(10)


# * Teams have equal win rate but win points are vary

# #### Sample Winner

# In[ ]:


sample_match_winner = sample_match[sample_match["winPlacePerc"] == max(sample_match["winPlacePerc"])]
sample_match_winner


# ## EDA
# 
# There are three modes in PUBG:
# * Solo (100 teams, each team has only one player)
# * Duo (50 teams, each team has two players)
# * Squad (less than 50 teams as each team has 2 to 4 players)
# 
# We will analyze this dataset in three separated sections (Solo, Duo, Squad) since different modes have different battle styles/strategies.

# ## Solo

# In[ ]:


solo_match = train[train["numGroups"] == 100]


# In[ ]:


solo_match_id = solo_match["matchId"].unique()
print("There are {} solo matches.".format(len(solo_match_id)))


# #### A peak of solo winners

# In[ ]:


solo_winners = solo_match[solo_match["winPlacePerc"] == 1.0]
solo_winners


# * There are $3$ winner players helped enemies
# * Some winner players are not even an MVP

# ### Win Points v.s. Win Place Percentage

# In[ ]:


trace = go.Scatter(x = solo_match["winPoints"],
                   y = solo_match["winPlacePerc"],
                   mode = "markers")
layout = dict(title = "Solo Match: Win Points v.s. Win Place Percentage",
              xaxis = dict(title = "Win Points"),
              yaxis = dict(title = "Win Place Percentage"))
iplot(dict(data = [trace], layout = layout))


# In[ ]:


temp = solo_match[solo_match["kills"] == 0]
trace1 = go.Scatter(x = temp["winPoints"],
                    y = temp["winPlacePerc"],
                    mode = "markers",
                    name = "No Kills")

temp = solo_match[(0 < solo_match["kills"]) & (solo_match["kills"] <= 2)]
trace2 = go.Scatter(x = temp["winPoints"],
                    y = temp["winPlacePerc"],
                    mode = "markers",
                    name = "1-2 Kills")

temp = solo_match[(2 < solo_match["kills"]) & (solo_match["kills"] <= 5)]
trace3 = go.Scatter(x = temp["winPoints"],
                    y = temp["winPlacePerc"],
                    mode = "markers",
                    name = "3-5 Kills")

temp = solo_match[(5 < solo_match["kills"]) & (solo_match["kills"] <= 10)]
trace4 = go.Scatter(x = temp["winPoints"],
                    y = temp["winPlacePerc"],
                    mode = "markers",
                    name = "6-10 Kills")


temp = solo_match[10 < solo_match["kills"]]
trace5 = go.Scatter(x = temp["winPoints"],
                    y = temp["winPlacePerc"],
                    mode = "markers",
                    name = "More Than 10 Kills")
layout = dict(title = "Win Points v.s. Win Place Percentage (separated by kills)",
              xaxis = dict(title = "Win Points"),
              yaxis = dict(title = "Win Place Percentage"))
iplot(dict(data = [trace1, trace2, trace3, trace4, trace5], layout = layout))


# In[ ]:


temp = solo_match[solo_match["killPlace"] >= 75]
trace1 = go.Scatter(x = temp["winPoints"],
                    y = temp["winPlacePerc"],
                    mode = "markers",
                    name = "100th - 75th Player")

temp = solo_match[(50 <= solo_match["killPlace"]) & (solo_match["killPlace"] < 75)]
trace2 = go.Scatter(x = temp["winPoints"],
                    y = temp["winPlacePerc"],
                    mode = "markers",
                    name = "74th - 50th Players")

temp = solo_match[(25 <= solo_match["killPlace"]) & (solo_match["killPlace"] < 50)]
trace3 = go.Scatter(x = temp["winPoints"],
                    y = temp["winPlacePerc"],
                    mode = "markers",
                    name = "50th - 25th Player")

temp = solo_match[(4 <= solo_match["killPlace"]) & (solo_match["killPlace"] < 25)]
trace4 = go.Scatter(x = temp["winPoints"],
                    y = temp["winPlacePerc"],
                    mode = "markers",
                    name = "25th - 4th Player")


temp = solo_match[solo_match["killPlace"] < 4]
trace5 = go.Scatter(x = temp["winPoints"],
                    y = temp["winPlacePerc"],
                    mode = "markers",
                    name = "First 3 Player")
layout = dict(title = "Win Points v.s. Win Place Percentage (separated by kill place)",
              xaxis = dict(title = "Win Points"),
              yaxis = dict(title = "Win Place Percentage"))
iplot(dict(data = [trace1, trace2, trace3, trace4, trace5], layout = layout))


# * Killing more enemies don't obtain more win win points
# * Longer time stay in the battle field provide higher chance to win

# ### Killpoints v.s. Win Place Percentage

# In[ ]:


trace = go.Scatter(x = solo_match["killPoints"],
                   y = solo_match["winPlacePerc"],
                   mode = "markers")
layout = dict(title = "Solo Match: Kill Points v.s. Win Place Percentage",
              xaxis = dict(title = "Kill Points"),
              yaxis = dict(title = "Win Place Percentage"))
iplot(dict(data = [trace], layout = layout))


# In[ ]:


temp = solo_match[solo_match["kills"] == 0]
trace1 = go.Scatter(x = temp["killPoints"],
                    y = temp["winPlacePerc"],
                    mode = "markers",
                    name = "No Kills")

temp = solo_match[(0 < solo_match["kills"]) & (solo_match["kills"] <= 2)]
trace2 = go.Scatter(x = temp["killPoints"],
                    y = temp["winPlacePerc"],
                    mode = "markers",
                    name = "1-2 Kills")

temp = solo_match[(2 < solo_match["kills"]) & (solo_match["kills"] <= 5)]
trace3 = go.Scatter(x = temp["killPoints"],
                    y = temp["winPlacePerc"],
                    mode = "markers",
                    name = "3-5 Kills")

temp = solo_match[(5 < solo_match["kills"]) & (solo_match["kills"] <= 10)]
trace4 = go.Scatter(x = temp["killPoints"],
                    y = temp["winPlacePerc"],
                    mode = "markers",
                    name = "6-10 Kills")


temp = solo_match[10 < solo_match["kills"]]
trace5 = go.Scatter(x = temp["killPoints"],
                    y = temp["winPlacePerc"],
                    mode = "markers",
                    name = "More Than 10 Kills")
layout = dict(title = "Kill Points v.s. Win Place Percentage (separated by kills)",
              xaxis = dict(title = "Kill Points"),
              yaxis = dict(title = "Win Place Percentage"))
iplot(dict(data = [trace1, trace2, trace3, trace4, trace5], layout = layout))


# * Players who eliminate $0-5$ enemies get about $1,000$ kill points if no headshot kills

# ### Kills v.s. Headshot Kills

# In[ ]:


trace = go.Scatter(x = solo_match["kills"],
                   y = solo_match["headshotKills"],
                   mode = "markers")
layout = dict(title = "Kills v.s. Headshot Kills",
              xaxis = dict(title = "Num of Kills"),
              yaxis = dict(title = "Num of Headshot Kills"))
iplot(dict(data = [trace], layout = layout))


# * Skilled playeres tend to aim on heads

# ### Kills v.s. Damage Dealt

# In[ ]:


trace = go.Scatter(x = solo_match["kills"],
                   y = solo_match["damageDealt"],
                   mode = "markers",
                   marker = dict(color = "red"))
layout = dict(title = "Kills v.s. Damage Dealt",
              xaxis = dict(title = "Num of Kills"),
              yaxis = dict(title = "Damage Dealt"))
iplot(dict(data = [trace], layout = layout))


# * Without loss of generality, more kills give more damage dealt.

# ### Kills v.s. Weapons Acquired

# In[ ]:


trace = go.Scatter(x = solo_match["kills"],
                   y = solo_match["weaponsAcquired"],
                   mode = "markers")
layout = dict(title = "Kills v.s. Weapons Acquired",
              xaxis = dict(title = "Kills"),
              yaxis = dict(title = "Weapons Acquired"))
iplot(dict(data = [trace], layout = layout))


# * There are weapon-collectors
# * Skilled players acquire less weapons. We assume that they prefer some specific weapons

# ### Boosts/Heals v.s. Kills

# In[ ]:


fig = tools.make_subplots(rows = 1, cols = 2, subplot_titles = ["Boosts v.s. Kills", 
                                                                "Heals v.s. Kills"])

trace1 = go.Scatter(x = solo_match["boosts"],
                    y = solo_match["kills"],
                    mode = "markers",
                    marker = dict(color = "blue"))
trace2 = go.Scatter(x = solo_match["heals"],
                    y = solo_match["kills"],
                    mode = "markers",
                    marker = dict(color = "red"))

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig["layout"]["xaxis1"].update(title = "Num of Boost Items")
fig["layout"]["xaxis2"].update(title = "Num of Healing Items")
fig["layout"]["yaxis1"].update(title = "Kills")
fig["layout"].update(height = 600, width = 800)

iplot(fig)


# In Solo mode
# * As players eliminate more enemies, they utilize more boost items
# * On the other hand, players that elimiate less enemies utilize more healing items

# ### Boosts v.s. Swim/Walk Distance

# In[ ]:


import plotly.figure_factory as ff

hist1 = solo_match["swimDistance"]
hist2 = solo_match["walkDistance"]
hist_data = [hist1, hist2]

group_labels = ["Swim Distance", "Walk Distance"]
fig = ff.create_distplot(hist_data, group_labels, bin_size = 1000, curve_type = "normal")
fig["layout"].update(title = "Walk/Swim Distance Distplot")
iplot(fig)


# In[ ]:


trace = go.Scatter(x = solo_match["boosts"],
                   y = solo_match["swimDistance"] + solo_match["walkDistance"],
                   mode = "markers",
                   marker = dict(color = "orange"))
layout = dict(title = "Boosts v.s. Total Distance",
              xaxis = dict(title = "Num of Boost Items"),
              yaxis = dict(title = "Total Distance"))
iplot(dict(data = [trace], layout = layout))


# ### Ride Distance v.s. Vehicle Destroys

# In[ ]:


trace = go.Scatter(x = solo_match["vehicleDestroys"],
                   y = solo_match["rideDistance"],
                   mode = "markers",
                   marker = dict(color = "black"))
layout = dict(title = "Vehicle Destroys v.s. Ride Distance",
              xaxis = dict(title = "Vehicle Destroys"),
              yaxis = dict(title = "Ride Distance"))
iplot(dict(data = [trace], layout = layout))


# * Surprisingly, there is no single vehible being destroyed by players in solo mode.

# ## Duo

# In[ ]:


duo_match = train[(train["numGroups"] == 50) & (train["maxPlace"] == 50)]


# In[ ]:


duo_match.head()


# ### A Peak of a Winner

# In[ ]:


duo_winners = duo_match[duo_match["winPlacePerc"] == 1.0]
sample_duo_winner = duo_winners[duo_winners["groupId"] == min(duo_winners["groupId"])]
sample_duo_winner


# ### Duo: Win Points v.s. Win Place Percentage

# In[ ]:


avg_points = duo_match.groupby(["groupId", "matchId"])[["winPoints", "winPlacePerc"]].mean()
trace = go.Scatter(x = avg_points["winPoints"],
                   y = avg_points["winPlacePerc"],
                   mode = "markers")
layout = dict(title = "Avg Win Points v.s. Win Place Percentage",
              xaxis = dict(title = "Avg Win Points"),
              yaxis = dict(title = "Win Place %"))
iplot(dict(data = [trace], layout = layout))


# * Win points in duo mode are calculated in the same method in solo mode

# ### Duo: Kills v.s. Win Place Percentage

# In[ ]:


total_kills = duo_match.groupby(["groupId", "matchId"])["kills"].sum()


# In[ ]:


temp = pd.merge(avg_points, total_kills.to_frame(), on = ["groupId", "matchId"], how = "left")
trace = go.Scatter(x = temp["kills"],
                   y = temp["winPlacePerc"],
                   mode = "markers")
layout = dict(title = "Duo: Total Kills v.s. Win Place %",
              xaxis = dict(title = "Total Kills"),
              yaxis = dict(title = "Win Place %"))
iplot(dict(data = [trace], layout = layout))


# * Surprisingly, there are teams won the games without any enemy eliminations
# * More enemy killings increase the winning chance

# In[ ]:


total_headshot = duo_match.groupby(["groupId", "matchId"])["headshotKills"].sum()


# In[ ]:


temp = pd.merge(total_kills.to_frame(), total_headshot.to_frame(), on = ["groupId", "matchId"], how = "left")
trace = go.Scatter(x = temp["kills"],
                   y = temp["headshotKills"],
                   mode = "markers")
layout = dict(title = "Duo: Num of Headshot Kills over Total Kills")
iplot(dict(data = [trace], layout = layout))


# * Skilled players are good at aimming at head

# In[ ]:





# In[ ]:





# ## To Be Continued...
