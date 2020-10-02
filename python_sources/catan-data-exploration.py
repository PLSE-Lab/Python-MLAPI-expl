#!/usr/bin/env python
# coding: utf-8

# # Catan Data Exploration
# 
# **Rand Seay**  
# Authored: *September 4th, 2016*  
# Updated: *November 8th, 2018*
# 
# Having some fun with the data. I focus on OP's data because I was interested in his habits and their ramifications. It is natural to assume his opponents varied.
# 
# ## Outline
# - How did OP (Original Poster) preform?
#   - Win percentage?
#   - Favorite starting resources?
#   - Starting chance?
#   - Exploring OP card gain
#   - Exploring OP card loss
#   - Exploring OP's trade data
#   - How did cards affect points?
# - What did the dice do?

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Import pandas for data processing
import pandas as pd

# Import seaborn for visualizations
import seaborn as sns

# "Take care" of warnings
import warnings
warnings.filterwarnings('ignore')

# Load the data into a dataframe
catan = pd.read_csv("../input/catanstats.csv")

# Checkout the data
catan.head()


# ## How did OP (Original Poster) perform?

# In[ ]:


# Get OP's data (Making a copy)
op_data = catan.loc[catan["me"] == 1].copy()

# Get OP's points in a new dataframe
op_points = op_data[["gameNum","points"]]

# Plot OP's points with seaborn
sns.regplot(x="gameNum", y="points", data=op_points).set(xlim=(0,50), ylim=(0,14))


# ### Win Percentage?

# In[ ]:


# Get OP's wins (games where he achieved 10 or more points)
op_wins = op_points.loc[op_points["points"] >= 10]

# Get OP's win percentage
"OP's Win Percentage is {}%... Not bad!".format(len(op_wins.index)/len(op_data.index)*100)


# In[ ]:


# Let's try another plot, but this time visually incorporating wins

# Add colors to the dataframe
op_data["color"] = ["#96CA2D" if x >= 10 else "#193441" for x in op_data["points"]]

# New plot
sns.regplot(x="gameNum", y="points", data=op_points, scatter_kws={"facecolors": op_data["color"]}).set(xlim=(0,50), ylim=(0,14))


# In[ ]:


# OP's points compared with total card gain
sns.regplot(x="totalGain", y="points", data=op_data, scatter_kws={"facecolors": op_data["color"]})


# While total cards gained can help you win, the data demonstrates that there are more factors involved.
# 
# ### Favorite starting resources?

# In[ ]:


# Did OP favor starting on a particular resource?

# Get starting settlement data
# op_starts = op_data[list(range(15,27))]
op_starts = op_data.iloc[:, list(range(15,27))]

# Get the resource columns
op_start_types = op_starts[op_starts.columns[1::2]]

# Stack the columns and count the values
op_start_totals = pd.DataFrame(op_start_types.stack().value_counts(), columns=["total"])
op_start_totals["resource"] = op_start_totals.index

# Resource Key
RES_KEY = {
    "W":"Wheat",
    "S":"Sheep",
    "O":"Ore",
    "C":"Clay",
    "L":"Lumber",
    "D":"Desert",
    "3G":"3:1 Port",
    "2W":"W Port",
    "2S":"S Port",
    "2O":"O Port",
    "2C":"C Port",
    "2L":"L Port",
}

# Replace resources with names
op_start_totals.replace({"resource": RES_KEY}, inplace=True)

# Check out the data
op_start_totals


# In[ ]:


# Plot
sns.barplot(x=op_start_totals["resource"], y=op_start_totals["total"])


# Wheat and sheep were the favorites, and clay and lumber were much lower than I would have thought. Not expecting that.
# 
# ### Starting chance?

# In[ ]:


# Get the chances from the starting settlement data
op_start_chance = op_starts[op_starts.columns[0::2]]

# Add starting chance to OP's data
op_data["startChance"] = op_start_chance.sum(axis=1)

# Plot starting chance and points
sns.JointGrid(x="startChance", y="points", data=op_data).plot(sns.regplot, sns.distplot)

# Plot starting chance and production
sns.JointGrid(x="startChance", y="production", data=op_data).plot(sns.regplot, sns.distplot)


# As a player, one would hope higher starting chance would lead to better production, but the data isn't terribly strong.
# 
# ### Exploring OP card gain

# In[ ]:


# Plot production and total gain together
sns.JointGrid(x="totalGain", y="production", data=op_data).plot(sns.regplot, sns.distplot)

# Plot cards gained from the robber and total gain together
sns.JointGrid(x="totalGain", y="robberCardsGain", data=op_data).plot(sns.regplot, sns.distplot)


# Production obviously makes up the bulk of total cards gained, but the robber factors in more than I would have thought.
# 
# ### Exploring OP card loss

# In[ ]:


# Plot tribute and total losses together
sns.JointGrid(x="totalLoss", y="tribute", data=op_data).plot(sns.regplot, sns.distplot)

# Plot cards lost to the robber and total losses together
sns.JointGrid(x="totalLoss", y="robberCardsLoss", data=op_data).plot(sns.regplot, sns.distplot)


# ### Exploring OP's trade data

# In[ ]:


# Plot trade gain and total gain together
sns.JointGrid(x="totalGain", y="tradeGain", data=op_data).plot(sns.regplot, sns.distplot)

# Plot trade loss and total loss together
sns.JointGrid(x="totalLoss", y="tradeLoss", data=op_data).plot(sns.regplot, sns.distplot)

# Plot trade gain and trade loss together
sns.JointGrid(x="tradeLoss", y="tradeGain", data=op_data).plot(sns.regplot, sns.distplot)


# In[ ]:


### How did cards affect points?


# In[ ]:


# Plot total available cards and points together
sns.JointGrid(x="totalAvailable", y="points", data=op_data).plot(sns.regplot, sns.distplot)

# Plot cards lost to the robber and points together
sns.JointGrid(x="robberCardsLoss", y="points", data=op_data).plot(sns.regplot, sns.distplot)


# In games where OP won, I would have expected him to be more of a target of the robber, but the data showed this was inconsistent.
# 
# ## What did the dice do?

# In[ ]:


# Grab the dice data from the main dataframe
dice_data = pd.DataFrame(catan.iloc[:, list(range(4,15))].sum(), columns=["totals"])
dice_data["rolls"] = range(2,13)

# Plot
sns.barplot(x=dice_data["rolls"], y=dice_data["totals"])


# The dice rolls present as expected. Thanks for looking!
