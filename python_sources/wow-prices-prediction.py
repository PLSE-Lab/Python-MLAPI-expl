#!/usr/bin/env python
# coding: utf-8

# # What is World of Wacraft?
# According to Wikipedia:
# 
# > World of Warcraft (WoW) is a massively multiplayer online role-playing game (MMORPG) released in 2004 by Blizzard Entertainment. It is the fourth released game set in the Warcraft fantasy universe. World of Warcraft takes place within the Warcraft world of Azeroth, approximately four years after the events at the conclusion of Blizzard's previous Warcraft release, Warcraft III: The Frozen Throne. The game was announced in 2001, and was released for the 10th anniversary of the Warcraft franchise on November 23, 2004. Since launch, World of Warcraft has had eight major expansion packs produced for it: The Burning Crusade, Wrath of the Lich King, Cataclysm, Mists of Pandaria, Warlords of Draenor, Legion, Battle for Azeroth and Shadowlands.
# 
# > World of Warcraft was the world's most popular MMORPG by player count of nearly 10 million in 2009. The game had a total of over a hundred million registered accounts by 2014. By 2017, the game had grossed over $9.23 billion in revenue, making it one of the highest-grossing video game franchises of all time. At BlizzCon 2017, a vanilla version of the game titled World of Warcraft Classic was announced, which planned to provide a way to experience the base game before any of its expansions launched. It was released in August 2019.
# 
# ### Why is it worth investigating?
# 
# The game World of Warcraft has already been a gold mine when it comes to scientific work (see [google scholar](https://scholar.google.com/scholar?hl=pl&as_sdt=0,5&q=world+of+warcraft)). From social studies to finance, the world created by Blizzard has given many information about how players behave with their created characters, modelling how they act in real life. In this notebook, I will explore the auction house prices driven by demand over a few months for raid supplies.
# 
# ### What is the meaning of all those strange words?
# 
# World of Wacraft as any other game has specific terminology for items that are present in the world. Here is a summary of the specific items and words that will be used while researching this notebook.
# 
# - **PvE** - Player versus Environment, gameplay that focuses on fighting computer generated enemies,
# - **PvP** - Player versus Player, gameplay that focuses on fighting other players,
# - **Party** - A group of 2-5 players, created for participating in PvE or PvP scenarios,
# - **Raid Group** - A group of 6-40 players divided in groups of 5, created for participating in PvE or PvP scenarios,
# - **Raid** - An instanced dungeon, with several bosses that, proves a challenge for a raid group,
# - **Flask** - One hour character stats buff that persists through death,
# - **Potion** - A short boost to stats, that increases damage, survivability etc,
# - **Food** - Long character stats buff that dissapears on death,
# - **Feast** - A table that provides food buffs, but can be used by multiple people in a group.
# - **Money** - The amount of money you have. It's divided into gold, silver and copper:
#    - 100 copper = 1 silver,
#    - 100 silver = 1 gold.
# - **Auction House (AH)** - Place where people sell their goods in-game. 
# 
# ### How was the data gathered?
# 
# The code for the auction house data gathering service can be previewed in the [lukzmu/wow-data-gather](https://github.com/lukzmu/wow-data-gather) repository. It is a Flask application hosted on Heroku, with a MySQL database to store the information. The application connects to the TradeSkill Master (TSM) API, and gathers selected data on raid consumables. The table was exported in csv, that is available to download on Kaggle or from the [github repository](https://github.com/lukzmu/data-science/tree/master/Other/World%20of%20Warcraft%20Auction%20House). At one point the Heroku server died, so you will have a gap in the gathered data. Sorry for that!
# 
# # Loading in the data
# 
# In this section we will load the data from the csv file and set a couple of things:
# 
# - Select `;` separator and select `id` as index,
# - `item_name` and `item_subclass` as categorical data,
# - Parse dates for `created_at` - this is the time the data was gathered from the AH,

# In[ ]:


# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


# In[ ]:


# Read the data and preview sample rows
wow = pd.read_csv(
    '../input/world-of-warcraft-auction-house-data/tsm_data.csv',
    index_col='id',
    parse_dates=['created_at'],
    dtype={
        'item_name': 'category',
        'item_subclass': 'category',
    },
)
display(wow.sample(5, random_state=42))
wow.info()


# # Data Cleaning
# 
# The good thing is that we don't have any null values. In this section we will do a couple of things tho:
# 
# - Remove `item_vendor_buy` and `item_vendor_sell` columns as they don't carry any relevant information,
# - Remove `item_market_value` as it's TSM related information,
# - Filter out items with `0` auctions at a specified time,
# - Remove outlier potions that someone put for horrendal prices (they would never sell anyway),
# - Update values to be in gold instead of copper, for readability (and noone really uses copper as their base),
# - Add "days since new raid" column to see if it influences the prices.

# In[ ]:


# Filter out empty rows
wow = wow[wow['item_num_auctions'] > 0]

# Remove irrelevant columns
wow.drop(
    columns=[
        'item_vendor_buy',
        'item_vendor_sell',
        'item_market_value',
    ],
    inplace=True
)

# Update copper to gold
wow['item_min_buyout'] = wow['item_min_buyout'] / 10000

# Remove outliers
bad_ids = [1884, 1903, 1922, 1941, 1960]
wow.drop(index=bad_ids, inplace=True)

# Add days since raid release
ETERNAL_PALACE_RELEASE = pd.Timestamp('2019.07.10')
NYALOTHA_RELEASE = pd.Timestamp('2020.01.21')
def calculate_days_since_new_raid(date):
    ep_release = ETERNAL_PALACE_RELEASE
    ny_release = NYALOTHA_RELEASE
    if date < ny_release:
        return (date - ep_release).days
    else:
        return (date - ny_release).days

wow['days_after_new_raid'] = wow.apply(
    lambda row: calculate_days_since_new_raid(row['created_at']),
    axis=1,
)


# In[ ]:


# Preview dataframe after cleaning
display(wow.sample(5, random_state=42))
wow.info()


# # Data Analysis
# Our dataset has 44702 non-null entries. The name of the item and its subclass are categorical values. The `item_min_buyout` is the value that people usually buy items at (with the buyout option, rather than actually doing auctions). The `item_quantity` is the number of items, while `item_num_auctions` is the number of auctions those items belong to (one auction can have multiple items).
# 
# We can preview the statistical numerics below. Due to the fact that we are comparing items of various subclasses, the minimum and maximum values will be quite different, we will deal with this when we go deeper into the analysis, but already we can see how consumables place on the auction house for the selected time period.

# In[ ]:


# Preview the dataset statistical numerics
wow.describe()


# The dataset starts at `2019.11.17 00:03` and the last entry is at `2020.03.29 15:27` with 4374 unique entries. Over the course of the months, the data was gathered roughly each hour, unless the server went down (2 times) or the TSM API was unavailable.

# In[ ]:


# Preview the timeseries of the dataset
wow['created_at'].describe()


# We can preview the items and subclasses next. There is a total of 19 unique items divided into 3 subclasses: Potion, Flask and Food & Drink. During the gathering of data, I focused on getting items useful for raiding at that time. Below you can preview the table describing which item goes into what subclass.

# In[ ]:


# How many different items and subclasses are there?
unique_items = wow['item_name'].unique().tolist()
unique_subclass = wow['item_subclass'].unique().tolist()
f'There are {len(unique_items)} unique items in {len(unique_subclass)} subclasses.'


# In[ ]:


# List the items and their classes
wow.drop_duplicates(subset=['item_name'])[['item_name', 'item_subclass']].set_index('item_subclass')


# As correlations go, we can see a few interesting connections:
# 
# - The `item_min_buyout` isn't really dependant on the quantity of the item or the number of auctions,
# - The `item_min_buyout` has a negative correlation with the number of days since last raid came out, meaning that the longer it is from the release, the prices go down,
# - The `item_quantity` has a quite high positive correlation with `item_num_auction`. That doesn't give us anything useful, as the data duplicates itself (the more auctions you have, the more items will be on the ah generally).
# - Interestingly `days_after_new_raid` has a positive correlation with `item_num_auctions`. This can happen from a couple of reasons: either more people can craft the recepies, the number of items that didn't sell increased over time, or less people buy the items in general... or all at the same time.

# In[ ]:


# Preview correlations
sns.heatmap(wow.corr())


# We will take out the `Famine Evaluator And Snack Table` to analyze it separately from other food items. This is because the prices for one feast are vastly higher than normal food items, so the plots would look horrible. We will also groupby the items by day and subcategory, taking the mean, to see what generally happens with the prices for selected subcategories.
# 
# As we can see in the two plots below, the prices vary over time. Interestingly we can see a big increase, then a huge drop and again a big increase around the release of Ny'alotha Raid. This is due how the playerbase behaves around new raid releases. Players come back to the game before the raid, just to get bored of content (again decreasing), to the point where hardcore players buy out items to progress through the new raid. The prices usually increase over the first couple of weeks, to drop slowly to the next raid release.

# In[ ]:


# Exclude feast from other due to big single item value
wow_feast = wow[wow['item_name'] == 'Famine Evaluator And Snack Table']
wow_other = wow[wow['item_name'] != 'Famine Evaluator And Snack Table']


# In[ ]:


# Check mean prices over the period for subclasses
wow_daily = wow_other.groupby([wow_other['created_at'].dt.date, 'item_subclass']).mean().reset_index()
wow_daily_feast = wow_feast.groupby([wow_feast['created_at'].dt.date]).mean().reset_index()

display('Mean values based on subclass:')
display(wow_daily.head())
display('Mean feast values based on subclass:')
display(wow_daily_feast.head())


# In[ ]:


# Daily mean prices per item subclass
plt.figure(figsize=(10,5))
ax = sns.lineplot(
    x='created_at',
    y='item_min_buyout',
    hue='item_subclass',
    data=wow_daily
)
ax.set_xlim(wow_daily['created_at'].min(), wow_daily['created_at'].max())
ax.set_xlabel('Dates')
ax.set_ylabel('Price in gold per item')
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
)
plt.axvline(NYALOTHA_RELEASE, color='black', linestyle='--', label='Ny\'alotha')
plt.show()


# In[ ]:


# The same but for Feasts
plt.figure(figsize=(10,5))
ax = sns.lineplot(
    x='created_at',
    y='item_min_buyout',
    data=wow_daily_feast
)
ax.set_xlim(wow_daily_feast['created_at'].min(), wow_daily_feast['created_at'].max())
ax.set_xlabel('Dates')
ax.set_ylabel('Price in gold per Feast')
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
),
plt.axvline(NYALOTHA_RELEASE, color='black', linestyle='--', label='Ny\'alotha')
plt.show()


# An interesting thing to check was how the numbers change during the weekday. Usually guilds go to raids on selected days of a week. Popular days are wednesday, thursday and sunday. As we can see, the prices drop closer to the reset (which happens on wednesday), to jump up on wednesday and thursday, to go slowly down again. This was expected behavior. The only interesting thing is that feasts are usually bought on thursday.
# 
# Hourly prices were also checked, but didn't provide any interesting results.

# In[ ]:


# Check mean prices for days of week
wow_weekday = wow_other.groupby([wow_other['created_at'].dt.dayofweek, 'item_subclass']).mean().reset_index()
wow_weekday_feast = wow_feast.groupby(wow_feast['created_at'].dt.dayofweek).mean().reset_index()

display('Mean values based on subclass:')
display(wow_weekday.head())
display('Mean feast values based on subclass:')
display(wow_weekday_feast.head())


# In[ ]:


# Weekday mean prices per item subclass
plt.figure(figsize=(10,5))
days = ['', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
ax = sns.lineplot(
    x='created_at',
    y='item_min_buyout',
    hue='item_subclass',
    data=wow_weekday,
)
ax.set_xlabel('Day of week')
ax.set_ylabel('Price in gold per item')
ax.set(xticklabels=days)
plt.show()


# In[ ]:


# Weekday Feast mean prices per item subclass
plt.figure(figsize=(10,5))
days = ['', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
ax = sns.lineplot(
    x='created_at',
    y='item_min_buyout',
    data=wow_weekday_feast,
)
ax.set_xlabel('Day of week')
ax.set_ylabel('Price in gold per Feast')
ax.set(xticklabels=days)
plt.show()


# # Auction House Prediction
# ### Imports

# In[ ]:


import torch
import torch.nn as nn 
import torch.autograd as autograd 
from torch.autograd import Variable
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# ### Create features and target

# In[ ]:


features = ['item_name', 'item_subclass', 'item_num_auctions', 'days_after_new_raid']
wow_features = wow_other[features]

# One Hot encoding
wow_features = pd.get_dummies(
    wow_features,
    columns=['item_name', 'item_subclass'],
)

# Scaling continuous values
wow_features[['item_num_auctions']] = preprocessing.scale(wow_features[['item_num_auctions']])
wow_features[['days_after_new_raid']] = preprocessing.scale(wow_features[['days_after_new_raid']])

# Display features
display(wow_features.columns)
display(wow_features[['item_num_auctions', 'days_after_new_raid']].head())

# Create and display target
wow_target = wow_other[['item_min_buyout']]
display(wow_target.head())


# ### Create training and test data

# In[ ]:


X_train, x_test, Y_train, y_test = train_test_split(
    wow_features,
    wow_target,
    test_size=0.2,
    random_state=42,
)


# ### Convert to Torch tensors

# In[ ]:


X_train_tr = torch.tensor(X_train.values, dtype=torch.float)
x_test_tr = torch.tensor(x_test.values, dtype=torch.float)
Y_train_tr = torch.tensor(Y_train.values, dtype=torch.float)
y_test_tr = torch.tensor(y_test.values, dtype=torch.float)

# Display sizes
display('X train size:', X_train_tr.shape)
display('Y train size:', Y_train_tr.shape)


# ### Neural network parameters

# In[ ]:


input_size = X_train_tr.shape[1]
output_size = Y_train_tr.shape[1]
hidden_layers = 100
loss_function = torch.nn.MSELoss()
learning_rate = 0.0001


# ### Define model

# In[ ]:


model = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_layers),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_layers, output_size),
)


# ### Train the model

# In[ ]:


for i in range(10000):
    y_pred = model(X_train_tr)
    loss = loss_function(y_pred, Y_train_tr)
    
    if i % 1000 == 0:
        print(i, loss.item())
    
    model.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad


# ### Using the model for predictions

# In[ ]:


sample = x_test.iloc[1410]
display(sample)

# Convert to tensor
sample_tr = torch.tensor(sample.values, dtype=torch.float)
display(sample_tr)


# In[ ]:


# Do predictions
y_pred = model(sample_tr)
print(f'Predicted price of item is: {int(y_pred.item())}')
print(f'Actual price of item is: {int(y_test.iloc[1410])}')


# In[ ]:


# Predict prices for entire dataset and show on graph
y_pred_tr = model(x_test_tr)
y_pred = y_pred_tr.detach().numpy()

plt.scatter(y_pred, y_test.values, s=1)
plt.xlabel("Actual Price")
plt.ylabel("Predicted price")

plt.title("Predicted prices vs Actual prices")
plt.show()

