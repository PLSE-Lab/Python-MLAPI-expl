#!/usr/bin/env python
# coding: utf-8

# I have played a ton of magic in my days, I want to examine some things.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (12, 12)
import seaborn as sns

# Data Frame containing all card sets from the time of creating of MTG
all_sets = pd.read_json('../input/AllSets-x.json')

# Generate a new dataframe containing every card from every card set
df_list = []
for i in all_sets.columns.values:
    deck = all_sets[str(i)]['cards']
    df = pd.DataFrame(deck)
    df_list.append(df)
all_cards = pd.concat(df_list)
# Perform a set of ops on data frame that have been developed through time working with data...
# all_cards.colors = all_cards.colors.astype(str)
all_cards.cmc.fillna(0)
# We need this properly index from all cards after combining sets
all_cards.reset_index(inplace=True)
# all_cards.describe()
all_cards.columns.values


# In[ ]:


# This card has a converted mana cost of 1 million. Obviously can't be played
crazy_card = all_cards[all_cards.cmc >= 1000000]
print(str(crazy_card.flavor))


# In[ ]:





# In[ ]:


# Used this to determine the cut-off of mana costs
# len(all_cards[all_cards.cmc < 1000000])
# len(all_cards[all_cards.cmc >= 1000000])
# len(all_cards[all_cards.cmc > 17])
all_cards = all_cards[all_cards.cmc < 17]

# Set the bin range for all Mana Curves for more standardized visualization
b = range(0,16)

sns.plt.title('Mana Curve - All Cards')
sns.distplot(
    all_cards.cmc[all_cards.cmc < 17], 
    bins = b, kde=False)


# In[ ]:


creature = all_cards[all_cards.type.str.contains('Creature')]
artifact = all_cards[all_cards.type.str.contains('Artifact')]
instant = all_cards[all_cards.type.str.contains('Instant')]
sorcery = all_cards[all_cards.type.str.contains('Sorcery')]
enchantment = all_cards[all_cards.type.str.contains('Enchantment')]
fig = plt.figure()
fig.suptitle("Mana Curves by Card Type", fontsize=16)

sns.plt.title('Mana Curve - All Cards')
plt.subplot(231)
sns.distplot(
    creature.cmc, 
    bins = b, kde=False,color='green')

plt.subplot(232)
sns.distplot(
    artifact.cmc, 
    bins = b, kde=False)
plt.subplot(233)
plt.hist(instant.cmc, facecolor='black', bins=b)
plt.subplot(234)
plt.hist(sorcery.cmc, facecolor='r', bins=b)
plt.subplot(235)
plt.hist(enchantment.cmc, facecolor='g', bins=b)


# In[ ]:


creature.text = creature.text.fillna('')

fig = plt.figure()
fig.suptitle("Occurences of Abilities", fontsize=16)
# anything with double strike also has first strike
searchfor = ['First strike', 'Double strike']
creature["FirstStrike"] = creature.text.str.contains('|'.join(searchfor))
creature["Flying"] = creature.text.str.contains('Flying')
creature["Haste"] = creature.text.str.contains('haste')
creature["Trample"] = creature.text.str.contains('trample')

plt.subplot(231)
sns.countplot(creature.Flying, color='silver')
plt.subplot(232)
sns.countplot(creature.FirstStrike, color = 'red')
plt.subplot(233)
sns.countplot(creature.Haste, color='blue')
plt.subplot(234)
sns.countplot(creature.Trample, color = 'white')


# In[ ]:


creature


# In[ ]:


creature.colors.to_string()
# creature = creature[creature.colors.fillna('None')]
green_creatures = creature[creature.colors.str.contains("Green", na=False)]
white_creatures = creature[creature.colors.str.contains("White", na=False)]
red_creatures = creature[creature.colors.str.contains("Red", na=False)]
blue_creatures = creature[creature.colors.str.contains("Blue", na=False)]
black_creatures = creature[creature.colors.str.contains("Black", na=False)]


# In[ ]:


creature


# In[ ]:


fig = plt.figure()
fig.suptitle("Mana Curves by Creature Color", fontsize=16)
plt.subplot(231)
plt.hist(white_creatures.cmc, facecolor='silver', bins=b)
plt.subplot(232)
plt.hist(blue_creatures.cmc, facecolor='blue', bins=b)
plt.subplot(233)
plt.hist(black_creatures.cmc, facecolor='black', bins=b)
plt.subplot(234)
plt.hist(red_creatures.cmc, facecolor='r', bins=b)
plt.subplot(235)
plt.hist(green_creatures.cmc, facecolor='g', bins=b)


# ### To Do: Create a method to dissect each of the different card types by color so that its not super repetitive.

# In[ ]:


valid_release = all_cards.releaseDate.dropna()
time_ser = pd.to_datetime(valid_release)
year = time_ser.dt.year
plt.hist(year, bins = 50)


# In[ ]:


all_cards.rarity.value_counts().plot(kind='bar')


# In[ ]:


mythics = all_cards[all_cards.rarity == "Mythic Rare"]
plt.hist(mythics.cmc, bins=range(0,16))


# # We see many of the mythic rare cards around the 4, 5, 6 and drop range. 
# 
# This is a point in Magic that can be anywhere from 4th turn or even 3rd turn to late game, depending on the strategy of your deck. I have built decks in which the highest converted mana cost (cmc) is 4. 

# In[ ]:


rares = all_cards[all_cards.rarity == "Rare"]
plt.hist(rares.cmc, bins=range(0,16))


# In[ ]:


uncommons = all_cards[all_cards.rarity == "Uncommon"]
plt.hist(uncommons.cmc, bins=range(0,16))


# From the Rare and Common mana curves, we see lower cost cards. 
# 
# # Creatures
# 
# Now I want to go back to the creatures and take a look at the color by color trends.

# In[ ]:


all_cards_power = all_cards.power.dropna()
all_cards_power = all_cards_power[all_cards_power.str.isnumeric()].astype(int)
all_cards_power = all_cards_power[all_cards_power < 12]
sns.distplot(all_cards_power.astype(int), kde=False)


# In[ ]:


all_cards_tough = all_cards.toughness.dropna()
all_cards_tough = all_cards_tough[all_cards_tough.str.isnumeric()].astype(int)
all_cards_tough = all_cards_tough[all_cards_tough < 12]
sns.distplot(all_cards_tough.astype(int), kde=False)


# In[ ]:


valid_power = creature[creature.power.dropna()]
valid_power = valid_power[valid_power.power.str.isnumeric()]
plt.scatter(valid_power.cmc, valid_power.power)


# In[ ]:


creature['Angel'] = creature.type.str.contains('Angel')


# # Looking at color presence in angels

# In[ ]:


white_angels = creature.Angel.groupby(creature.colors.str.contains("White", na=False)).sum()
axis('equal');
fig = pie(white_angels, labels=white_angels.index);
plt.title('Angels contain White')


# In[ ]:


black_angels = creature.Angel.groupby(creature.colors.str.contains("Black", na=False)).sum()
axis('equal');
fig = pie(black_angels, labels=black_angels.index);
plt.title('Angels contain Black')


# In[ ]:


green_angels = creature.Angel.groupby(creature.colors.str.contains("Green", na=False)).sum()
axis('equal');
fig = pie(green_angels, labels=green_angels.index);
plt.title('Angels contain Green')


# In[ ]:


green_angels = creature.Angel.groupby(creature.colors.str.contains("Red", na=False)).sum()
axis('equal');
fig = pie(green_angels, labels=green_angels.index);
plt.title('Angels contain Red')


# In[ ]:


green_angels = creature.Angel.groupby(creature.colors.str.contains("Blue", na=False)).sum()
axis('equal');
fig = pie(green_angels, labels=green_angels.index);
plt.title('Angels contain Blue')


# In[ ]:




