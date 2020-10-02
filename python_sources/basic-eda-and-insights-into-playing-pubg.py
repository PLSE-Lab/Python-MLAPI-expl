#!/usr/bin/env python
# coding: utf-8

# # PUBG Finish Placement Prediction Challenge

# *The customary description and image.*

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


# ## Loading the Data

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# ## Getting a little insight into the Data

# In[ ]:


train_data.shape, test_data.shape


# Okay, there are 4.36 million training examples, 1.87 million test examples. 25 features + one to serve as the answer *(winPlacePerc)*. Let's take a look at them.

# In[ ]:


train_data.info()


# In[ ]:


train_data.describe()


# In[ ]:


test_data.describe()


# In[ ]:


train_data.head()


# In[ ]:


test_data.tail()


# ## Basic EDA Workflow

# Now that the data's loaded, let's take a moment to explore it. Here are a few thoughts before exploring the data.
# 
# * The final place prediction is pretty obviously related to the **no of people killed**. 
# * Next, we can also take a look at **no of assists**, and various **killing techniques** (DBNOs, damages, headshots, etc). Although, they ought to be less important than the plain no of kills
# * The **distance** a player has covered. Well, if they are able to travel here and there without getting themselves killed, then that's some skill! Walking + Driving + *Swimmimg*
# *  **Boosts**, **heals** and **revivals**. Only on Duo and Squad modes.
# * The game is played in three modes. Solo, Duo, Squad (upto 4). And what matches are played in which mode.
# * Weapons maybe.

# In[ ]:


# Plot countplots and distplots depicting how various features are related to the final place, 
# and some features to each other. Ditch some features and combine some others to get more meaningful
# features. Then, train using some basic scikit learn model.


# ##  1. Kills : Perhaps the most important part of the game

# In[ ]:


data = train_data.copy()


# In[ ]:


avg = data['kills'].mean()
ninety = data['kills'].quantile(0.95)
highest = data['kills'].max()


# In[ ]:


print('Average no. of kills: {0} \nKills of 95% of people: {1} \nMax Kills: {2}'.format(avg, ninety, highest))


# Wow, 95% of people only manage to get 4 kills, while the overall record is 60!

# In[ ]:


# Plotting no. of Kills

plt.figure(figsize = (15, 10))
sns.countplot(data['kills'].sort_values())
plt.title('Kill Count')
plt.show()


# It's not even visible after 4. So, let's club all the kills greater than 4 into one group

# In[ ]:


# Still plotting no. of Kills

data1 = data.copy()

data1['kills'].astype('str')
data1.loc[data['kills'] > data['kills'].quantile(0.95)] = '4+'

plt.figure(figsize = (15, 10))
sns.countplot(data1['kills'].astype('str').sort_values())
plt.title('Modified Kill Count')
plt.show()


# In[ ]:


# Kills vs Damage Dealt

plt.figure(figsize = (15, 10))
sns.scatterplot(x = data['kills'], y = data['damageDealt'])
plt.title('Kills vs Damage Dealt')
plt.show()


# Okay, the more the killings, the more damage is caused by the player. That seems right!

# In[ ]:


# Kills vs Winning Percentage

plt.figure(figsize = (15, 10))
sns.jointplot(y = data['kills'], x = data['winPlacePerc'], height=10, ratio=3, color="b")
plt.title('Kills vs Winning Precentage')
plt.show()


# So, we get all range of results with kills between 0 and 10 kills (some people win even with such low kills), but as kills increase the result only improves (almost no one loses after around 20+ kills). Thus, kills are strongly related to winning.

# ## 2. Headshots, Assists and DBNOs
# 
# ### 2.1 Headshots

# In[ ]:


data = train_data.copy()


# In[ ]:


avg = data['headshotKills'].mean()
ninety = data['headshotKills'].quantile(0.95)
highest = data['headshotKills'].max()


# In[ ]:


print('Average no. of headshots: {0} \nNo. of headshots of 95% of people: {1} \nMax Headshots: {2}'.format(avg, ninety, highest))


# We can say that headshots are very difficult to achieve. 95% people can manage only one per game. But just look at the maximum!

# In[ ]:


# Plotting no. of Headshots

plt.figure(figsize = (15, 10))
sns.countplot(data['headshotKills'].sort_values())
plt.title('Headshot Count')
plt.show()


# The values 4-26 are so low, they aren't even visible. Not much use of this plot.

# In[ ]:


# Headshots vs Winning Prediction

plt.figure(figsize = (15, 10))
sns.jointplot(y = data['headshotKills'], x = data['winPlacePerc'], height=10, ratio=3, color="c")
plt.title('Headshots vs Winning Precentage')
plt.show()


# More headshots definitely indicate better chances of winning, but people also win the game with little or no headshots.
# 
# *Note that 'Headshots vs Kills' is not plotted because Headshots themselves are Kills. Thus, it will only be redundant.*
# 
# ### 2.2 Assists

# In[ ]:


avg = data['assists'].mean()
ninety = data['assists'].quantile(0.95)
highest = data['assists'].max()


# In[ ]:


print('Average no. of assists: {0} \nNo. of assists of 95% of people: {1} \nMax assists: {2}'.format(avg, ninety, highest))


# As far as I know, assists only happen in Duo or Squad games, but they're worth checking out.

# In[ ]:


# Plotting no. of Assists

plt.figure(figsize = (15, 10))
sns.countplot(data['assists'].sort_values())
plt.title('Assist Count')
plt.show()


# Almost non-existant after 4.

# In[ ]:


# Assists vs Winning Prediction

plt.figure(figsize = (15, 10))
sns.jointplot(y = data['assists'], x = data['winPlacePerc'], height=10, ratio=3, color="y")
plt.title('Assists vs Winning Precentage')
plt.show()


# Thus, more assists tend to favour the chances of a better rank, but this relation is not quite as prominent.
# 
# ### 2.3 DBNOs (Down But Not Outs)
# DBNOs are different from Assists. A player is knocked out, but is still alive and can be revived by his/her team mates.

# In[ ]:


avg = data['DBNOs'].mean()
ninety = data['DBNOs'].quantile(0.95)
highest = data['DBNOs'].max()


# In[ ]:


print('Average no. of DBNOs: {0} \nNo. of DBNOs of 95% of people: {1} \nMax DBNOs: {2}'.format(avg, ninety, highest))


# In[ ]:


# Plotting no. of DBNOs

plt.figure(figsize = (15, 10))
sns.countplot(data['DBNOs'].sort_values())
plt.title('DBNOs Count')
plt.show()


# Better distributed than Assists or Headshots. 

# In[ ]:


# DBNOs vs Kills

plt.figure(figsize = (10, 5))
sns.scatterplot(x = data['DBNOs'], y = data['kills'])
plt.title('DBNOs vs Kills')
plt.show()


# We see DBNOs are strongly correlated to Kills. The players who kill more often have higher chance of knocking players out. 

# In[ ]:


# DBNOs vs Winning Prediction

plt.figure(figsize = (15, 10))
sns.jointplot(y = data['DBNOs'], x = data['winPlacePerc'], height=10, ratio=3, color="g")
plt.title('DBNOs vs Winning Precentage')
plt.show()


# If we were to rank these three features in order of level of correspondence to winning percentage, then I would say:
# 
# Assists < DBNOs < Headshots

# In[ ]:


# heals, boosts, distances, game modes (solo, duo, squad)


# ## *More to follow soon.*
# 
# ### Thank you for making this far. Hope you enjoyed this kernel. Comments and suggestions are welcome :) 
