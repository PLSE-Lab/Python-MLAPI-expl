#!/usr/bin/env python
# coding: utf-8

# Let's try plotting some of the data we recorded from our Pokemon snake game last week!

# Import libraries and load data

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

plt.style.use('seaborn-ticks')

# This is some sample data, but you can upload your own instead if you like!
game_data = pd.read_csv("../input/1552513645.59.csv")


# Let's see what our data looks like!

# In[18]:


game_data.head()


# One thing that's really easy to do with pandas is to pull out all the rows where a certain condition is true.
# 
# First we check whether the column "Event" matches a certain Pokemen. Let's use Jigglypuff as an example:

# In[17]:


pokemon = 'Jigglypuff'
game_data.Event == pokemon


# That gives us "True" for all the Jigglypuff entries, and "False" for everything else. Now let's use that list of Trues and Falses to select the data we want:

# In[19]:


game_data[game_data.Event == pokemon]


# # Pokemon counts over time

# We recorded our game data in terms of "events," but we want a count of how many of each Pokemon type we have over time. For each Pokemon type, we need to count up the number of times we caught that Pokemon (also called the <b>cumulative sum</b>). That way, we can tell you for any given moment in time how many Jigglypuffs we have, for example.

# In[5]:


for pokemon in set(game_data.Event):
    game_data[pokemon] = (game_data.Event == pokemon).cumsum()


# Now we can see at each time point how many of each Pokemon we have:

# In[20]:


game_data.head()


# And we can make a plot!

# In[21]:


for pokemon in set(game_data.Event):
    plt.plot(game_data.Time, game_data[pokemon], label = pokemon)
plt.xlabel('Time')
plt.ylabel('Number caught')
plt.legend()
plt.show()


# # Plotting Pokemon positions

# We also recorded the X and Y positions where we caught each Pokemon. Let's plot them to see if there are any patterns to where the different types show up.
# 
# For this, maybe we want to merge together a few dataframes from different games we've played. To do that, we'll first load all our files into a list, and then use the function "concat" to mush them together.

# In[8]:


file_path = "../input/"
files = []
for f in os.listdir(file_path):
    df = pd.read_csv(file_path + f, index_col = 0)
    if df.shape[1] == 5:
        df.drop('Unnamed: 0.1', axis = 1, inplace = True)
        df.to_csv(file_path + f)
    files.append(df)

game_data = pd.concat(files, sort = False)


# First of all, we want to pull out just the events where we caught a particular type of Pokemon. Let's use Jigglypuff again.

# In[10]:


pokemon = 'Jigglypuff'
subset = game_data[game_data.Event == pokemon].copy()


# In[22]:


subset.head()


# Now, we can make an array the same size as the screen we were playing on earlier, and fill in each position according to how many times its position shows up in our dataframe.

# In[12]:


screen = np.zeros((19, 19))

for x in range(19):
    for y in range(19):
        screen[y, x] = subset[(subset.X == x + 1) & (subset.Y == y + 1)].shape[0]


# We can easily plot the array we just made as a heatmap.

# In[23]:


plt.imshow(screen, cmap = 'viridis', origin = 'lower')
c = plt.colorbar()
c.set_label('Number of ' + pokemon + 's seen')
plt.axis('off')
plt.show()


# Do any of the Pokemon tend to show up in only some places? (Hint: why would we ask if they didn't?)

# Let's see a couple more types of Seaborn plot!
# 
# One type of plot is the violin plot, which shows you the distribution of some data points, split by category. In this example, the data points are the X coordinates of the Pokemon, and the categories are the type of Pokemon. You can see from this that a couple of them have different distributions than the others.

# In[24]:


sns.violinplot(x = 'Event', y = 'X', data = game_data)
plt.gca().set_xlabel('')
plt.show()


# The KDE plot is similar to the violin plot in that it tries to show the distribution of your data, but you can plot in 2 dimensions instead of 1. This gives us something that looks a bit like the heatmap from before, but we can plot two on top of each other to compare the "habitats" of two Pokemon.

# In[26]:


subset = game_data[game_data.Event == 'Spearow'].copy()
subset2 = game_data[game_data.Event == 'Nidorina'].copy()

sns.kdeplot(subset2.X, subset2.Y, cmap = 'Reds_d')
sns.kdeplot(subset.X, subset.Y, cmap = 'Blues_d')
plt.show()


# # What other data could you plot?
# 
# - Instead of plotting the positions of the Pokemon, you could ask how much time the player spends in different parts of the screen (you would have to change your Trinket game to record this extra type of data as well).
# - Plot your score over time versus a friend's score over time.
# 
# What other ideas can you think of?
# 
# You can find more examples using Seaborn [here](https://seaborn.pydata.org/examples/index.html).

# In[ ]:




