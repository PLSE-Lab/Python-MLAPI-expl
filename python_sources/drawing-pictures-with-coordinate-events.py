#!/usr/bin/env python
# coding: utf-8

# # Drawing Pictures With Coordinate Events
# We have x, y positions for the coordinates that users are interacting with when they are doing various activities. Let's see if we can plot the interactions to get an idea about how these activities are played and possible generate useful features.

# In[ ]:


import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import NonUniformImage

from math import floor
from random import sample

import json

import missingno as msno

get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(0)


# ## Read Data
# 
# Read in all the data and process to get some convenient lists.
# 
# Some functions taken from this [kernel](https://www.kaggle.com/braquino/890-features).

# In[ ]:


def read_data():
    print('Reading train.csv file....')
    train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
    print('Training.csv file have {} rows and {} columns'.format(
        train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
    print('Test.csv file have {} rows and {} columns'.format(
        test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(
        train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(
        specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(
        sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, train_labels, specs, sample_submission


# read data
train, test, train_labels, specs, sample_submission = read_data()


# We'll also load the order of the games for plotting later on.
# 
# Taken from the [competiton creators](https://www.kaggle.com/c/data-science-bowl-2019/discussion/121215).

# In[ ]:


titles = [["Welcome to Lost Lagoon!"], ["Tree Top City - Level 1"], ["Ordering Spheres"], ["All Star Sorting"], ["Costume Box"], ["Fireworks (Activity)"], ["12 Monkeys"], ["Tree Top City - Level 2"], ["Flower Waterer (Activity)"], ["Pirate's Tale"], ["Mushroom Sorter (Assessment)"], ["Air Show"], ["Treasure Map"], ["Tree Top City - Level 3"], ["Crystals Rule"], ["Rulers"], ["Bug Measurer (Activity)"], ["Bird Measurer (Assessment)"], ["Magma Peak - Level 1"], ["Sandcastle Builder (Activity)"], ["Slop Problem"], ["Scrub-A-Dub"], ["Watering Hole (Activity)"], ["Magma Peak - Level 2"], ["Dino Drink"], ["Bubble Bath"], ["Bottle Filler (Activity)"], ["Dino Dive"], ["Cauldron Filler (Assessment)"], ["Crystal Caves - Level 1"], ["Chow Time"], ["Balancing Act"], ["Chicken Balancer (Activity)"], ["Lifting Heavy Things"], ["Crystal Caves - Level 2"], ["Honey Cake"], ["Happy Camel"], ["Cart Balancer (Assessment)"], ["Leaf Leader"], ["Crystal Caves - Level 3"], ["Heavy, Heavier, Heaviest"], ["Pan Balance"], ["Egg Dropper (Activity)"], ["Chest Sorter (Assessment)"]]
ordered_titles = pd.DataFrame.from_records(titles, columns=['title'])
ordered_titles = ordered_titles.reset_index().rename(columns={'index': 'order'}).set_index('title')
ordered_titles.head()


# ## Data Augmentation
# Convert the events data to a dataframe. We should also get various game information to help us later on. We won't be looking at Clip events since they aren't interesting in terms of interaction and we'll filter only coordinate events first.

# In[ ]:


train = train[(train["event_data"].notnull())]
train = train[(train["type"] != "Clip")]
train = train[train["event_data"].apply(lambda x: 'coordinates' in x)]

train = train.sample(1000000)

event_data = pd.DataFrame.from_records(
    train.event_data.apply(json.loads).tolist(),
    index=train.index
)
# Sort the most non-null columns at the start
event_data = pd.merge(
    event_data[event_data.isnull().sum().sort_values().index],
    train[['title', 'type', 'world']],
    left_index=True,
    right_index=True)

del train


# In[ ]:


print("Total of {} rows and {} features.".format(*event_data.shape))


# In[ ]:


event_data.head()


# We see that there are many features related to the coordinate events.
# 
# Let's check our missing values.

# In[ ]:


msno.matrix(event_data.iloc[:, :50].sample(250))
fig = plt.gcf()


# But most of these features are still null and not necessarily useful for this analysis, so let's drop everythign except the coordinates and some key descriptive info.

# In[ ]:


event_data = event_data[['title', 'world', 'type', 'coordinates']]
event_data.head()


# Finally, we should convert the coordinates into useful columns.

# In[ ]:


event_data = pd.merge(pd.DataFrame.from_records(
    event_data.coordinates.values.tolist(), index=event_data.index),
                      event_data.drop('coordinates', axis=1),
                      left_index=True,
                      right_index=True)
event_data.head()


# We now have coordinates and the corresponding stage information, but we should normalise so that we can more easily plot. Let's scale everything to max width 100, and keep the aspect ratio for height.

# In[ ]:


event_data['scale'] = 100 / event_data.stage_width

event_data[['x', 'y', 'stage_width', 'stage_height']] =     event_data[['x', 'y', 'stage_width', 'stage_height']]     .multiply(event_data['scale'], axis=0)


# In[ ]:


event_data.head()


# How many different maps do we have?

# In[ ]:


print(f'{event_data.title.nunique()} unique maps to plot')


# In[ ]:


event_data = pd.merge(
    event_data,
    ordered_titles,
    left_on='title',
    right_index=True)


# ## Plotting
# Let's plot these and see how they look! Do we see certain areas where most click? If children are more commonly clicking these, they may be more likely to succeed.
# 
# We can plot them in various orders for our convenience.

# In[ ]:


event_data.query('title == "Welcome to Lost Lagoon!"')


# In[ ]:


def plot_heatmaps(event_data):
    xedges = np.linspace(0, 100, 51)
    yedges = np.linspace(0, 100, 51)
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    cols = 4
    rows = int(24/cols)

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 22))

    color_maps = {'MAGMAPEAK': 'Reds',
                  'CRYSTALCAVES': 'Blues',
                  'TREETOPCITY': 'Greens'}

    for i, _ in enumerate(event_data[['title', 'world', 'type']].drop_duplicates().iterrows()):
        game, world, game_type = _[1].values
        game_data = event_data.query('title == @game')
        # Linear x array for cell centers:
        H, xedges, yedges = np.histogram2d(game_data.x,
                                           game_data.y,
                                           bins=(xedges, yedges))
        H = H.T

        ax = axs[floor(i/cols), i%cols]
        # ax = fig.add_subplot(int('44'+str(i+1)))

        interp = 'nearest' # bilinear, nearest

        im = NonUniformImage(ax, interpolation=interp, extent=(xedges.min(), xedges.max(), yedges.min(), yedges.max()),
                            cmap=plt.get_cmap(color_maps[world]))
        im.set_data(xcenters, ycenters, H)
        ax.images.append(im)
        ax.set_xlim(xedges.min(), xedges.max())
        ax.set_ylim(yedges.min(), yedges.max())
        ax.set_title('{} ({})'.format(game, game_type) if game_type=='Game' else game)
        ax.set_aspect(aspect='equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.show()


# First, let's have a look at them in the intended order.
# 
# The dark regions will show where users are more likely to be clicking, light regions show areas of relative inactivity.

# In[ ]:


event_data = event_data.sort_values(by=['order'])

plot_heatmaps(event_data)


# Many interesting patterns! 
# 
# We can certainly see concentrated areas for interacting vs. the lighter areas with few interactions. This shows that there are places where users should be interacting vs. those where it's wasted.
# 
# We also see that worlds have similar patterns. For example, in the Crystal Caves shown an the bottom there are many cases of even interacting on left and right compared to the asymmetric behaviour of the other worlds. It also seems that Tree Top City has many more vertical lines - totally in line with tree tops!
# 
# To create a new feature, we could define 'hit boxes' based on these diagrams and then judge any coordinate event as good or bad. If it's inside the hit box it is good and if not it is bad. This could be an interesting feature to experiment with! (To some degree it should be correlated with number of events since the more bad events, the more total events as you will likely need to do the same amount of good events to complete the game).
# 
# Next, how about ordering by the game type? We should be able to see the similar ones more readily.

# In[ ]:


event_data = event_data.sort_values(by=['type', 'order'])

plot_heatmaps(event_data)


# Comparing activities and games to assessments, the latter seems to have many more different spots to interact with - difficult! Though, there are some activities which are more similar. For example, Flower Waterer or Chow Time - these might be better indicators for performance on assessments than more simple games like "Crystals Rules".

# ## Conclusion
# By looking in more detail at the coordinate events, we're able to see some interesting patterns in the data. These patterns make sense based on our outside knowledge of the games.
# 
# We could use these plots to make some interesting features to capture the quality of the coordinate events.
