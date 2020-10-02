#!/usr/bin/env python
# coding: utf-8

# ### Hey there!
# 
# This will be a quick, for now, notebook about the costs of cards to play on Magic, we will:
# 
# - Read the data file
# - Order releases by time
# - Remove the outliers that was causing problems
# - Plot some stats about the mana costs across sets

# In[ ]:


# load libraries
import re
import pandas as pd
import numpy as np
import seaborn as sns
import warnings  # ignore warnings, or you crash or leave me alone
from termcolor import colored
from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.simplefilter("ignore", category=RuntimeWarning)
sns.set_palette('Set2')

def log(s):
    print(colored('[OK]', 'green'), s)


# In[ ]:


# ingest the data!
data = pd.read_json('../input/AllSets-x.json').to_dict()
data = OrderedDict(data)
log('Data Ingested')


# After some time on this analysis I found that all the times outliers come out, so I took a time to investigate, since it was find coming from the mean values, and from my prior experience with Magic I've never seen a card with cost above 15 so I used this code to find the corrupt set, so I can remove it.
# 
# ```python
# mana = Mana()
# for release in by_release:
#     if np.nanmean(np.array([mana.paying_cost(card.get('manaCost')) for card in data[release]['cards']], np.float)) > 15:
#         print(release)
# ```
# 
# Output:  `UNH`
# 
# With a bit of [Wikipedia](https://en.wikipedia.org/wiki/Unhinged_%28Magic:_The_Gathering%29), I found that `UNH` is a "humor" expansion, because of that it was showing as an outlier from a
# the tradicional data of Magic.
# 
# Also we found that there is another one of that type of expansion called `Unglued` with code `UGL` that we will remove as well
# 
# > Unhinged is a humor and parody themed expansion set to the collectible card game Magic: The Gathering. Unhinged was released on November 19, 2004. Its tone is less serious than traditional Magic expansions. It is a follow-on to Unglued, an earlier humor themed expansion set.

# In[ ]:


# remove UNH from the data
_ = data.pop('UNH')
_ = data.pop('UGL')


# In[ ]:


# order the sets by release
def by_date(tpl):
    return tpl[1]['releaseDate']

by_release = OrderedDict(sorted(data.items(), key=by_date))
log('Ordered')


# # Mana cost per set
# 
# Mana is the resource in the Magic: The Gathering's game, used in order to cast spells and activate abilities. Mana is most usually acquired by tapping lands. 
# 
# We will analyse how the sets evolved their mana cost, so we can have a clue on how it develops, maintaing the same amount or variantes.

# In[ ]:


# -*- coding: utf-8 -*-
import re

class Mana:
    """
    Mana class wrapper.
    It provides methods to clean and compute mana values into readable,
    numerical, and categorical values to use on explorations.
    """

    MANA_PALETTE = {
        'B': '#cdc3c1',
        'R': '#faac90',
        'G': '#9cd4af',
        'W': '#fffcd7',
        'U': '#abe1fa',
        'C': '#cccccc'
    }

    COST_IDENTITIES = {
        'B': 'Black',
        'R': 'Red',
        'G': 'Green',
        'W': 'White',
        'U': 'Blue',
        'C': 'Colorless',
        'N': 'Generic'
    }

    def __init__(self):
        self.any_color_regex = r'\d+'
        self.mana_types = ['B', 'R', 'G', 'W', 'U', 'C']

    def paying_cost(self, mana_cost):
        """
        Computes the paying cost, parsing the input while ignoring the colors
        of mana required, returning the raw minimal number of manas needs.
        The minimal number of manas needed means that a card of:
        :code:`{1}{U}` have a value of :code:`2` since it needs 1 of any color
        and 1 of blue while a card with cost :code:`{X}{U}` will have the value
        of 1 because of :code:`X` is arbitrary.

        :param mana_cost: The cost of a card to be computed
        :type mana_cost: str
        :return: The computed paying cost
        :rtype: int
        """

        if mana_cost is None:
            return 0

        # search for generic manas
        generic = sum([int(num) for num in
                       re.findall(self.any_color_regex, mana_cost)])
        # count the numbers of colored manas
        colored = sum([mana_cost.count(mana_type) for mana_type in
                       self.mana_types])

        return sum([generic, colored])

    def mana_type_cost(self, mana_cost):
        """
        Computes the mana cost for each of the present type of mana
        present on the cost, inputing :code:`{1}{U}` will result on a
        mapping to the type of costs and their count: :code:`{'U': 1, 'N': 1}`
        :param mana_cost: The cost to be computed
        :type mana_cost: str
        :return: a dict contaning the counts for each mana type
        :rtype: dict
        """

        accumulator = {
            'B': 0,
            'R': 0,
            'G': 0,
            'W': 0,
            'U': 0,
            'C': 0,
            'N': 0,
        }

        if mana_cost is None:
            return accumulator

        generic = sum([int(num)
                       for num in re.findall(self.any_color_regex, mana_cost)])
        accumulator['N'] = generic

        for mana_type in self.mana_types:
            accumulator[mana_type] = mana_cost.count(mana_type)

        return accumulator


# In[ ]:


def costs_by_set():
    mana = Mana()
    acc = dict()

    for release in by_release:
        costs = [mana.paying_cost(card.get('manaCost'))
                 for card in data[release]['cards']]
        acc[release] = costs
        log('Costs of {}'.format(release))
    return acc


# In[ ]:


costs = costs_by_set()


# In[ ]:


def apply_on_cost(costs, func):
    acc = dict()
    for items in costs.items():
        name = items[0]
        value = func(items[1])
        acc[name] = value
    return acc


# In[ ]:


mean_costs_per_set = apply_on_cost(costs, np.nanmean)
var_costs_per_set = apply_on_cost(costs, np.nanvar)
min_costs_per_set = apply_on_cost(costs, np.nanmin)
max_costs_per_set = apply_on_cost(costs, np.nanmax)


# In[ ]:


def within_acceptable(cost):
    """
    Since we got alot of noise on the mana cost for cards, we often got alot
    of mean and min of 0 because of the no existence of data, so to 
    our visualizations lead us to belive the majority of cards as a mean of
    0 which would be unprecise, since all cards needs a cost, we chop only the ones
    that have a cost between 0 and 10
    """
    return 0 < cost <= 10


# In[ ]:


gs = GridSpec(3, 2)
fig = plt.figure(figsize=(10, 12))
fig.suptitle('Distribution of Mana Cost on Sets')

# fist axes the min of mana
plot_data = [cost for cost in list(min_costs_per_set.values()) if within_acceptable(cost)]
ax = plt.subplot(gs[0, 0])
ax.set_title('Minimun value of Mana on Sets')
_ = sns.boxplot(y=plot_data, ax=ax, color=Mana.MANA_PALETTE['G'])
# second plot the max of mana
plot_data = [cost for cost in list(max_costs_per_set.values()) if within_acceptable(cost)]
ax = plt.subplot(gs[0, 1])
ax.set_title('Maximun value of Mana on Sets')
_ = sns.boxplot(y=plot_data, ax=ax, color=Mana.MANA_PALETTE['R'])
# thir plot the mean of mana
plot_data = [cost for cost in list(mean_costs_per_set.values()) if within_acceptable(cost)]
ax = plt.subplot(gs[1, :])
ax.set_title('Mean value of Mana on Sets')
_ = sns.violinplot(x=plot_data, color=Mana.MANA_PALETTE['U'], ax=ax)
# variance of mana cost per set
ax = plt.subplot(gs[2, :])
ax.set_title('Evolution of Variance of Mana on Sets')
ax.plot(list(var_costs_per_set.values()), c=Mana.MANA_PALETTE['R'])
ax.set_xlabel('Set releases by time')
_ = ax.set_ylabel('Variance')


# We can conclude from the plot above that the "average" minimal amout of mana on sets floats around **2.5**, which makes cards with cost of 1 pretty rare inside a set compared to others but they are present. On the maximum amount of cost per set we can see that we have outliers floating from **2** to **4**, those could be from sets that have a reduced amount of cards, due to comemorations, promotions and *not so often stuff*, the same we can say of the top costs of 10 and maybe even higher, since we chopped the costs above 10 to prevent noise data.
# 
# The mean value reveal to us that usually cards have a cost of **3**, not strapolating from **4**, we still see sets with mean of 5, 6 even 7, but due to the irregularity of the set size this can occur.
# 
# -----
