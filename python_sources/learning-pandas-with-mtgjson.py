#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# In order to start teaching myself data science and Python, I decided to work with a data set near and dear to my heart. Magic cards are an ideal data set to study for several reasons - there are a large number of observations (over 10,000 cards and nearly 200 sets released) while still being fairly tractable. There are a variety of interesting characteristics, both numerical and qualitative, to analyze. Most importantly, a number of relevant questions about the design and development of Magic cards can be explored with methods in statistical learning, such as:
# 
# * Have creatures gotten more powerful over the years?
# * Has removal gotten worse?
# * Given the text of a card, is it possible to predict its color and mana cost?
# 
# This notebook will document my process of working with this data set to produce answers to questions like these. I'm still a beginner, so this will be as much about documenting mistakes I make and issues I encounter as it will be about any insights I may discover. Any feedback is appreciated.
# 
# ## Data Extraction
# 
# Before doing any analysis, I need to process the data the make it easy to work with in Python. My first task is to convert `all_sets.json` into a pandas DataFrame using the `read_json` method:

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import itertools as it
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# read sets into a DataFrame; index is chosen as the orient so the set names are the indices
all_sets = pd.read_json("../input/AllSets-x.json", orient = "index")
all_sets.head()


# The `all_sets` DataFrame is a list of all Magic sets (through September 2016) indexed by the set codename. I used the `index` orientation since the structure of the `all_sets.json`, as described by the documentation on the [MTGJSON website]](https://mtgjson.com/documentation.html), is of the form
# 
# `{
#     ...
#     set_code1 : { /* set data /* },
#     set_code2 : { /* set data /* },
#     set_code3 : { /* set data /* },
#     ...
# }`
# 
# which is of the form `{index -> {column -> value}}` described in the [pandas documentation for the `read_json` method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_json.html#pandas.read_json). We can now examine some basic characteristics of `all_sets`:

# In[ ]:


# the set names are the indices, and the columns are the attributes of each set (border, cards, etc)
all_sets.shape


# In[ ]:


all_sets.columns


# In[ ]:


all_sets.describe()


# The `describe` method doesn't provide much useful information here, since so far, the only numeric columns of `all_sets` are the set's id number on the Magic vendor [MagicCardMarket](https://www.cardmarket.com/en/Magic), and a column indicating whether or not the set is only available online. The bulk of the useful data in this DataFrame is contained in the `cards` column, which holds a JSON object containing all the cards in each set. We can convert each element of the `cards` column into a pandas DataFrame using the `json` library in Python and `read_json`:

# In[ ]:


# the cards column contains the cards of each set in json format, so each set of cards can be
# converted from a json object into a DataFrame
all_sets.cards = all_sets.cards.apply(lambda x: pd.read_json(json.dumps(x), orient = "records"))


# In[ ]:


all_sets.cards["RAV"].head()


# In[ ]:


# the shape of this DataFrame gives the number of cards in the set
all_sets.cards["RAV"].shape


# Before cleaning the data, we'll add a relevant numeric column to `all_sets` - the number of cards in each set, which we'll call `setSize`. Each row of each DataFrame in `cards` is one card, so the number of rows gives the number of cards in the set - in the example above, we can see that the set Ravnica: City of Guilds contains 306 cards.

# In[ ]:


setSizeCol = all_sets.apply(lambda x: x.cards.shape[0], axis = 1)
all_sets = all_sets.assign(setSize = setSizeCol)
all_sets.sample(10)


# Now we can begin to clean the data. There are a number of cards in this dataset that aren't intended for tournament play, or intentionally shirk design and development principles. For the sake of simplicity, it makes sense to remove these cards from the dataset. First, there are some sets that we can directly remove from `all_sets`. These include:
# 
# * The Un-sets **Unglued** and **Unhinged** are joke sets that, outside of basic lands, aren't meant for tournament play and include many cards that intentionally violate design principles and break the game.
# * Certain sets of **Promotional cards** that were printed for holidays or other special events.
# * The set of **Vanguard avatars** that are meant to be used in the online-only Vanguard format.

# In[ ]:


# before analyzing the dataset we remove some pathological cards and sets.
# first we remove sets not intended for tournament play.
# these include Un-sets, certain promotional cards, online-only Vanguard avatars, etc.
invalid_sets = ["UGL", "UNH", "pCEL", "pHHO", "VAN"]

def test_invalid_setcode(s, invalid_sets):
    
    for setname in invalid_sets:
        if s == setname:
            return True
        
    return False

all_sets = all_sets.loc[~all_sets.code.map(lambda x: test_invalid_setcode(x, invalid_sets))]


# Next we remove some pathological card types and layouts, which can be done by applying a function to the elements of the `cards` column that removes the following cards:
# 
# * Cards with **plane** or **phenomenon** layouts, which are exclusive to the Planechase format
# * Cards with the **scheme** layout, which are exclusive to the Archenemy format
# * Cards with the **token** layout, which are themselves not Magic cards but are representations of permenents created by other cards
# * Cards with the **conspiracy** card type, which are exclusive to the Conspiracy draft format

# In[ ]:


# we also remove cards that don't have the "typical" format of a Magic card
# these include cards specific to the Planechase and Archenemy formats (planes, schemes, etc),
# cards with the Conspiracy card type, and token cards
card_layouts = ["double-faced", "flip", "leveler", "meld", "normal", "split"]

all_sets.cards = all_sets.cards.apply(lambda x: x.loc[x.layout.map(lambda y: y in card_layouts)])
all_sets.cards = all_sets.cards.apply(lambda x: x.loc[x.types.map(lambda y: y != ["Conspiracy"])])


# Next, we deal with a significant corner case - creatures with variable power and toughness. Normally power and toughness are both fixed integers, but there are many creatures whose power and/or toughness depends on a variable characteristic, such as the number of creatures on the battlefield, the number of cards in the graveyard, etc. Again, for the sake of simplicity, we set these values to `NaN` so that the `power` and `toughness` columns can be treated as numeric columns. Because there are many creatures that actually have 0 power or toughness, we use `NaN` instead of 0 here to prevent skewing the data.
# 
# This is the first example of a step in the data cleaning process where I made significant changes in how I went about in doing this step. Checking the power and toughness and removing variable power/toughness values was initially done much later in the process, after I had joined the cards from each set into one large DataFrame, `all_cards`, containing a copy of every Magic card. I also initally wrote this step (and some other steps later on) as a `for` loop iterating through a list of card names. That proved to be very slow, so I spent some time modifying the process to apply a function to each set of cards modifying power and toughness when necessary.

# In[ ]:


# next we modify creature cards with variable power/toughness - for the sake of numerical analysis, it
# is simpler to remove these values so the power and toughness columns can be cast as numeric columns.
def testfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def new_pt(s):
    if testfloat(s):
        return float(s)
    else:
        return np.nan
    
def fix_pts(c):
    col_list = list(c.columns)
    
    if "power" in col_list and "toughness" in col_list:
        c.loc[:, "power"] = pd.to_numeric(c.loc[:, "power"], errors = "coerce")
        c.loc[:, "toughness"] = pd.to_numeric(c.loc[:, "toughness"], errors = "coerce")
    
    return c
    
all_sets.cards = all_sets.cards.apply(lambda x: fix_pts(x))


# Now we start preparing the DataFrames in `cards` to be combined to form one DataFrame, `all_cards`, containing all cards. It will be useful to have access to both the `cards` DataFrames in `all_sets` and `all_cards` because the former preserves information about set releases, which may be useful when analyzing design trends over time, while the latter will make it easier to directly access information about the cards themselves. The first step is to remove some extraneous columns: 

# In[ ]:


# we remove columns that won't be useful in our analysis.
cols_to_remove = ["multiverseid", "imageName", "border", "mciNumber", "foreignNames",
                  "originalText", "originalType", "source"]

all_sets.cards = all_sets.cards.apply(lambda x: x.loc[:, list(set(x.columns) - set(cols_to_remove))])


# The second is to standardize the columns across sets by taking the union of the remaining labels and appending the appropriate columns to each DataFrame. The second step ensures that when these DataFrames are all combined, the columns will line up correctly.
# 
# This particular method is another example of a step that I had to rework so that I could avoid using iterating through the rows of `all_sets`. Initially, I used a loop to iteratively take the left join of `all_cards` with each set, one by one, using the `align` method to ensure that the columns matched up. I avoided having to deal with the columns directly, but like before, this was prohibitively slow, so I worked out this approach instead.

# In[ ]:


# we standardize the columns of each cards DataFrame by taking the set-theoretic union of the columns
# and appending the remaining columns to each DataFrame.
union_set = set()
set_cols = all_sets.cards.map(lambda x: set(x.columns))

for setname in set_cols.index:
    union_set = union_set | set_cols[setname]
    
union_set


# In[ ]:


def addcols(cards, union_set):
    unused_cols = union_set - set(cards.columns)
    new_cols = pd.DataFrame(data = None, index = cards.index, columns = list(unused_cols))
    return cards.join(new_cols)
    
# after appending the columns we sort them in alphabetical order    
all_sets.cards = all_sets.cards.apply(lambda x: addcols(x, union_set))
all_sets.cards = all_sets.cards.apply(lambda x: x.reindex_axis(sorted(list(x.columns)), axis = 1))


# In[ ]:


# now we can start preparing the all_cards DataFrame, which will be a list of every tournament-legal
# Magic card
# first we select the columns from the cards DataFrames that will be useful
all_cards_columns = ['names', 'layout', 'manaCost', 'cmc', 'colors', 'colorIdentity',
                    'supertypes', 'types', 'subtypes', 'text', 'power', 'toughness',
                    'loyalty', 'rulings', 'foreignNames', 'printings', 'legalities']


# In[ ]:


# set the index of all_cards to be the name column, so we can search cards by name
all_cards = pd.DataFrame(data = None, columns = all_cards_columns)
all_cards.rename_axis("name", inplace = True)
all_cards.head()


# Finally, we need to modify the `rarity` and `printings` columns in each set. When looking at each set individually, it makes sense for `rarity` to be a separate column since each card in a set has a single rarity. But a card can be reprinted at different rarities in different sets, so in `all_cards`, without the context of what set each card is in, it makes sense to store the rarities associated to each printing of the card. A reasonable way to store this information is in a dictionary where the key/value pairs are printings and rarities.
# 
# This process isn't actually completed here; what happens below is that for each card in a given set, the `rarity` column is converted from a string to a dictionary where the keys are taken from the `printings` of the card, but the corresponding values are all empty except for the value of the set, which will be the card's rarity. So for instance, a common in `TSP`, that has been printed in `TSP` and `RAV`, will be given the dictionary
# 
# `{ 'RAV' : None, 'TSP' : 'Common' }`
# 
# The reason for this is because the methods below are applied to each set, so they don't have access to the rarities of cards printed in other sets. It's more efficient to fill out the dictionaries when removing duplicates from `all_cards` later on.

# In[ ]:


# we want to preserve the printing/rarity information in all_cards; we represent this information
# as a dictionary where the key/value pairs are printings and rarities
def convert_printings(x, set_name):
    x["printings"] = dict.fromkeys(x["printings"])
    x["printings"].update({set_name : x["rarity"]})
    
    return x

def convert_row(row):
    row["cards"] = row["cards"].apply(lambda x: convert_printings(x, row["code"]), 
                                      axis = 1).set_index("name")
    
    return row

def filter_columns(row, all_cards_cols):
    set_cols = list(row.columns)
    intersection = list(set(set_cols) & set(all_cards_cols))
    
    return row.filter(intersection)


# In[ ]:


only_cards = all_sets.apply(lambda x: convert_row(x), axis = 1)["cards"]


# In[ ]:


only_cards = only_cards.apply(lambda x: filter_columns(x, all_cards_columns))
test = only_cards["RAV"]
test.head()


# In[ ]:


all_cards = pd.concat(list(only_cards))
all_cards.sample(10)


# There are two more changes that we need to make to `all_cards`. The first is removing cards that were both in a non-tournament legal set and had another printing as a promotional card. Since we only removed non-tournament legal sets from `all_sets`, and we didn't remove cards from any of the sets themselves, these cards are still in `all_cards` and need to be removed.
# 
# Since basic lands are printed in every set, we make sure to exclude cards with the `Basic` supertype from the cards we remove.

# In[ ]:


# there are a non-tournament legal cards remaining in this list, reprinted as promos, so we remove
# those cards from the list
all_cards = all_cards.loc[~(all_cards.printings.map(lambda x: bool(set(invalid_sets) & set(x)))
              & all_cards.supertypes.map(lambda x: x != ["Basic"]))]


# The second change we need to make is removing duplicate entries in `all_cards` - we want each row to be a single card, but the way we constructed `all_cards`, each row is a printing of a single card, so each card will have an additional row for each time it has been reprinted:

# In[ ]:


all_cards.loc["Lightning Bolt"]


# When removing duplicates from `all_cards`, we also merge the dictionaries in the `printings` column for each unique card, to obtain a complete dictionary of printing/rarity pairs. This additional step is the reason why we can't use `drop_duplicates` right off the bat. The code below takes each set of reprints, merges the dictionaries in `printings`, and updates the first entry in the set of reprints with the completed dictionary.
# 
# This is the only part of this process that I couldn't figure out how to do without iteration - it might be possible to do with the `GroupBy` method but I would have to do some reading to figure out if that's the case. As a consequence, the main loop of iterating through unique card names takes a while to actually run (although previous iterations took much longer).

# In[ ]:


# merges a list of dictionaries where for each key, only one dictionary from the list will have a
# non-null value corresponding to the key. The keys of the merged dictionary will be the union of 
# the keys of the dictionaries in the list, and the corresponding value will be that non-null value
# corresponding to the key.
def merge_dicts(dicts):
    merged_dicts = {}
    
    for d in dicts:
        for k, v in d.items():
            if bool(v):
                merged_dicts.update({k : v})
    
    return merged_dicts


# In[ ]:


# loop that iterates through unique cardnames - for each cardname, check whether the card has reprints,
# and if so, update the first entry in the list of reprints with the merged printing/rarity dictionary
for cardname in all_cards.index.unique():
    reprints = all_cards.loc[cardname]
    
    # this checks that the DataFrame above actually has more than 1 card - if it had only one, then
    # reprints would instead be a column where the 16 attributes of the card are the rows
    if reprints.shape != (16,):
        merged_dicts = merge_dicts(list(reprints.printings))
        reprints.iat[0, list(reprints.columns).index("printings")].update(merged_dicts)


# In[ ]:


# for each reprinted card, the first reprint has the completed printing/rarity dictionary, so we can get
# rid of every other duplicate
all_cards = all_cards[~all_cards.index.duplicated(keep = "first")]


# Now we're done, and can check some simple summary statistics of `all_cards`.

# In[ ]:


all_cards.describe()


# In[ ]:


all_cards.sample(10)


# In[ ]:


colorless = all_cards.loc[all_cards.colors.isnull() &
              ~all_cards.types.apply(lambda x: "Land" in x)]
all_cards.loc[colorless.index, "colors"] = colorless.colors.apply(lambda x: [])


# In[ ]:


all_cards.loc["Umezawa's Jitte"]


# In[ ]:


colors = ["White", "Blue", "Black", "Red", "Green"]


# In[ ]:


def subsets(lst):
    powerset = []
    
    for i in range(len(lst)):
        powerset += map(lambda x: list(x), list(it.combinations(lst, i)))
        
    powerset.append(lst)
    return powerset


# In[ ]:


color_combos = subsets(colors)


# In[ ]:


subsets_by_color = {}

for color_combo in color_combos:
    cards = all_cards.loc[all_cards.colors.apply(lambda x: x == color_combo)]
    subsets_by_color.update({tuple(color_combo) : cards})


# In[ ]:


cmcs = all_cards.loc[:, "cmc"].dropna()


# In[ ]:


def plot_int_hist(df_, title, x_axis, y_axis, fig_x, fig_y):
    df = df_.dropna()
    num_bins = len(np.unique(df.values))
    
    fig, ax = plt.subplots(figsize = (fig_x, fig_y))
    
    n, bins, patches = ax.hist(df, num_bins, normed = True)
    
    df_mean = df.mean()
    df_std = df.std()
    y = mlab.normpdf(bins, df_mean, df_std)
    
    ax.plot(bins, y, '--')
    ax.set_ylabel(y_axis)
    ax.set_xlabel(x_axis)
    ax.set_title(title)
    plt.text(10, 0.20, "mean = " + str(round(df_mean, 5)))
    plt.text(10, 0.18, "stdev = " + str(round(df_std, 5)))
    
    fig.tight_layout()
    plt.show()


# In[ ]:


plot_int_hist(cmcs, title = "Distribution of Converted Mana Cost - All Nonland Cards",
              x_axis = "CMC", y_axis = "Percentage", fig_x = 12, fig_y = 8)


# In[ ]:


lm_pt_cmc = all_cards.loc[:, ["power", "toughness", "cmc"]]
lm_pt_cmc = lm_pt_cmc.loc[lm_pt_cmc.power.notnull() | lm_pt_cmc.toughness.notnull()]


# In[ ]:


fig, hm = plt.subplots(figsize = (15, 10))

hm.hist2d(lm_pt_cmc.power, lm_pt_cmc.toughness, bins = np.arange(-1.5, 16.5), range = ((-1, 16), (-1, 16)), 
          cmap = "summer", norm = matplotlib.colors.LogNorm())
#hm.hexbin(lm_pt_cmc.power, lm_pt_cmc.toughness, gridsize = 17, bins = "log", cmap = "summer")
hm.set_xlabel("Power")
hm.set_ylabel("Toughness")
hm.set_xticks(np.arange(-1, 16))
hm.set_yticks(np.arange(-1, 16))
hm.set_title("Power/Toughness Heatmap")


# In[ ]:


avg_cmc_pivot = pd.pivot_table(data = lm_pt_cmc, values = "cmc", index = ["power", "toughness"])
avg_cmc_pivot.index


# In[ ]:


len(avg_cmc_pivot)


# In[ ]:


avg_cmc_pivot.loc['power' == 13]


# In[ ]:


avg_cmc_pivot.loc[0.0]


# In[ ]:


unstacked = avg_cmc_pivot.unstack()
unstacked


# In[ ]:


unstacked.index


# In[ ]:


unstacked.columns


# In[ ]:


df1 = avg_cmc_pivot.index.to_frame()


# In[ ]:


df1["cmc"] = avg_cmc_pivot["cmc"]


# In[ ]:


df1


# In[ ]:


df1 = df1.loc[df1.cmc.notnull()]
df1


# In[ ]:


new_index = list(np.arange(0, 99))
df2 = pd.DataFrame(data = None, index = new_index)
df2['power'] = df1['power'].tolist()
df2['toughness'] = df1['toughness'].tolist()
df2['cmc'] = df1['cmc'].tolist()
df2


# In[ ]:


fig, hm = plt.subplots(figsize = (15, 15))

hm.scatter(df2.power, df2.toughness, s = 2250, c = df2.cmc, marker = "s", cmap = "summer")


# In[ ]:


# put the p/t heatmap and avg cmc scatter plot side by side
fig2 = plt.figure()
plt.show()


# In[ ]:




