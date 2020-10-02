#!/usr/bin/env python
# coding: utf-8

# ## You going to eat that?
# **Data mining/analysis of the UCI ML mushroom dataset**
# 
# This dataset caught my eye because what could be more high stakes than life and death? I'm not the type of person who often finds themselves picking random mushrooms growing and eating them, but it be cool to know if there was a quick way to identify whether or not it be safe to eat. With that in mind, this kernel hopes to answer the question, "are there any heuristics for identifying whether or not a mushroom is poisonous"?
# 
# Let's dig in!

# In[ ]:


# Imports
import numpy as np 
import pandas as pd

# Load the dataset
mushrooms = pd.read_csv("../input/mushrooms.csv")
# Check the dataframe shape
print("Shape:", mushrooms.shape, "\n")

# Check out the columns and data types
print(mushrooms.dtypes, "\n\n")

# Look at the first few rows
print(mushrooms.head())


# Looks like we have 8124 observations with 23 columns, 22 of which are features of the mushroom and 1 being its classification (edible or poisonous). Each feature is **nominal**, meaning they are of discrete values with (seemingly) no numerical relationship between the different categories. I make this distinction because I feel it would be a mistake to immediately use sklearn's LabelEncoder, or something equivalent to it, to encode each column to some numeric values. This is because it would imply that one value is "better" than the other, aka veil-color yellow is "greater than" veil-color brown, without first anaylizing the data. I don't want to jump the gun and start convincing myself of trends in the data that only exist because I applied an artificial order to the values.
# 
# Now then...
# 
# The first thing I want to do is get a better understanding of the data and make myself at home. I'm not a fan of having to flip back and forth from the notebook to the legend just to remember what the single letters for each feature stand for, so I'll go ahead and replace those with the actual words.

# In[ ]:


# Column Legends
classes = {'p': 'poisonous', 'e': 'edible'}
cap_shape = {'x': 'convex', 'b': 'bell', 's': 'sunken', 'f': 'flat', 'k': 'knobbed', 'c': 'conical'}
cap_surface = {'s': 'smooth', 'y': 'scaly', 'f': 'fibrous', 'g': 'grooves'}
cap_color = {'n': 'brown', 'y': 'yellow', 'w': 'white', 'g': 'gray', 'e': 'red', 'p': 'pink', 'b': 'buff', 'u': 'purple', 'c': 'cinnamon', 'r': 'green'}
bruises = {'t': 'yes', 'f': 'no'}
odor = {'p': 'pungent', 'a': 'almond', 'l': 'anise', 'n': 'none', 'f': 'foul', 'c': 'creosote', 'y': 'fishy', 's': 'spicy', 'm': 'musty'}
gill_attachment = {'f': 'free', 'a': 'attached', 'd': 'descending', 'n': 'notched'}
gill_spacing = {'c': 'close', 'w': 'crowded', 'd': 'distant'}
gill_size = {'n': 'narrow', 'b': 'broad'}
gill_color = {'k': 'black', 'n': 'brown', 'g': 'gray', 'p': 'pink', 'w': 'white', 'h': 'chocolate', 'u': 'purple', 'e': 'red', 'b': 'buff', 'r': 'green', 'y': 'yellow', 'o': 'orange'}
stalk_shape = {'e': 'enlarging', 't': 'tapering'}
stalk_root = {'e': 'equal', 'c': 'club', 'b': 'bulbous', 'r': 'rooted', '?': 'missing', 'e': 'equal', 'z': 'rhhizomorphs'}
stalk_surface_above_ring = {'s': 'smooth', 'f': 'fibrous', 'k': 'silky', 'y': 'scaly'}
stalk_surface_below_ring = {'s': 'smooth', 'f': 'fibrous', 'y': 'scaly', 'k': 'silky'}
stalk_color_above_ring = {'w': 'white', 'g': 'gray', 'p': 'pink', 'n': 'brown', 'b': 'buff', 'e': 'red', 'o': 'orange', 'c': 'cinnamon', 'y': 'yellow'}
stalk_color_below_ring = {'w': 'white', 'p': 'pink', 'g': 'gray', 'b': 'buff', 'n': 'brown', 'e': 'red', 'y': 'yellow', 'o': 'orange', 'c': 'cinnamon'}
veil_type = {'p': 'partial', 'u': 'universal'}
veil_color = {'w': 'white', 'n': 'brown', 'o': 'orange', 'y': 'yellow'}
ring_number = {'o': 'one', 't': 'two', 'n': 'none'}
ring_type = {'p': 'pendant', 'e': 'evanescent', 'l': 'large', 'f': 'flaring', 'n': 'none', 'c': 'cobwebby', 's': 'sheathing', 'z': 'zone'}
spore_print_color = {'k': 'black', 'n': 'brown', 'u': 'purple', 'h': 'chocolate', 'w': 'white', 'r': 'green', 'o': 'orange', 'y': 'yellow', 'b': 'buff'}
population = {'s': 'scattered', 'n': 'numerous', 'a': 'abundant', 'v': 'several', 'y': 'solitary', 'c': 'clustered'}
habitat = {'u': 'urban', 'g': 'grasses', 'm': 'meadows', 'd': 'woods', 'p': 'paths', 'w': 'waste', 'l': 'leaves'}

# New Dataset
full_names = mushrooms.copy()
full_names['class'] = full_names['class'].replace(classes)
full_names['cap-shape'] = full_names['cap-shape'].replace(cap_shape)
full_names['cap-surface'] = full_names['cap-surface'].replace(cap_surface)
full_names['cap-color'] = full_names['cap-color'].replace(cap_color)
full_names['bruises'] = full_names['bruises'].replace(bruises)
full_names['odor'] = full_names['odor'].replace(odor)
full_names['gill-attachment'] = full_names['gill-attachment'].replace(gill_attachment)
full_names['gill-spacing'] = full_names['gill-spacing'].replace(gill_spacing)
full_names['gill-size'] = full_names['gill-size'].replace(gill_size)
full_names['gill-color'] = full_names['gill-color'].replace(gill_color)
full_names['stalk-shape'] = full_names['stalk-shape'].replace(stalk_shape)
full_names['stalk-root'] = full_names['stalk-root'].replace(stalk_root)
full_names['stalk-surface-above-ring'] = full_names['stalk-surface-above-ring'].replace(stalk_surface_above_ring)
full_names['stalk-surface-below-ring'] = full_names['stalk-surface-below-ring'].replace(stalk_surface_below_ring)
full_names['stalk-color-above-ring'] = full_names['stalk-color-above-ring'].replace(stalk_color_above_ring)
full_names['stalk-color-below-ring'] = full_names['stalk-color-below-ring'].replace(stalk_color_below_ring)
full_names['veil-type'] = full_names['veil-type'].replace(veil_type)
full_names['veil-color'] = full_names['veil-color'].replace(veil_color)
full_names['ring-number'] = full_names['ring-number'].replace(ring_number)
full_names['ring-type'] = full_names['ring-type'].replace(ring_type)
full_names['spore-print-color'] = full_names['spore-print-color'].replace(spore_print_color)
full_names['population'] = full_names['population'].replace(population)
full_names['habitat'] = full_names['habitat'].replace(habitat)

print(full_names.head())


# Much better. Next I'd like to know if there is any missing data and how many observations of each class we have to work with.

# In[ ]:


print(full_names.isnull().sum(), "\n\n")
print(full_names['class'].value_counts())


# There are no nulls in the dataset, which is great, and the dataset is pretty balanced. Now I'll do some cross tabulation to get a better sense of each classes features, and the proportions of values within them.

# In[ ]:


for c in full_names.columns[1:]:
    counts = full_names[c].value_counts()
    feature_table = pd.crosstab(full_names[c], full_names['class'], normalize='index')
    feature_table['totals'] = counts
    
    print("-- {} --\n".format(c))
    print(feature_table.sort_values(by=['totals']))
    print("\n\n")


# If you don't feel like pouring through each of these tables, I'll tell you the things that jumped out at me. 
# * Of the 3528 mushrooms that had no odor, about 97% of them were edible.
# * The mushrooms that did have an odor, other than "almond" and "anise", were all poisonous.
# * Veil type is useless, as there is only one value for it in the dataset.
# 
# At this point, it seems like knowing the odor for a given mushroom is a good way to indicate if it is safe to eat or not, i.e. odor is correlated with the class. Before moving forward, I'll drop out veil type because it have the one value.

# In[ ]:


full_names = full_names.drop(columns=['veil-type'])
print(full_names.columns)


# I next wanted to determine the strength of the correlation between odor (and the other features) with its class. After hunnting around for ways to quantify correlation between categorical data, I came across this great [article](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9). Oddly enough, he was looking at this very same data set. 
# 
# In the article, Shaked gives an interesting description of correlation in general and how it can be quantified for categorical data. He uses, and I will as well, Theil's U aka the Uncertainty Coefficient. This measure captures the asymmetric relationship between the mushroom's features and class.
# 
# Below I create a heatmap for the Uncertainty Coefficient of the features compared to the classes.

# In[ ]:


import math
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def conditional_entropy(x,y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def theil_u(x,y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


theilu = pd.DataFrame(index=['class'],columns=full_names.columns)
columns = full_names.columns
for j in range(0,len(columns)):
    u = theil_u(full_names['class'].tolist(),full_names[columns[j]].tolist())
    theilu.loc[:,columns[j]] = u
theilu.fillna(value=np.nan,inplace=True)
plt.figure(figsize=(20,1))
sns.heatmap(theilu,annot=True,fmt='.2f')
plt.show()


# As you can see, this backs up the observation that a mushroom's odor seems highly correlated with its class. Based on this, if I was dying of hunger in the woods and came across wild mushrooms, I would eat them if they had no smell, or smelled like almonds or anise (like licorice).
# 
# Thanks for reading!

# 
