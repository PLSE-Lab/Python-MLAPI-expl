#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv("../input/en.openfoodfacts.org.products.tsv", sep="\t", nrows=100000)


# In[ ]:


data.shape


# ## Peek at the data
# 

# In[ ]:


data.head()


# ### Take-Away
# 
# * A bunch of nan-values are present. Thus not all information seem to be useful to use for machine learning tasks. Or perhaps only for some groups in the data some columns with lot of nans may be useful. 
# * Some features seem to be more informative than others. For example in the creator-feature the value "openfoodfacts-contributors" is too general and in the ingredients-feature there is a lot additional information we can extract and work with.
# 
# I like to dive through the EDA in a question-based manner. I hope we can find some interesting patterns to work with later on ;-) .  

# ## How many nans do we have?
# 
# Let's answer this questions on two different ways:
# 
# 1. Some columns may have more nans than others. Hence there are some more or less informative features.
# 2. Some products may have more nans than others. Counting the nans per product may help us later for grouping products. 
# 

# Ok, to get rid of useless features, we will explore the first question first ;-) . How many percent are nans in the data per column?

# In[ ]:


percent_of_nans = data.isnull().sum().sort_values(ascending=False) / data.shape[0] * 100


# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(percent_of_nans, bins=100, kde=False)
plt.xlabel("% nans")
plt.ylabel("Number of columns")
plt.title("Percentage of nans per feature column")


# ### Take-Away
# 
# * Most of the feature columns are not informative at all and contain close to 100 % nan-values. 
# * There is an interesting group of features that have around 20 % nan-values.
# * We can drop a lot of features. 

# ## Useless feature columns
# 
# Let's drop all those feature columns that do not contain any other information than nan. How many columns are useless?

# In[ ]:


useless_features = percent_of_nans[percent_of_nans == 100].index
useless_features


# Unfortunately there are some features that could be very useful for people with intolerances like fructose ore lactose. The feature allergens_en may be to general at all.  And what does en mean by the way? In addition the glycemic index could be interesting. Maybe we could reconstruct such kind of information from the ingredients that are given per product. 
# 
# How many useless features do we have?

# In[ ]:


len(useless_features)


# Let's drop them!

# In[ ]:


data.drop(useless_features, axis=1, inplace=True)
data.shape


# ## Zero nan features
# 
# Which feature columns are always without nan values?

# In[ ]:


zero_nan_features = percent_of_nans[percent_of_nans == 0].index
zero_nan_features


# What does the states_en and states_tags features mean? 

# In[ ]:


example = data.loc[0,zero_nan_features]
print(example["states_tags"])
print(example["states_en"])


# Ahh! :-) Here we find a lot of additional information that we may extract into separate features. For example it may help if we could have a separate feature "photo uploaded" with values "yes" or "no". This way we could perhaps build an algorithm that works with images to complete additional information that has the state "to be completed". This may be the packaging feature or categories. We will keep this in mind for feature extraction!

# What can we do with the datetime features?

# In[ ]:


example.loc[['created_datetime', 'last_modified_datetime']]


# In[ ]:


example.loc["last_modified_t"] - example.loc["created_t"]


# Would be nice if we could see how many modifications an entry has been gone through. At least we can say, if the entry was modified or not. 

# ## Splitting the nan-groups
# 
# By manually setting thresholds we are going to split the features based on their nans. After visualizing the percentage of nans per feature, we will collect features to drop and keep in mind those we want to work with during feature extraction.

# ### Low nan group

# In[ ]:


low_nans = percent_of_nans[percent_of_nans <= 15]
middle_nans = percent_of_nans[(percent_of_nans > 15) & (percent_of_nans <= 50)]
high_nans = percent_of_nans[(percent_of_nans > 50) & (percent_of_nans < 100)]


# In[ ]:


def rotate_labels(axes):
    for item in axes.get_xticklabels():
        item.set_rotation(45)


# In[ ]:


plt.figure(figsize=(20,5))
lows = sns.barplot(x=low_nans.index.values, y=low_nans.values, palette="Greens")
rotate_labels(lows)
plt.title("Features with fewest nan-values")
plt.ylabel("% of nans ")


# There is one topic that hasn't gained attention so far: Many features occur multiple times. For example:
# 
# * countries
# * countries_tags
# * countries_en
# 
# * additives
# * additives_n

# In[ ]:


data.loc[1,'additives_n']


# In[ ]:


data.loc[1,'additives']


# Hmm. That looks strange.... more than ingredients than additives. And why do we have additives_n = 0 and a bunch of additives?

# In[ ]:


data.loc[1,'ingredients_text']


# Ok, the additives feature contains the ingredients but somehow like tags and again we find this structure something -> en:something.

# In[ ]:


data.loc[1,['countries', 'countries_tags', 'countries_en']]


# Ok, three features that tell us all the same. 

# ### Middle nan group

# In[ ]:


plt.figure(figsize=(20,5))
middle = sns.barplot(x=middle_nans.index.values, y=middle_nans.values, palette="Oranges")
rotate_labels(middle)
plt.title("Features with medium number of nan-values")
plt.ylabel("% of nans ")


# Amazing! Although we have low nan-percentage for additives and additives_n, the additives_tags belong to the second group.

# In[ ]:


data.loc[7,['additives', 'additives_n', 'additives_tags']]


# In[ ]:


data.loc[7,['ingredients_text']].values


# Ok, now we can see that additives_n and additives_tags carry the information of true additives like vitamin E (with number E307) while additives is somehoe the same as ingredients_text but splitted into all subcomponents. 

# ### High nan group

# In[ ]:


plt.figure(figsize=(15,30))
high = sns.barplot(y=high_nans.index.values, x=high_nans.values, palette="Reds")
plt.title("Features with most nan-values")
plt.ylabel("% of nans ")


# Should we drop all of them? What about the packaging feature? Maybe we can extract it by using the images. What about the categories or quantity? What is meant by labels? The allergens feature would be nice as well.  

# #### Allergens

# In[ ]:


data[data['allergens'].isnull() == False][['allergens','ingredients_text']][0:10]


# We encountered a new problem... different languages. But there is new hope as well: The allergens should be mentioned in the ingrendients as well. Consequently we may extract allergens by looking at the ingredients. :-)

# #### Categories

# In[ ]:


data[data['categories'].isnull() == False].categories


# That's difficult. Again we can see the mixture of languages. One idea: The ingredients are in the low-nan-group whereas the categories are not. Perhaps it would be nicer to build categories out of some clustering of ingredients and nutrition facts such that we have more or less some kind of "natural" category which is given by the data itself. Hmm..

# ## Final Take Away
# 
# * We have discovered that some features contain much information that can be splitted into more features.
# * Some features are somehow duplicated as they tell us the same.
# * A lot of features are useless so far as they have too many nan values.
# 
# I will proceed the analysis dropping the high nan group. 

# In[ ]:


data.drop(high_nans.index, axis=1, inplace=True)
data.shape


# Ok, let's store the data as output to work with it in a new kernel:

# In[ ]:


data.to_csv("cropped_open_food_facts.csv")

