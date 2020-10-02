#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Maps allow us to transform data in a `DataFrame` or `Series` one value at a time for an entire column. However, often we want to group our data, and then do something specific to the group the data is in. We do this with the `groupby` operation.
# \
# 
# # Relevant Resources
# - [**Grouping Reference and Examples**](https://www.kaggle.com/residentmario/grouping-and-sorting-reference).  
# - [Pandas cheat sheet](https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf)

# In[ ]:


import pandas as pd

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head(10)


# # Exercises

# **Exercise 1**: Who are the most common wine reviewers in the dataset? Create a `Series` whose index is the `taster_twitter_handle` category from the dataset, and whose values count how many reviews each person wrote.

# In[ ]:


reviews.groupby('taster_twitter_handle').count()


# **Exercise 2**: What is the best wine I can buy for a given amount of money? Create a `Series` whose index is wine prices and whose values is the maximum number of points a wine costing that much was given in a review. Sort the valeus by price, ascending (so that `4.0` dollars is at the top and `3300.0` dollars is at the bottom).

# In[ ]:


reviews.groupby('price').points.count()


# **Exercise 3**: What are the minimum and maximum prices for each `variety` of wine? Create a `DataFrame` whose index is the `variety` category from the dataset and whose values are the `min` and `max` values thereof.

# In[ ]:


reviews.groupby('variety').price.agg([min,max])


# The rest of the exercises are visual.
# 
# **Exercise 4**: Are there significant differences in the average scores assigned by the various reviewers? Create a `Series` whose index is reviewers and whose values is the average review score given out by that reviewer. Hint: you will need the `taster_name` and `points` columns.

# In[ ]:


reviews.groupby('taster_name').points.mean()


# **Exercise 5**: What are the most expensive wine varieties? Create a `DataFrame` whose index is wine varieties and whose values are columns with the `min` and the `max` price of wines of this variety. Sort in descending order based on `min` first, `max` second.

# In[ ]:


reviews.groupby('variety').price.agg([min, max]).sort_values(by=['min', 'max'], ascending=False)
#or
#x = pd.DataFrame( {'min': reviews.groupby('variety').price.min(),'max': reviews.groupby('variety').price.max()},index = reviews.variety)
#x.head()


# **Exercise 6**: What combination of countries and varieties are most common? Create a `Series` whose index is a `MultiIndex`of `{country, variety}` pairs. For example, a pinot noir produced in the US should map to `{"US", "Pinot Noir"}`. Sort the values in the `Series` in descending order based on wine count.
# 
# Hint: first run `reviews['n'] = 0`. Then `groupby` the dataset and run something on the column `n`. You won't need `reset_index`.
# 
# reviews['n'] = 0
# reviews.groupby('')

# In[ ]:


reviews.groupby(['country', 'variety']).n.count().sort_values(ascending=False)


# # Keep Going
# 
# Move on to [**Data types and missing data workbook**](https://www.kaggle.com/residentmario/data-types-and-missing-data-workbook).
# 
# ___
# This is part of the [*Learn Pandas*](https://www.kaggle.com/learn/pandas) series.
