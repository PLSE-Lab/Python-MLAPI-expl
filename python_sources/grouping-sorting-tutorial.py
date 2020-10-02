#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Maps allow us to transform data in a `DataFrame` or `Series` one value at a time for an entire column. However, often we want to group our data, and then do something specific to the group the data is in. We do this with the `groupby` operation.
# 
# In these exercises we'll apply groupwise analysis to our dataset.
# 
# # Relevant Resources
# - [**Grouping Reference and Examples**](https://www.kaggle.com/residentmario/grouping-and-sorting-reference).  
# - [Pandas cheat sheet](https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf)

# # Set Up
# **First, fork this notebook using the "Fork Notebook" button towards the top of the screen.**
# 
# Run the code cell below to load the data before running the exercises.

# In[1]:


import pandas as pd

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)


# # Checking Your Answers
# 
# You can check your answers in each of the exercises that follow using the  `check_qN` function provided in the code cell above (replacing `N` with the number of the exercise). For example here's how you would check an incorrect answer to exercise 1:

# In[ ]:


check_q1(pd.DataFrame())


# For the "visual questions", using this function to check a correct answer will present you will an informative graph!
# 
# If you get stuck, you may also use the companion `answer_qN` function to print the answer outright.
# 

# # Exercises

# **Exercise 1**: Who are the most common wine reviewers in the dataset? Create a `Series` whose index is the `taster_twitter_handle` category from the dataset, and whose values count how many reviews each person wrote.

# In[12]:


# Your code here
# common_wine_reviewers = _______
common_wine_reviewers=reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
common_wine_reviewers
#check_q1(common_wine_reviewers)
#answer_q1()


# **Exercise 2**: What is the best wine I can buy for a given amount of money? Create a `Series` whose index is wine prices and whose values is the maximum number of points a wine costing that much was given in a review. Sort the valeus by price, ascending (so that `4.0` dollars is at the top and `3300.0` dollars is at the bottom).

# In[18]:


# Your code here
#best_wine = reviews.groupby('price').apply(lambda df:df.loc[df.points.argmax()]).sort_values(by='price')
best_wine=reviews.groupby('price').points.max().sort_index()
best_wine
#check_q2(best_wine)
#answer_q2()


# **Exercise 3**: What are the minimum and maximum prices for each `variety` of wine? Create a `DataFrame` whose index is the `variety` category from the dataset and whose values are the `min` and `max` values thereof.

# In[20]:


# Your code here
wine_price_extremes =reviews.groupby('variety').price.agg([min,max])
wine_price_extremes
#check_q3(wine_price_extremes)


# The rest of the exercises are visual.
# 
# **Exercise 4**: Are there significant differences in the average scores assigned by the various reviewers? Create a `Series` whose index is reviewers and whose values is the average review score given out by that reviewer. Hint: you will need the `taster_name` and `points` columns.

# In[28]:


# Your code here
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
reviewer_mean_ratings
# check_q4(reviewer_mean_rating)
#answer_q4()


# **Exercise 5**: What are the rarest, most expensive wine varieties? Create a `DataFrame` whose index is wine varieties and whose values are columns with the `min` and the `max` price of wines of this variety. Sort in descending order based on `min` first, `max` second.

# In[31]:


# Your code here
wine_price_range =reviews.groupby('variety').price.agg([min,max]).sort_values(by=['min','max'],ascending=False)
wine_price_range
check_q5(wine_price_range)


# **Exercise 6**: What combination of countries and varieties are most common? Create a `Series` whose index is a `MultiIndex`of `{country, variety}` pairs. For example, a pinot noir produced in the US should map to `{"US", "Pinot Noir"}`. Sort the values in the `Series` in descending order based on wine count.
# 
# Hint: first run `reviews['n'] = 0`. Then `groupby` the dataset and run something on the column `n`. You won't need `reset_index`.

# In[34]:


# Your code here
reviews['n']=0
country_variety_pairs = reviews.groupby(['country','variety']).n.count().sort_values(ascending=False)
country_variety_pairs
check_q6(country_variety_pairs)


# # Keep Going
# 
# Move on to [**Data types and missing data workbook**](https://www.kaggle.com/residentmario/data-types-and-missing-data-workbook).
# 
# ___
# This is part of the [*Learn Pandas*](https://www.kaggle.com/learn/pandas) series.
