#!/usr/bin/env python
# coding: utf-8

# **[Pandas Micro-Course Home Page](https://www.kaggle.com/learn/pandas)**
# 
# ---
# 

# # Introduction
# Maps allow us to transform data in a `DataFrame` or `Series` one value at a time for an entire column. However, often we want to group our data, and then do something specific to the group the data is in. We do this with the `groupby` operation.
# 
# In these exercises we'll apply groupwise analysis to our dataset.
# 
# # Relevant Resources
# - [**Grouping Reference and Examples**](https://www.kaggle.com/residentmario/grouping-and-sorting-reference)  
# - [Pandas cheat sheet](https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf)

# # Set Up
# Run the code cell below to load the data before running the exercises.

# In[ ]:


import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
#pd.set_option("display.max_rows", 5)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.grouping_and_sorting import *
print("Setup complete.")


# # Exercises

# ## 1.
# Who are the most common wine reviewers in the dataset? Create a `Series` whose index is the `taster_twitter_handle` category from the dataset, and whose values count how many reviews each person wrote.

# In[ ]:


# Your code here
reviews_written = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
#reviews_written
#reviews.head()
q1.check()


# In[ ]:


#q1.hint()
#q1.solution()


# ## 2.
# What is the best wine I can buy for a given amount of money? Create a `Series` whose index is wine prices and whose values is the maximum number of points a wine costing that much was given in a review. Sort the values by price, ascending (so that `4.0` dollars is at the top and `3300.0` dollars is at the bottom).

# In[ ]:


best_rating_per_price = reviews.groupby('price').points.max()
q2.check()


# In[ ]:


#q2.hint()
#q2.solution()


# ## 3.
# What are the minimum and maximum prices for each `variety` of wine? Create a `DataFrame` whose index is the `variety` category from the dataset and whose values are the `min` and `max` values thereof.

# In[ ]:


price_extremes=pd.DataFrame({'min':reviews.groupby('variety').price.min(),'max':reviews.groupby('variety').price.max()})
q3.check()


# In[ ]:


#q3.hint()
#q3.solution()


# ## 4.
# What are the most expensive wine varieties? Create a variable `sorted_varieties` containing a copy of the dataframe from the previous question where varieties are sorted in descending order based on minimum price, then on maximum price (to break ties).

# In[ ]:


sorted_varieties = price_extremes.sort_values(by=['min','max'], ascending =False)
q4.check()


# In[ ]:


#q4.hint()
#q4.solution()


# ## 5.
# Create a `Series` whose index is reviewers and whose values is the average review score given out by that reviewer. Hint: you will need the `taster_name` and `points` columns.

# In[ ]:


reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()

q5.check()


# In[ ]:


#q5.hint()
#q5.solution()


# Are there significant differences in the average scores assigned by the various reviewers? Run the cell below to use the `describe()` method to see a summary of the range of values.

# In[ ]:


reviewer_mean_ratings.describe()


# ## 6.
# What combination of countries and varieties are most common? Create a `Series` whose index is a `MultiIndex`of `{country, variety}` pairs. For example, a pinot noir produced in the US should map to `{"US", "Pinot Noir"}`. Sort the values in the `Series` in descending order based on wine count.

# In[ ]:


country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)
q6.check()


# In[ ]:


#q6.hint()
#q6.solution()


# # Keep Going
# 
# Move on to the [**Data types and missing data workbook**](https://www.kaggle.com/kernels/fork/598827).

# ---
# **[Pandas Micro-Course Home Page](https://www.kaggle.com/learn/pandas)**
# 
# 
