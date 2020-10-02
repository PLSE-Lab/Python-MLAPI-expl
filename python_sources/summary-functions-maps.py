#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# You've learned how to select relevant data out of our `pandas` `DataFrame` and `Series` objects. Plucking the right data out of our data representation is critical to getting work done, as we demonstrated in the visualization exercises attached to the workbook.
# 
# However, the data does not always in the format we want it in right out of the bat. Sometimes we have to do some more work ourselves to reformat it for our task.
# 
# The remainder of this tutorial will cover different operations we can apply to our data to get the input "just right". We'll start off in this section by looking at the most commonly looked built-in reshaping operations. Along the way we'll cover data `dtypes`, a concept essential to working with `pandas` effectively.
# 
# # Relevant Resources
# * **[Summary functions and maps](https://www.kaggle.com/residentmario/summary-functions-and-maps-reference)**
# * [Official pandas cheat sheet](https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf)
# 
# # Set Up
# **First, fork this notebook using the "Fork Notebook" button towards the top of the screen.**
# 
# Run the code cell below to load your data and the necessary utility funcitons.

# In[1]:


import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)


# # Checking Answers
# 
# You can check your answers in each of the exercises that follow using the  `check_qN` function provided in the code cell above (replacing `N` with the number of the exercise). For example here's how you would check an incorrect answer to exercise 1:

# In[ ]:


check_q1(20)


# For the first set of questions, if you use `check_qN` on your answer, and your answer is right, a simple `True` value will be returned.
# 
# For the second set of questions, using this function to check a correct answer will present you will an informative graph!
# 
# If you get stuck, use the companion `answer_qN` function to print the answer.

# # Preview the Data
# 
# Run the cell below to preview your data

# In[2]:


reviews.head()


# # Exercises

# **Exercise 1**: What is the median of the `points` column?

# In[3]:


# Your code here
reviews.points.median ()


# **Exercise 2**: What countries are represented in the dataset?

# In[6]:


# Your code here
reviews.country.unique()


# **Exercise 3**: What countries appear in the dataset most often?

# In[8]:


# Your code here
countries_max = reviews.country.value_counts()
print (countries_max)


# **Exercise 4**: Remap the `price` column by subtracting the median price.

# In[ ]:


# Your code here
price_median = reviews.price.median ()
reviews.price.map (lambda p: p - price_median)


# **Exercise 5**: I"m an economical wine buyer. What is the name (`title`) of the "best bargain" wine, e.g., the one which has the highest points-to-price ratio in the dataset?
# 
# Hint: use a map and the [`idxmax` function](http://pandas.pydata.org/pandas-docs/version/0.19.2/generated/pandas.Series.idxmax.html).

# In[23]:


# Your code here
reviews['points_to_price'] = reviews.points/reviews.price
max_index = reviews.points_to_price.idxmax (skipna = True)
reviews.loc [max_index, 'title']


# Now it's time for some visual exercises.

# **Exercise 6**: There are only so many words you can use when describing a bottle of wine. Is a wine more likely to be "tropical" or "fruity"? Create a `Series` counting how many times each of these two words appears in the `description` column in the dataset.
# 
# Hint: use a map to check each description for the string `tropical`, then count up the number of times this is `True`. Repeat this for `fruity`. Create a `Series` combining the two values at the end.

# In[40]:


# Your code here
count_tropical = reviews['description'].str.contains('tropical').value_counts ()[True]
count_fruity = reviews['description'].str.contains('fruity').value_counts ()[True]
# print(count_tropical)
# print(count_fruity)
pd.Series([count_tropical, count_fruity], index=['Tropical','Fruity'])


# **Exercise 7**: What combination of countries and varieties are most common?
# 
# Create a `Series` whose index consists of strings of the form `"<Country> - <Wine Variety>"`. For example, a pinot noir produced in the US should map to `"US - Pinot Noir"`. The values should be counts of how many times the given wine appears in the dataset. Drop any reviews with incomplete `country` or `variety` data.
# 
# Note that some of the `Country` and `Wine Variety` values are missing data. We will learn more about missing data in a future section of the tutorial. For now you may use the included code snippet to normalize these columns.
# 
# Hint:  Use a map to create a series whose entries are a `str` concatenation of those two columns. Then, generate a `Series` counting how many times each label appears in the dataset.

# In[31]:


# Your code here

reviews_dropped_rows = reviews.dropna (subset = ['country','variety'])
# print ("Before dropping empty values, there were %d rows and %d columns \n" % (reviews.shape[0],  reviews.shape[1]))
# print ("Before dropping empty values, there were %d rows and %d columns \n" % (reviews_dropped_rows.shape[0],  reviews_dropped_rows.shape[1]))

country_winevariety = reviews_dropped_rows ['country'] + ' - ' + reviews_dropped_rows ['variety']
country_wine_count = country_winevariety.value_counts()
print (country_wine_count)


# # Keep going
# **[Continue to grouping and sorting](https://www.kaggle.com/residentmario/grouping-and-sorting-workbook).**