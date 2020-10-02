#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# You have learned how to select relevant data from `DataFrame` and `Series` objects. Plucking the right data out of our data representation is critical to getting work done.
# 
# However, the data does not always come in the format we want. Sometimes we have to do some more work ourselves to reformat it for our desired task.
# 
# The remainder of this tutorial will cover different operations we can apply to our data to get the input "just right". We'll start off in this section by looking at the most commonly looked built-in reshaping operations. Along the way we'll cover data `dtypes`, a concept essential to working with `pandas` effectively.

# # Relevant Resources
# * **[Summary functions and maps](https://www.kaggle.com/residentmario/summary-functions-and-maps-reference)**
# * [Official pandas cheat sheet](https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf)
# 
# # Set Up
# Run the code cell below to load your data and the necessary utility functions.

# In[ ]:


import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)


# Look at an overview of your data by running the line below:

# # Checking Answers
# 
# **Check your answers in each exercise using the  `check_qN` function** (replacing `N` with the number of the exercise). For example here's how you would check an incorrect answer to exercise 1:

# In[ ]:


check_q1(pd.DataFrame())


# If you get stuck, **use the `answer_qN` function to see the code with the correct answer.**
# 
# For the first set of questions, running the `check_qN` on the correct answer returns `True`.
# 
# For the second set of questions, using this function to check a correct answer will present an informative graph!
# 

# ## Exercises
# 
# Look at your data by running the cell below:

# In[ ]:


reviews.head()


# **Exercise 1**: What is the median of the `points` column?

# In[ ]:


# Your code here
reviews['points'].median()


# **Exercise 2**: What countries are represented in the dataset?

# In[ ]:


# Your code here
reviews.country.unique()


# **Exercise 3**: What countries appear in the dataset most often?

# In[ ]:


reviews.country.mode()


# **Exercise 4**: Remap the `price` column by subtracting the median price. Use the `Series.map` method.

# In[ ]:


med = reviews['price'].median()
reviews.price.map(lambda p: p - med)


# **Exercise 5**: I"m an economical wine buyer. Which wine in is the "best bargain", e.g., which wine has the highest points-to-price ratio in the dataset?
# 
# Hint: use a map and the [`argmax` function](http://pandas.pydata.org/pandas-docs/version/0.19.2/generated/pandas.Series.argmax.html).

# In[ ]:


highest_points_ratio_index = (reviews.points / reviews.price).idxmax()
reviews.loc[highest_points_ratio_index].title


# Now it's time for some visual exercises. In the questions that follow, generate the data that we will need to have in order to produce the plots that follow. These exercises will use skills from this workbook as well as from previous ones. They look a lot like questions you will actually be asking when working with your own data!

# <!--
# **Exercise 6**: Sometimes the `province` and `region_1` provided in the dataset is the same value. Create a `Series` whose values counts how many times this occurs (`True`) and doesn't occur (`False`).
# -->

# **Exercise 6**: Is a wine more likely to be "tropical" or "fruity"? Create a `Series` counting how many times each of these two words appears in the `description` column in the dataset.
# 
# Hint: use a map to check each description for the string `tropical`, then count up the number of times this is `True`. Repeat this for `fruity`. Create a `Series` combining the two values at the end.

# In[ ]:


has_tropical = reviews['description'].map(lambda r: "tropical" in r)
tropical = reviews[has_tropical]
has_fruity = reviews['description'].map(lambda r: "fruity" in r)
fruity = reviews[has_fruity]
s = pd.Series([tropical['title'].count(),fruity['title'].count()],index=['tropical','fruity'])
check_q7(s)


# **Exercise 7**: What combination of countries and varieties are most common?
# 
# Create a `Series` whose index consists of strings of the form `"<Country> - <Wine Variety>"`. For example, a pinot noir produced in the US should map to `"US - Pinot Noir"`. The values should be counts of how many times the given wine appears in the dataset. Drop any reviews with incomplete `country` or `variety` data.
# 
# Hint: you can do this in three steps. First, generate a `DataFrame` whose `country` and `variety` columns are non-null. Then use a map to create a series whose entries are a `str` concatenation of those two columns. Finally, generate a `Series` counting how many times each label appears in the dataset.

# In[ ]:


not_null = reviews.dropna(subset=['country', 'variety'])
reviews['new_index'] = reviews['country'] + ' - ' + reviews['variety']
counts = reviews['new_index'].value_counts()
check_q8(counts)


# # Keep Going
# **[Continue to grouping and sorting](https://www.kaggle.com/kernels/fork/598715).**
