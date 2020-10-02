#!/usr/bin/env python
# coding: utf-8

# **[Pandas Micro-Course Home Page](https://www.kaggle.com/learn/pandas)**
# 
# ---
# 

# # Summary functions and maps workbook
# 
# ## Introduction
# 
# This is the workbook component to the "Summary functions and maps" section of the Advanced Pandas tutorial. For the reference section, [**click here**](https://www.kaggle.com/residentmario/summary-functions-and-maps-reference).
# 
# In the last section we learned how to select relevant data out of our `pandas` `DataFrame` and `Series` objects. Plucking the right data out of our data representation is critical to getting work done, as we demonstrated in the visualization exercises attached to the workbook.
# 
# However, the data does not always come out of memory in the format we want it in right out of the bat. Sometimes we have to do some more work ourselves to reformat it for the task at hand.
# 
# The remainder of this tutorial will cover different operations we can apply to our data to get the input "just right". We'll start off in this section by looking at the most commonly looked built-in reshaping operations. Along the way we'll cover data `dtypes`, a concept essential to working with `pandas` effectively.

# In[ ]:


import pandas as pd
#pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.summary_functions_and_maps import *
print("Setup complete.")

reviews.head(20)


# ## Exercises

# ## 1.
# 
# What is the median of the `points` column in the `reviews` DataFrame?

# In[ ]:


median_points = reviews.points.median()

q1.check()


# In[ ]:


#q1.hint()
#q1.solution()


# ## 2. 
# What countries are represented in the dataset? (Your answer should not include any duplicates.)

# In[ ]:


countries = reviews.country.unique()

q2.check()


# In[ ]:


#q2.hint()
#q2.solution()


# ## 3.
# How often does each country appear in the dataset? Create a Series `reviews_per_country` mapping countries to the count of reviews of wines from that country.

# In[ ]:


reviews_per_country = reviews.country.value_counts()

q3.check()


# In[ ]:


#q3.hint()
#q3.solution()


# ## 4.
# Create variable `centered_price` containing a version of the `price` column with the mean price subtracted.
# 
# (Note: this 'centering' transformation is a common preprocessing step before applying various machine learning algorithms.) 

# In[ ]:


mean_price=reviews.price.mean()
centered_price = reviews.price.map(lambda p: p-mean_price)
q4.check()


# In[ ]:


#q4.hint()
#q4.solution()


# ## 5.
# I'm an economical wine buyer. Which wine is the "best bargain"? Create a variable `bargain_wine` with the title of the wine with the highest points-to-price ratio in the dataset.

# In[ ]:


index= (reviews.points/reviews.price).idxmax()
bargain_wine=reviews.loc[index,'title']
q5.check()


# In[ ]:


#q5.hint()
#q5.solution()


# ## 6.
# There are only so many words you can use when describing a bottle of wine. Is a wine more likely to be "tropical" or "fruity"? Create a Series `descriptor_counts` counting how many times each of these two words appears in the `description` column in the dataset.

# In[ ]:


import numpy as np
trop=reviews.description.map(lambda x: 'tropical' in x)
fruity=reviews.description.map(lambda x: 'fruity' in x)
trop=np.sum(trop)
fruity=np.sum(fruity)
descriptor_counts=pd.Series([trop,fruity], index=['tropical','fruity'])
q6.check()


# In[ ]:


#q6.hint()
#q6.solution()


# ## 7.
# We'd like to host these wine reviews on our website, but a rating system ranging from 80 to 100 points is too hard to understand - we'd like to translate them into simple star ratings. A score of 95 or higher counts as 3 stars, a score of at least 85 but less than 95 is 2 stars. Any other score is 1 star.
# 
# Also, the Canadian Vintners Association bought a lot of ads on the site, so any wines from Canada should automatically get 3 stars, regardless of points.
# 
# Create a series `star_ratings` with the number of stars corresponding to each review in the dataset.

# In[ ]:


def fun(row):
    if row.points>=95:
        row.points=3
    elif row.points>=85 and row.points<95:
        row.points=2
    elif row.country=='Canada':
        row.points=3
    else:
        row.points=1
    return row
val = reviews.apply(fun, axis='columns')
star_ratings=val.loc[:,'points']
q7.check()


# In[ ]:


#q7.hint()
#q7.solution()


# # Keep going
# Continue to **[grouping and sorting](https://www.kaggle.com/kernels/fork/598715)**.

# ---
# **[Pandas Micro-Course Home Page](https://www.kaggle.com/learn/pandas)**
# 
# 
