#!/usr/bin/env python
# coding: utf-8

# **[Pandas Home Page](https://www.kaggle.com/learn/pandas)**
# 
# ---
# 

# # Introduction
# 
# Now you are ready to get a deeper understanding of your data.
# 
# Run the following cell to load your data and some utility functions (including code to check your answers).

# In[ ]:


import pandas as pd
pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.summary_functions_and_maps import *
print("Setup complete.")

reviews.head()


# # Exercises

# ## 1.
# 
# What is the median of the `points` column in the `reviews` DataFrame?

# In[ ]:


median_points = reviews['points'].median()

q1.check()


# In[ ]:


#q1.hint()
#q1.solution()


# ## 2. 
# What countries are represented in the dataset? (Your answer should not include any duplicates.)

# In[ ]:


countries = reviews['country'].unique()

q2.check()


# In[ ]:


#q2.hint()
#q2.solution()


# ## 3.
# How often does each country appear in the dataset? Create a Series `reviews_per_country` mapping countries to the count of reviews of wines from that country.

# In[ ]:


reviews_per_country = reviews['country'].value_counts()

q3.check()


# In[ ]:


#q3.hint()
#q3.solution()


# ## 4.
# Create variable `centered_price` containing a version of the `price` column with the mean price subtracted.
# 
# (Note: this 'centering' transformation is a common preprocessing step before applying various machine learning algorithms.) 

# In[ ]:


centered_price = reviews['price'] - reviews['price'].mean()

q4.check()


# In[ ]:


#q4.hint()
#q4.solution()


# ## 5.
# I'm an economical wine buyer. Which wine is the "best bargain"? Create a variable `bargain_wine` with the title of the wine with the highest points-to-price ratio in the dataset.

# In[ ]:


reviews['points_to_price_ratio'] = reviews['points'] / reviews['price']
bargain_wine = reviews.iloc[reviews['points_to_price_ratio'].idxmax()]['title']
display(bargain_wine)
q5.check()


# In[ ]:


q5.hint()
q5.solution()


# ## 6.
# There are only so many words you can use when describing a bottle of wine. Is a wine more likely to be "tropical" or "fruity"? Create a Series `descriptor_counts` counting how many times each of these two words appears in the `description` column in the dataset.

# In[ ]:


tropical_count = reviews['description'].str.contains('tropical').sum()
fruit_count = reviews['description'].str.contains('fruity').sum()
descriptor_counts = pd.Series([tropical_count, fruit_count], index=['tropical', 'fruity'])

q6.check()


# In[ ]:


# q6.hint()
# q6.solution()


# ## 7.
# We'd like to host these wine reviews on our website, but a rating system ranging from 80 to 100 points is too hard to understand - we'd like to translate them into simple star ratings. A score of 95 or higher counts as 3 stars, a score of at least 85 but less than 95 is 2 stars. Any other score is 1 star.
# 
# Also, the Canadian Vintners Association bought a lot of ads on the site, so any wines from Canada should automatically get 3 stars, regardless of points.
# 
# Create a series `star_ratings` with the number of stars corresponding to each review in the dataset.

# In[ ]:


def points_to_star(value: int) -> int:
    if value > 94:
        return 3
    elif 84 < value < 95:
        return 2
    else:
        return 1


# In[ ]:


star_ratings = reviews['points'].map(points_to_star)
star_ratings[reviews.country == 'Canada'] = 3

q7.check()


# In[ ]:


# q7.hint()
q7.solution()


# # Keep going
# Continue to **[grouping and sorting](https://www.kaggle.com/residentmario/grouping-and-sorting)**.

# ---
# **[Pandas Home Page](https://www.kaggle.com/learn/pandas)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum) to chat with other Learners.*
