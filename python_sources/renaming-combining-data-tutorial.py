#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In most large projects, you'll end up with multiple variables or objects containing data. This can be a source of huge confusion.  
# 
# In these exercises you'll learn how to rename columns of data to keep them organized, as well as ways combine multiple data variables into a single DataFrame.
# 
# # Relevant Resources
# * **[Renaming and Combining Reference](https://www.kaggle.com/residentmario/renaming-and-combining-reference)**
# * [Essential Basic Functionality](https://pandas.pydata.org/pandas-docs/stable/basics.html#renaming-mapping-labels) section of Pandas documentation. 
# * [Merge, join, concatenate](https://pandas.pydata.org/pandas-docs/stable/merging.html) section of Pandas documentation.
# 
# # Set Up
# **First, fork this notebook using the "Fork Notebook" button towards the top of the screen.**
# Run the following cell to load your data and utility functions.

# In[1]:


import pandas as pd

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)


# Then preview the data with the following command

# In[2]:


reviews.head()


# # Checking Answers
# 
# You can check your answers in each of the exercises that follow using the  `check_qN` function provided in the code cell above (replacing `N` with the number of the exercise). For example here's how you would check an incorrect answer to exercise 1:

# In[ ]:


check_q1(pd.DataFrame())


# For the questions that follow, if you use `check_qN` on your answer, and your answer is right, a simple `True` value will be returned.
# 
# If you get stuck, you may also use the companion `answer_qN` function to print the answer outright.

# # Exercises

# **Exercise 1**: `region_1` and `region_2` are pretty uninformative names for locale columns in the dataset. Rename these columns to `region` and `locale`.

# In[7]:


# Your code here
reviews.rename(columns={'region_1': 'region','region_2': 'locale'},inplace=True)
reviews.info()


# **Exercise 2**: Set the index name in the dataset to `wines`.

# In[10]:


# Your code here
reviews.rename_axis("wines",axis='rows')
#answer_q2()


# **Exercise 3**: The [Things on Reddit](https://www.kaggle.com/residentmario/things-on-reddit/data) dataset includes product links from a selection of top-ranked forums ("subreddits") on Reddit.com. Create a `DataFrame` of products mentioned on *either* subreddit. Use the following data:

# In[11]:


gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"


# In[12]:


gaming_products.head()


# In[13]:


movie_products.head()


# Hint: before jumping into this exercise, you may want to take a minute to leaf through and familiarize yourself with the data.

# In[17]:


# Your code here
total=pd.concat([gaming_products,movie_products],join='outer')
total


# In[18]:


answer_q3()


# **Exercise 4**: The [Powerlifting Database](https://www.kaggle.com/open-powerlifting/powerlifting-database) dataset on Kaggle includes one CSV table for powerlifting meets and a separate one for powerlifting competitors. Both tables include references to a `MeetID`, a unique key for each meet (competition) included in the database. Using this, generate a dataset combining the two tables into one. Use the following data:

# In[19]:


powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")


# In[20]:


# Your code here
powerlifting_meets.head()


# In[22]:


powerlifting_competitors.head()


# In[24]:


result=pd.merge(powerlifting_meets,powerlifting_competitors,on=['MeetID'])
result.head()


# In[25]:


answer_q4()


# # Keep going
# 
# [**Continue to the method chaining workbook**](https://www.kaggle.com/residentmario/method-chaining-workbook).
