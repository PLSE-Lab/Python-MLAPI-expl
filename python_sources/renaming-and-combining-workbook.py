#!/usr/bin/env python
# coding: utf-8

# # Renaming and combining workbook
# 
# ## Introduction
# 
# This is the worbook part of the "Renaming and combining" section of the Advanced Pandas tutorial. For the reference section, click [here](https://www.kaggle.com/residentmario/renaming-and-combining-reference).
# 
# Renaming is covered in its own section in the ["Essential Basic Functionality"](https://pandas.pydata.org/pandas-docs/stable/basics.html#renaming-mapping-labels) section of the extensive official documentation. Combining is covered by the ["Merge, join, concatenate"](https://pandas.pydata.org/pandas-docs/stable/merging.html) section there.

# In[1]:


import pandas as pd

from learntools.advanced_pandas.renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)


# # Checking Answers
# 
# **Check your answers in each exercise using the  `check_qN` function** (replacing `N` with the number of the exercise). For example here's how you would check an incorrect answer to exercise 1:

# In[2]:


check_q1(pd.DataFrame())


# If you get stuck, **use the `answer_qN` function to see the code with the correct answer.**
# 
# For the first set of questions, running the `check_qN` on the correct answer returns `True`.
# 
# For the second set of questions, using this function to check a correct answer will present an informative graph!

# # Exercises
# 
# Look at your data by running the cell below:

# In[3]:


reviews.head()


# **Exercise 1**: `region_1` and `region_2` are pretty uninformative names for locale columns in the dataset. Rename these columns to `region` and `locale`.

# In[9]:


# Your code here
reviews.rename(columns={'region_1':'region','region_2':'locale'})


# In[8]:


check_q1(reviews.rename(columns={'region_1':'region','region_2':'locale'}))


# **Exercise 2**: Set the index name in the dataset to `wines`.

# In[18]:


# Your code here
reviews.rename_axis('wines')


# In[17]:


print(answer_q2())


# In[21]:


check_q2(reviews.rename_axis("wines", axis="rows"))


# **Exercise 3**: The [Things on Reddit](https://www.kaggle.com/residentmario/things-on-reddit/data) dataset includes product links from a selection of top-ranked forums ("subreddits") on Reddit.com. Create a `DataFrame` of products mentioned on *either* subreddit. Use the following data:

# In[22]:


gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"


# In[25]:


# Your code here
movie_products.head()


# In[27]:


gaming_products.head()


# In[29]:


pd.concat([gaming_products,movie_products])


# In[31]:


check_q3(pd.concat([gaming_products,movie_products]))


# **Exercise 4**: The [Powerlifting Database](https://www.kaggle.com/open-powerlifting/powerlifting-database) dataset on Kaggle includes one CSV table for powerlifting meets and a separate one for powerlifting competitors. Both tables include references to a `MeetID`, a unique key for each meet (competition) included in the database. Using this, generate a dataset combining the two tables into one. Use the following data:

# In[32]:


powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")


# In[34]:


powerlifting_meets.head()


# In[35]:


powerlifting_competitors.head()


# In[48]:


df = pd.merge(powerlifting_meets,powerlifting_competitors,on='MeetID',how='outer')
df.set_index('MeetID')


# In[49]:


check_q4(df.set_index('MeetID'))


# In[ ]:




