#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Congratulations! In this section we will put all of the things that we learned together to do some truly interesting things with some datasets. 
# 
# The exercises in this section are therefore more difficult! 
# 
# While working through the exercises, tTry using method chaning syntax (use the resource below if you don't know what method chaining means). Also, take advantage the hints we provide.
# 
# # Relevant Resource
# 
# * **[Method chaining resource](https://www.kaggle.com/residentmario/method-chaining-reference). **
# 
# # Set Up
# **First, fork this notebook using the "Fork Notebook" button towards the top of the screen.**
# Run the code cell below to load data and the libraries you'll use.

# In[1]:


import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")


# # Checking Answers
# 
# Check your answers in each of the exercises that follow using the  `check_qN` function provided in the code cell above (replacing `N` with the number of the exercise). For example here's how you would check an incorrect answer to exercise 1:

# In[2]:


check_q1(pd.DataFrame())


# For the first set of questions, if you use `check_qN` on your answer, and your answer is right, a simple `True` value will be returned.
# 
# For the second set of questions, using this function to check a correct answer will present you will an informative graph!
# 
# If you get stuck, you may also use the companion `answer_qN` function to print the answer outright.
# 
# # Preview Data
# Run the cell below to preview the data

# In[3]:


chess_games.head()


# # Exercises

# **Exercise 1**: It's well-known that in the game of chess, white has a slight first-mover advantage against black. Can you measure this effect in this dataset? Use the `winner` column to create a `pandas` `Series` showing how often white wins, how often black wins, and how often the result is a tie, as a ratio of total games played. In other words, a `Series` that looks something like this:
# 
#     white    0.48
#     black    0.44
#     draw     0.08
#     Name: winner, dtype: float64
#     
# Hint: use `len` to get the length of the initial `DataFrame`, e.g. the count of all games played.

# In[4]:


# Your code here
total = len(chess_games.id)
total

chess_games['winner'].value_counts()/total


# In[5]:


check_q1(chess_games['winner'].value_counts()/total)


# **Exercise 2**: The `opening_name` field of the `chess_games` dataset provides interesting data on what the most commonly used chess openings are.  However, it gives a bit _too_ much detail, including information on the variation used for the most common opening types. For example, rather than giving `Queen's Pawn Game`, the dataset often includes `Queen's Pawn Game: Zukertort Variation`.
# 
# This makes it a bit difficult to use for categorical purposes. Here's a function that can be used to separate out the "opening archetype":
# 
#  ```python
#  lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()
#  ```
# 
# Use this function to parse the `opening_name` field and generate a `pandas` `Series` counting how many times each of the "opening archetypes" gets used. Hint: use a map.

# In[6]:


# Your code here
chess_games.head()


# In[7]:


opening_archetypes = chess_games.opening_name.map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
opening_archetypes.value_counts()


# In[8]:


check_q2(opening_archetypes.value_counts())


# **Exercise 3**: In this dataset various players play variably number of games. Group the games by `{white_id, victory_status}` and count how many times each white player ended the game in `mate` , `draw`, `resign`, etcetera. The name of the column counting how many times each outcome occurred should be `n` (hint: `rename` or `assign` may help).

# In[9]:


# Your code here
chess_games.head()


# In[10]:


chess_games.assign(n=0).groupby(['white_id','victory_status']).n.count().reset_index()


# In[11]:


check_q3(chess_games.assign(n=0).groupby(['white_id','victory_status']).n.count().reset_index())


# **Exercise 4**: There are a lot of players in the dataset who have only played one or a small handful of games. Create a `DataFrame` like the one in the previous exercise, but only include users who are in the top 20 users by number of games played as white. See if you can do this using method chaining alone! Hint: reuse the code from the previous example. Then, use `pipe`.

# In[12]:


# Your code here
df = chess_games.assign(n=0).groupby(['white_id','victory_status']).n.count().reset_index()
df2 = chess_games.white_id.value_counts().head(20)


# In[13]:


df[df.white_id.isin(df2.index)]


# In[14]:


check_q4(df[df.white_id.isin(df2.index)])


# Next, let's do some visual exercises.
# 
# The next exercise uses the following dataset:

# In[15]:


kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler


# **Exercise 5**: The Kepler space observatory is in the business of finding potential exoplanets (planets orbiting stars other suns) and, after collecting the evidence, generating whether or not to confirm, decline to confirm, or deny that a given space body is, in fact, an exoplanet. In the dataset above, the "before" status of the body is `koi_pdisposition`, and the "after" status is `koi_disposition`. 
# 
# Using the dataset above, generate a `Series` counting all of the possible transitions between pre-disposition and post-disposition. In other words, generate a `Series` whose index is a `MultiIndex` based on the `{koi_pdisposition, koi_disposition}` fields, and whose values is a count of how many times each possible combination occurred.

# In[18]:


# Your code here
kepler.groupby(['koi_pdisposition','koi_disposition']).kepid.count()


# In[19]:


check_q5(kepler.groupby(['koi_pdisposition','koi_disposition']).kepid.count())


# The next few exercises use the following datasets:

# In[20]:


wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()


# In[21]:


ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
ramen_reviews.head()


# **Exercise 6**: As we demonstrated in previous workbooks, the `points` column in the `wine_reviews` dataset is measured on a 20-point scale between 80 and 100. Create a `Series` which normalizes the ratings so that they fit on a 0-to-5 scale instead (e.g. a score of 80 translates to 1 star, while a score of 100 is five stars). Set the `Series` name to "Wine Ratings", and sort by index value (ascending).

# In[26]:


# Your code here
(((wine_reviews['points'].dropna() - 80)/4).value_counts().sort_index().rename_axis('Wine Ratings'))


# In[27]:


check_q6((((wine_reviews['points'].dropna() - 80)/4).value_counts().sort_index().rename_axis('Wine Ratings')))


# **Exercise 7**: The `Stars` column in the `ramen_reviews` dataset is the ramen equivalent to the similar data points in `wine_reviews`. Luckily it is already on a 0-to-5 scale, but it has some different problems...create a `Series` counting how many ramens earned each of the possible scores in the dataset. Convert the `Series` to the `float64` dtype and drop rames whose rating is `"Unrated"`. Set the name of the `Series` to "Ramen Ratings". Sort by index value (ascending).

# In[46]:


# Your code here
# change to float type
ramen_reviews.Stars = ramen_reviews.Stars.replace('Unrated',None).dropna().astype('float64')
# summary
ramen_reviews.Stars.value_counts().rename_axis('Ramen Ratings').sort_index()


# In[47]:


check_q7(ramen_reviews.Stars.value_counts().rename_axis('Ramen Ratings').sort_index())


# **Exercise 8**: We can see from the result of the previous exercise that whilst the wine reviewers stick to a strict 20-point scale, ramen reviews occassionally deviate into fractional numbers. Modify your answer to the previous exercise by rounding review scores to the nearest half-point (`{0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5}`). So for example a score of 4.4 should round to 4.5, a score of 4.6 to 4.5, and so on.

# In[76]:


# Your code here
a = (ramen_reviews
    .Stars
    .replace('Unrated',None)
    .dropna()
    .astype('float64')
    .map(lambda v: int(v) if v - int(v) < 0.5 else int(v) + 0.5)
    .value_counts()
    .rename_axis('Ramen Reviews')
    .sort_index()
)


# In[77]:


check_q8(a)


# # Congratulations
# 
# You've finished the Pandas track.  Many data scientist feel efficiency with Pandas is the most useful and practical skill they have, because it allows you to progress quickly in any project you have.
# 
# You can take advantage of your Pandas skills by entering a [Kaggle Competition](https://www.kaggle.com/competitions) or answering a question you find interesting using [Kaggle Datasets](https://www.kaggle.com/datasets).

# In[ ]:




