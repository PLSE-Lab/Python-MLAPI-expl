#!/usr/bin/env python
# coding: utf-8

# ## Load the sample of the Jester jokes recomendation dataset 
# * Melt data into tidy format
# * Add  target based features for creating and evaluating recommender system (e.g. normalize by baseline of the user or Item's average or median rating)
# * Minor EDA
# 
# * Original data (including more ratings +- more jokes): http://eigentaste.berkeley.edu/dataset/ 

# In[ ]:


import pandas as pd
import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/UserRatings1.csv")
print(df.shape)
df.head()


# ### Melt data into tidy format: 1 row per Joke X User Rating
# * we'll also drop items that were not rated (NaNs/blanks), although they could be included, e.g. for implicit feedback systems or features based on # jokes rated. 
#     * We only have 100 jokes in this version of the data

# In[ ]:


df = df.melt(id_vars="JokeId",value_name="rating")


# In[ ]:


df.dropna(inplace=True)
df.variable = df.variable.str.replace("User","")
df.rename(columns={"variable":"User"},inplace=True)
print(df.shape)
df.head()


# In[ ]:


df["mean_joke_rating"] = df.groupby("JokeId")["rating"].transform("mean")
df["mean_user_rating"] = df.groupby("User")["rating"].transform("mean")
df["user-count"] = df.groupby("User")["rating"].transform("count")
df["joke-count"] = df.groupby("JokeId")["rating"].transform("count")
df.describe()


# #### Note that we have a very dense user-item interaction matrix here! 
# * users rated 56-100 jokes = "no cold start" problem!

# In[ ]:


df.head()


# ## Merge with the text data
# * Optional - can use for additional  metadata or pretrained embeddings. Makes file larger and diverges from simplest triplet ({user} {item} {rating}) format 

# In[ ]:


df = df.merge(pd.read_csv("../input/JokeText.csv"),on="JokeId")
df.tail()


# ## The funniest jokes?
# * Note that the mean and median are very different here!.
# * We might even want to sort by Z-score (i.e normalize by each user's average rating first, as some people are just sourpusses, ).
# * We could even get the [bayesian average](http://www.evanmiller.org/bayesian-average-ratings.html) : http://www.evanmiller.org/bayesian-average-ratings.html  ,  https://fulmicoton.com/posts/bayesian_rating/   
#     * i.e use the prior mean rating of all jokes, and look at how many times a joke was rated

# In[ ]:


df.rating.mean()


# In[ ]:


df.rating.median()


# In[ ]:


df3 = df.drop_duplicates(subset=["JokeId"]).loc[:,['JokeId', 'mean_joke_rating', 'joke-count', 'JokeText']].sort_values('mean_joke_rating',ascending=False)
df3.head()


# In[ ]:


for j in list(df3.head().JokeText): print (j)


# In[ ]:


# The least funny?
for j in list(df3.tail().JokeText): print (j)


# * More sorting by median top jokes or bayesian goes here - I leave it as an exercise to the reader :)

# In[ ]:





# ### Export

# In[ ]:


df.to_csv("jokerRatingsMerged.csv.gz",index=False,compression="gzip")

