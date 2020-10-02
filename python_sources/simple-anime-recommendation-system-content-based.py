#!/usr/bin/env python
# coding: utf-8

# **INTRODUCTION**
# 
# Hi, in this notebook, I'll create a very simple anime recommendation system based on ratings and genre.
# let's get started by importing necessary library and the dataset

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/rating.csv')


# In[ ]:


anime = pd.read_csv('../input/anime.csv')


# **PART 1. DATA PREPROCESSING**
# 
# First, I'll merge the user rating dataframe with anime dataframe in order to get anime name

# In[ ]:


df = pd.merge(df,anime.drop('rating',axis=1),on='anime_id')


# In[ ]:


df.head()


# now, let's check anime by its rating

# In[ ]:


df.groupby('name')['rating'].mean().sort_values(ascending=False).head(10)


# Hmmm... seems something's not right here. maybe that animes with 10 rating only got a few users watched them so the rating goes up so high.
# 
# so, we need another attribute in order to get better recommendation. **number of users** seems logical since the more users watched the anime, higher probability the anime gets the actual rating based on many users.
# 
# let's check it out

# In[ ]:


df.groupby('name')['rating'].count().sort_values(ascending=False).head(10)


# Now we see some popular anime here, like **Shingeki No Kyojin, Naruto, and even Fullmetal Alchemist** (*I've watched all of them though, except Code Geass and Elfen Lied, maybe I should add them to my watchlist*)

# **PART 2. EXPLORATORY DATA ANALYSIS**
# 
# Let's do a very simple EDA.

# In[ ]:


ratings = pd.DataFrame(df.groupby('name')['rating'].mean())
ratings['num of ratings'] = pd.DataFrame(df.groupby('name')['rating'].count())

genre_dict = pd.DataFrame(data=anime[['name','genre']])
genre_dict.set_index('name',inplace=True)


# In[ ]:


ratings.head()


# Now let's check anime number of ratings distribution

# In[ ]:


plt.figure(figsize=(15,5))
ratings['num of ratings'].hist(bins=300)
plt.xlim(0,3000)


# In[ ]:


ratings['rating'].hist(bins=50)


# In[ ]:


sns.jointplot(x='rating',y='num of ratings',data=ratings)


# From above scatterplot, we can see the higher number of users give rating, higher chance of the anime gets high rating too.

# **PART 3. FUNCTION CREATION**
# 
# Now I'll create the function to be executed when a user accessing an anime page on myanimelist, so that user can get the recommendation based on that anime. This recommendation will be generated based on ratings and genre

# In[ ]:


def check_genre(genre_list,string):
    if any(x in string for x in genre_list):
        return True
    else:
        return False
    
def get_recommendation(name):
    #generating list of anime with the same genre with target
    anime_genre = genre_dict.loc[name].values[0].split(', ')
    cols = anime[anime['genre'].apply(
        lambda x: check_genre(anime_genre,str(x)))]['name'].tolist()
    
    #create matrix based on generated list
    animemat = df[df['name'].isin(cols)].pivot_table(
        index='user_id',columns='name',values='rating')
       
    #create correlation table
    anime_user_rating = animemat[name]
    similiar_anime = animemat.corrwith(anime_user_rating)
    corr_anime = pd.DataFrame(similiar_anime,columns=['correlation'])
    corr_anime = corr_anime.join(ratings['num of ratings'])
    corr_anime.dropna(inplace=True)
    corr_anime = corr_anime[corr_anime['num of ratings']>5000].sort_values(
        'correlation',ascending=False)
    
    return corr_anime.head(10)


# **PART 4. TESTING**

# In[ ]:


get_recommendation('Shingeki no Kyojin')


# In[ ]:


get_recommendation('Kimi no Na wa.')


# In[ ]:


get_recommendation('Naruto')


# In[ ]:


get_recommendation('Mushishi')


# In[ ]:


get_recommendation('Noragami')


# In[ ]:




