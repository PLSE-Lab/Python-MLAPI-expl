#!/usr/bin/env python
# coding: utf-8

# # Item Based Collaborative Filtering
# ![banner](https://raw.githubusercontent.com/varian97/Anime-Recommender-System/master/image.png)

# In[ ]:


import numpy as np
import pandas as pd
import warnings
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')


# # Read the data

# In[ ]:


anime = pd.read_csv("../input/anime.csv")
anime.head()


# In[ ]:


# only select tv show and movie
print(anime.shape)
anime = anime[(anime['type'] == 'TV') | (anime['type'] == 'Movie')]
print(anime.shape)


# In[ ]:


# only select famous anime, 75% percentile
m = anime['members'].quantile(0.75)
anime = anime[(anime['members'] >= m)]
anime.shape


# In[ ]:


rating = pd.read_csv("../input/rating.csv")
rating.head()


# In[ ]:


rating.shape


# # Replacing missing rating with NaN

# In[ ]:


rating.loc[rating.rating == -1, 'rating'] = np.NaN
rating.head()


# # Create index for anime name

# In[ ]:


anime_index = pd.Series(anime.index, index=anime.name)
anime_index.head()


# # Join the data

# In[ ]:


joined = anime.merge(rating, how='inner', on='anime_id')
joined.head()


# # Create a pivot table

# In[ ]:


joined = joined[['user_id', 'name', 'rating_y']]

pivot = pd.pivot_table(joined, index='name', columns='user_id', values='rating_y')
pivot.head()


# In[ ]:


pivot.shape


# # Drop all users that never rate an anime

# In[ ]:


pivot.dropna(axis=1, how='all', inplace=True)
pivot.head()


# In[ ]:


pivot.shape


# # Center the mean around 0 (centered cosine / pearson)

# In[ ]:


pivot_norm = pivot.apply(lambda x: x - np.nanmean(x), axis=1)
pivot_norm.head()


# # Item Based Collaborative Filtering

# In[ ]:


# fill NaN with 0
pivot_norm.fillna(0, inplace=True)
pivot_norm.head()


# ## Calculate Similar Items

# In[ ]:


# convert into dataframe to make it easier
item_sim_df = pd.DataFrame(cosine_similarity(pivot_norm, pivot_norm), index=pivot_norm.index, columns=pivot_norm.index)
item_sim_df.head()


# In[ ]:


def get_similar_anime(anime_name):
    if anime_name not in pivot_norm.index:
        return None, None
    else:
        sim_animes = item_sim_df.sort_values(by=anime_name, ascending=False).index[1:]
        sim_score = item_sim_df.sort_values(by=anime_name, ascending=False).loc[:, anime_name].tolist()[1:]
        return sim_animes, sim_score


# In[ ]:


animes, score = get_similar_anime("Steins;Gate")
for x,y in zip(animes[:10], score[:10]):
    print("{} with similarity of {}".format(x, y))


# ## Helper Function

# In[ ]:


# predict the rating of anime x by user y
def predict_rating(user_id, anime_name, max_neighbor=10):
    animes, scores = get_similar_anime(anime_name)
    anime_arr = np.array([x for x in animes])
    sim_arr = np.array([x for x in scores])
    
    # select only the anime that has already rated by user x
    filtering = pivot_norm[user_id].loc[anime_arr] != 0
    
    # calculate the predicted score
    s = np.dot(sim_arr[filtering][:max_neighbor], pivot[user_id].loc[anime_arr[filtering][:max_neighbor]])             / np.sum(sim_arr[filtering][:max_neighbor])
    
    return s


# In[ ]:


predict_rating(3, "Steins;Gate")


# In[ ]:


predict_rating(3, "Cowboy Bebop")


# ## Get Recommendation

# In[ ]:


# recommend top n_anime for user x based on item collaborative filtering algorithm
def get_recommendation(user_id, n_anime=10):
    predicted_rating = np.array([])
    
    for _anime in pivot_norm.index:
        predicted_rating = np.append(predicted_rating, predict_rating(user_id, _anime))
    
    # don't recommend something that user has already rated
    temp = pd.DataFrame({'predicted':predicted_rating, 'name':pivot_norm.index})
    filtering = (pivot_norm[user_id] == 0.0)
    temp = temp.loc[filtering.values].sort_values(by='predicted', ascending=False)

    # recommend n_anime anime
    return anime.loc[anime_index.loc[temp.name[:n_anime]]]


# In[ ]:


get_recommendation(3)


# In[ ]:


get_recommendation(5)


# Compared to the user based collaborative filtering, the recommendation given may be very different. As we know, in the user based, so many user did not rate the anime. In the item based however, I think it is more robust because there is no anime that never rated by users.

# In[ ]:




