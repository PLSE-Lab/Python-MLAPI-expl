#!/usr/bin/env python
# coding: utf-8

# # Anime Recommendation System
# 
# Recommend Anime to User based on the Cosine Similarity Scores of Synopsis and Genres

# ## Libraries

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
import random


# ## Dataset
# 
# We have 5 CSV Files available to us in the dataset these are:
#   1. <i>datagenre-all-share-new:</i> It contains Anime ID along with Genre for that Anime
#   2. <i>datascorehist-all-share-new:</i> Scores for each Anime
#   3. <i>datastaff-all-share-new:</i> ID's of the Staff Member involved in the Anime.
#   4. <i>datasynopsis-all-share-new:</i> Synopsis of the Anime i.e A short summary of the Anime Plot.
#   5. <i>datatitle-all-share-new:</i> Anime ID with Anime Names
#   
# We are gonna use Title, Genre and Synopsis of the Anime.

# In[2]:


# Reading Datasets
anime_title = pd.read_csv('../input/datatitle-all-share-new.csv', sep = "|")
anime_genre = pd.read_csv('../input/datagenre-all-share-new.csv', sep = "|")
anime_syn = pd.read_csv('../input/datasynopsis-all-share-new.csv', sep = "|")


# In[3]:


print (f"Shape of Anime Title: {anime_title.shape}")
print (f"Shape of Anime Synopsis: {anime_syn.shape}")
print (f"Shape of Anime Genre: {anime_genre.shape}")


# In[4]:


anime_title.head()


# In[5]:


anime_genre.head()


# In[6]:


anime_syn.head()


# Merging the 3 DataFrame into single DataFram <b><i>anime</i></b>.

# In[7]:


anime = pd.merge(anime_title, anime_syn, on = ['Anime_ID'])
anime = pd.merge(anime, anime_genre, on = ['Anime_ID'])


# In[8]:


anime.head()


# In[9]:


anime.info()


# Synopsis and Genres both contains some <b>nan</b> values. We need to take care of them while processing.

# In[10]:


# Genres are in the format action;ecchi;adventure etc
# Clean (or Parse) the Genres for each anime
def clean_genres(x):
    if isinstance(x['Genres'], str):
        gen = x['Genres'].replace(";", ", ")
    else:
        gen = ""
        
    return gen


# In[11]:


# Replace NaN with "" i.e with empty string in the synopsis
anime['Synopsis'].fillna("", inplace = True)

anime['Genres'] = anime.apply(clean_genres, axis = 1)


# ## Building Recommendation
# The Recommendation Engine that we are building is based on Synopsis and Genre of each of the animes.
# 
# We are gonna use Tf-Idf for Synopsis and Count Vectorizer for Genres to generate matrix. Then, using these matrix we are gonna find cosine similarity b/w animes using the linear_kernel of scikit-learn

# In[13]:


# Construct Similarity Matrix for Synopsis and Genres

# 1. Get the Indicies
indices = pd.Series(anime.index, index = anime['Anime_name'])


# In[14]:


# 2 Setup the TfidfVect and  CountVec
tfidf = TfidfVectorizer(stop_words = "english")
countvec = CountVectorizer(stop_words = "english")


# In[16]:


# 3. Get Matrix for both
tfidf_mat = tfidf.fit_transform(anime['Synopsis'])
countvect_mat = countvec.fit_transform(anime['Genres'])


# In[17]:


# 4. Cosine Similarity Score
syn_similarity = linear_kernel(tfidf_mat, tfidf_mat)
genre_similarity = linear_kernel(countvect_mat, countvect_mat)


# ```
# For getting recommendation, I have taken average of the two cosine scores and used that to sort the list and then select the top 10 animes to recommend to the user.
# ```

# In[22]:


# 5. Get Recommendation
def getRecommendation(title):
    
    # Get the Index of the Anime.
    idx = indices[title]
    
    # We have 2 Similarity Metrics
    ## 1. Synopsis Similarity
    ## 2. Genre Similarity
    
    score_1 = list(enumerate(syn_similarity[idx]))
    score_2 = list(enumerate(genre_similarity[idx]))
    
    # Sort the scores in reverse order
    score_1 = sorted(score_1, key = lambda x: x[0], reverse = False)
    score_2 = sorted(score_2, key = lambda x: x[0], reverse = False)    
    
    # Average of the two Similarity (Cosine) Scores
    combined_score = [(idx, (sc_1 + sc_2) / 2) for (idx, sc_1), (_, sc_2) in zip(score_1, score_2)]
    
    # Sorting the Combined Score.
    combined_score = sorted(combined_score, key = lambda x: x[1], reverse = True)
    
    # Get ID of Top 10 Similar Animes
    anime_ids = [i[0] for i in combined_score[1:11]]
    
    # Returning the Top Anime Names.
    return anime['Anime_name'].iloc[anime_ids]


# We now have Cosine Similarity b/w anime based on Synopsis and Genre of the Animes.
# 
# Let's try this.

# In[25]:


def showRecommendation(anime_name):
    g = getRecommendation(title = anime_name)

    print (f"Your Anime: {anime_name}\n\nRecommended Anime for you: \n")
    for i, name in g.iteritems():
        print (name)


# In[26]:


showRecommendation(random.choice(anime['Anime_name']))


# In[28]:


showRecommendation(random.choice(anime['Anime_name']))


# ## END

# I cross-checked the Recommended Anime for a given Anime_Name with the Recommendation on MyAnimeList.
# 
# The Recommendation are not 100% accurate but are acceptable. <br>
# Synopsis and Genres are giving us a good recommendation for a given anime.

# In[ ]:




