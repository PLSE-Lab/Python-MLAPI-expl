#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">Movie Recommendation Model</font></center></h1>
# 
# 
# <img src="https://storage.googleapis.com/kaggle-datasets-images/3405/5520/155700dd4800b6486f19dcab0e0b0cb8/dataset-card.jpg" width="400"></img>
# 
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>Introduction</a>  
# - <a href='#2'>Analysis preparation</a>  
# - <a href='#3'>Data preparation and exploration</a>  
# - <a href='#4'>Static rating model - top 10 movies</a>
# - <a href='#5'>Recommendation simple models</a>  
# - <a href='#6'>Conclusions</a>
# - <a href='#7'>References</a>

# # <a id="1">Introduction</a>  
# 
# ## Data
# 
# We are using only reduced data from the movie dataset, as following:
# *	**ratings_small**
# *   **links_small**
# *	**movies_metadata**  
# *   **credits**  
# *   **keywords**
# 
# We will create few recommendation models.
# 
# <a href="#0"><font size="1">Go to top</font></a>

# # <a id="2">Analysis preparation</a>  
# 
# We start by loading the packages needed for the analysis.
# 
# ## Load packages

# In[54]:


import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import gc
from datetime import datetime 
from sklearn.model_selection import train_test_split
import os
pd.set_option('display.max_columns', 100)


# ## Read the data
# 
# Let's read the data. In this exercise, we store the data in the current directory.

# In[3]:


PATH="../input"
print(os.listdir(PATH))


# In[55]:


ratings_df = pd.read_csv(os.path.join(PATH,"ratings_small.csv"), low_memory=False)


# In[57]:


links_df = pd.read_csv(os.path.join(PATH,"links_small.csv"), low_memory=False)


# In[58]:


movies_metadata_df = pd.read_csv(os.path.join(PATH,"movies_metadata.csv"), low_memory=False)


# In[59]:


credits_df = pd.read_csv(os.path.join(PATH,"credits.csv"), low_memory=False)


# In[60]:


keywords_df = pd.read_csv(os.path.join(PATH,"keywords.csv"), low_memory=False)


# ## Glimpse the data
# 
# Let's glimpse the data. We check the number of rows and columns, sample 5 rows and also run preliminary statistics (with *describe*) on the data.

# In[61]:


print("Ratings data contains {} rows and {} columns".format(ratings_df.shape[0], ratings_df.shape[1]))
print("Links data contains {} rows and {} columns".format(links_df.shape[0], links_df.shape[1]))
print("Movie metadata contains {} rows and {} columns".format(movies_metadata_df.shape[0], movies_metadata_df.shape[1]))
print("Credits data contains {} rows and {} columns".format(credits_df.shape[0], credits_df.shape[1]))
print("Keywords data contains {} rows and {} columns".format(keywords_df.shape[0], keywords_df.shape[1]))


# In[62]:


ratings_df.head()


# In[63]:


links_df.head()


# We will need to extract `genres` as lists of strings with movie genres.

# In[64]:


movies_metadata_df.head()


# In[66]:


keywords_df.head()


# In[7]:


from ast import literal_eval
# Returns the list top l elements or entire list; whichever is more.
def get_list(x, l=5):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than l elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > l:
            names = names[:l]
        return names

    #Return empty list in case of missing/malformed data
    return []

movies_metadata_df['genres'] = movies_metadata_df['genres'].apply(literal_eval)
movies_metadata_df['genres'] = movies_metadata_df['genres'].apply(get_list)


# Let's check also the data types.

# In[9]:


pd.DataFrame({'feature':ratings_df.dtypes.index, 'dtype':ratings_df.dtypes.values})


# In[8]:


movies_metadata_df.head()


# In[10]:


pd.DataFrame({'feature':movies_metadata_df.dtypes.index, 'dtype':movies_metadata_df.dtypes.values})


# All features, `genre`, `id` and `title` are strings (or list of strings). 

# Mean value for rating is ~3.5, min is 0.5 and max is 5. The ratings are given between Jan 1995 and Aug 2017.

# In[11]:


ratings_df.describe()


# In[12]:


import datetime
min_time = datetime.datetime.fromtimestamp(min(ratings_df.timestamp)).isoformat()
max_time = datetime.datetime.fromtimestamp(max(ratings_df.timestamp)).isoformat()
print('Timestamp for ratings from {} to {}:'.format(min_time, max_time))


# ## Check missing

# We can confirm that we do not have missing data in `ratings_df`.
# Let's also check `movies_metadata_df`.

# In[13]:


def check_missing(data_df):
    total = data_df.isnull().sum().sort_values(ascending = False)
    percent = (data_df.isnull().sum()/data_df.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()

check_missing(ratings_df)


# In[14]:


check_missing(movies_metadata_df)


# There are 6 movies without a title or 0.01%.
# Let's drop these rows.

# In[72]:


movies_metadata_df.dropna(subset=['title'], inplace=True)
check_missing(movies_metadata_df)


# In[73]:


movies_metadata_df['id'] = pd.to_numeric(movies_metadata_df['id'])


# ## Filter only votes to movies in movies metadata

# In[74]:


ratings_df.shape


# In[75]:


ratings_df = ratings_df.merge(movies_metadata_df[['id']], left_on=['movieId'], right_on=['id'], how='inner')


# In[76]:


ratings_df.shape


# 
# <a href="#0"><font size="1">Go to top</font></a>

# # <a id="3">Data preparation</a>  
# 
# Let's start to verify in more detail and curate the data.
# 
# 
# ## Extract datetime

# In[21]:


ratings_df['time_dt'] = ratings_df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))


# In[22]:


ratings_df.head()


# Let's extract also few date/time attributes.

# In[23]:


ratings_df['year'] = ratings_df['time_dt'].dt.year
ratings_df['month'] = ratings_df['time_dt'].dt.month
ratings_df['day'] = ratings_df['time_dt'].dt.day
ratings_df['dayofweek'] = ratings_df['time_dt'].dt.dayofweek


# In[24]:


ratings_df[['year', 'month', 'day', 'dayofweek']].describe()


# All date and time looks fine, we will not need to eliminate or correct any value.

# ## Check date/time distribution
# 
# Let's proceed now to check date/time distribution. 

# In[25]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(18,3))
s = sns.boxplot(ax = ax1, y="year", data=ratings_df, palette="Greens",showfliers=True)
s = sns.boxplot(ax = ax2, y="month", data=ratings_df, palette="Blues",showfliers=True)
s = sns.boxplot(ax = ax3, y="day", data=ratings_df, palette="Reds",showfliers=True)
s = sns.boxplot(ax = ax4, y="dayofweek", data=ratings_df, palette="Reds",showfliers=True)
plt.show()


# Let's also show the number and average value of ratings variation in time.

# In[26]:


dt = ratings_df.groupby(['year'])['rating'].count().reset_index()
fig, (ax) = plt.subplots(ncols=1, figsize=(12,6))
plt.plot(dt['year'],dt['rating']); plt.xlabel('Year'); plt.ylabel('Number of votes'); plt.title('Number of votes per year')
plt.show()


# In[27]:


dt = ratings_df.groupby(['year'])['rating'].mean().reset_index()
fig, (ax) = plt.subplots(ncols=1, figsize=(12,6))
plt.plot(dt['year'],dt['rating']); plt.xlabel('Year'); plt.ylabel('Average ratings'); plt.title('Average ratings per year')
plt.show()


# In[28]:


fig, (ax) = plt.subplots(ncols=1, figsize=(12,4))
s = sns.boxplot(x='year', y="rating", data=ratings_df, palette="Greens",showfliers=True)
plt.show()


# In[29]:


fig, (ax) = plt.subplots(ncols=1, figsize=(10,4))
s = sns.boxplot(x='month', y="rating", data=ratings_df, palette="Blues",showfliers=True)
plt.show()


# In[30]:


fig, (ax) = plt.subplots(ncols=1, figsize=(6,4))
s = sns.boxplot(x='dayofweek', y="rating", data=ratings_df, palette="Reds",showfliers=True)
plt.show()


# Observations:
# * We are not observing a special behavior other than variation in time of the ratings averages, with a descending trend from 1995 to 2004, also with 2 peaks in 1997 and 1999 and ascending trend since 2004 to 2012, to start decreasing again.
# * Number of votes shows a lot of peaks and valleys, and also an ascendent trend to 2005, followed by a descending one from 2005, a sharp increase to 2015.
# * There are a number of outliers for each year, month and day of week. We observe a strange alignment of the rating distribution per year intervals 1996-2002, 2003-2011 and 2012-2017 which could prompt us to conclude that the selection used some artificial sampling.

# ## Users distribution
# 
# 
# Let's check if we have special users (users that give many votes, users that give preponderently high ratings, users that give mostly low ratings, users that are giving the perfect average value). Also, users with only one vote.

# In[31]:


print("There is a total of {} users, with an average number of {} votes.".format(ratings_df.userId.nunique(),                                                 round(ratings_df.shape[0]/ratings_df.userId.nunique()),2))


# In[ ]:


print("Top 5 voting users:\n")
tmp = ratings_df.userId.value_counts()[:5]
pd.DataFrame({'Votes':tmp.values, 'Id':tmp.index})


# In[ ]:


tmp = ratings_df.userId.value_counts()
df = pd.DataFrame({'Votes':tmp.values, 'Id':tmp.index})
print("There are {} users that voted only once.".format(df[df['Votes']==1].nunique().values[0]))


# In[ ]:


tmp = ratings_df.groupby(['userId'])['rating'].mean().reset_index()
tmp['rating'] = tmp['rating'].apply(lambda x: round(x,3))
df_max = tmp[tmp['rating']==5]
df_min = tmp[tmp['rating']==0.5]
print("Users giving only '5': {}\nUsers giving only '0.5':{}".format(df_max.shape[0], df_min.shape[0]))


# In[ ]:


mean_rating = round(ratings_df['rating'].mean(),3)
print("Average value of rating is {}.".format(mean_rating))
print("There are {} users that have their average score with the overall average score (approx. with 3 decimals).".format(                            tmp[tmp['rating']==mean_rating]['userId'].nunique()))


# ## Movie distribution
# 
# Let's see what are the movies with the largest number of votes, with the biggest rating, with the lowest rating, how many movies have ratings close to average rating.

# In[32]:


print("There is a total of {} movies, with an average number of {} votes.".format(ratings_df.movieId.nunique(),                                                 round(ratings_df.shape[0]/ratings_df.movieId.nunique()),2))


# In[ ]:


print("Top 10 voted movies:\n")
tmp = ratings_df.movieId.value_counts()[:10]
pd.DataFrame({'Votes':tmp.values, 'id':tmp.index})


# Let's see what movies are those:

# In[ ]:


top_10 = pd.DataFrame({'Votes':tmp.values, 'id':tmp.index}).merge(movies_metadata_df)
top_10


# Observation: now all the movies in the `ratings_df` are present also in `movies_metadata_df` dataset.

# In[ ]:


tmp = ratings_df.movieId.value_counts()
df = pd.DataFrame({'Votes':tmp.values, 'Id':tmp.index})
print("There are {} movies that were voted only once.".format(df[df['Votes']==1].nunique().values[0]))


# In[ ]:


tmp = ratings_df.groupby(['movieId'])['rating'].mean().reset_index()
tmp['rating'] = tmp['rating'].apply(lambda x: round(x,3))
df_max = tmp[tmp['rating']==5]
df_min = tmp[tmp['rating']==0.5]
print("Movies with only '5': {}\nMovies with only '0.5':{}".format(df_max.shape[0], df_min.shape[0]))


# **Note**: these values should be interpreted considering that we only use a 20% sample from the ratings_df total data.

# In[ ]:


mean_rating = round(ratings_df['rating'].mean(),3)
print("Average value of rating is {}.".format(mean_rating))
print("There are {} movies that have their average score with the overall average score (approx. with 3 decimals).".format(                            tmp[tmp['rating']==mean_rating]['movieId'].nunique()))


# Let's check now genres distribution.

# In[33]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=17,
        max_font_size=40, 
        scale=5,
        random_state=1
    ).generate(str(data))

    fig = plt.figure(1, figsize=(10,10))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(movies_metadata_df['genres'], title = 'Movie Genres Prevalence in The Movie Dataset')


# <a href="#0"><font size="1">Go to top</font></a>
# 
# 
# # <a id="4">Static rating model - top 10 movies</a>  
# 
# Let's build now a baseline static rating model to create the top 10 movies.  
# 
# We use the formulas borrowed from two Kernels:
# * https://www.kaggle.com/fabiendaniel/film-recommendation-engine    
# * https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system/data  
# 
# 
# We will use for this the **IMDB** weighted rating formula, as following:
# 
# $$IMDB = {\frac{v}{v+m}}{R} + {\frac{m}{v+m}}{C} $$
# 
# where:
# * **IMDB** is weighted rating;  
# * **v** is the number of votes for the movie;
# * **m** is the minimum votes required to be included in the calculation;
# * **R** is the average rating of the movie; 
# * **C** is the mean vote across the whole set.
# 
# Let's calculate these values. For `m`, we consider quantile 0.9.

# In[34]:


tmp = ratings_df.groupby(['movieId'])['rating'].mean()
R = pd.DataFrame({'id':tmp.index, 'R': tmp.values})
tmp = ratings_df.groupby(['movieId'])['rating'].count()
v = pd.DataFrame({'id':tmp.index, 'v': tmp.values})
C = ratings_df['rating'].mean()


# In[35]:


m_df = movies_metadata_df.merge(R, on=['id'])
m_df = m_df.merge(v, on=['id'])
m_df['C'] = C
m= m_df['v'].quantile(0.9)
m_df['m'] = m


# In[36]:


m_df.head()


# In[38]:


m_df['IMDB'] = (m_df['v'] / (m_df['v'] + m_df['m'])) * m_df['R'] + (m_df['m'] / (m_df['v'] + m_df['m'])) * m_df['C']


# We can show now the top 10 movies according to the IMDB score.

# In[39]:


m_df.sort_values(by=['IMDB'], ascending=False).head(10)


# The result: 

# ## <font color="blue">Static IMDB model: top 10</font>  
# 
# This is our reference model.

# In[40]:


m_df[['title', 'IMDB']].sort_values(by=['IMDB'], ascending=False).head(10)


# Let's compare this reference model with top 10 by only number of votes and by total value of votes.
# 
# ## <font color="grey">Top 10 by number of votes</font>

# In[41]:


m_df['R_x_v'] = m_df['R'] * m_df['v']


# In[42]:


m_df[['title', 'v']].sort_values(by=['v'], ascending=False).head(10)


# ## <font color="grey">Top 10 by product of average rating and number of votes</font>

# In[ ]:


m_df[['title', 'R_x_v']].sort_values(by=['R_x_v'], ascending=False).head(10)


# ## Memory cleanup

# In[43]:


del tmp, top_10
gc.collect()


# ## <a id="5">Recommendation simple models</a>
# 
# 
# ## Simple model using similarities of movie title
# 
# The model will use similarities, calculated based on movie title.
# 
# We will use the already calculated `m_df` dataset.   
# The methods to calculate cosine simmilarities are taken from:   
# * https://www.kaggle.com/fabiendaniel/film-recommendation-engine    
# * https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system/data  

# In[44]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english',max_features=10000)
tokens = m_df[['title']]
tokens['title'] = tokens['title'].fillna('')
tfidf_matrix = tfidf.fit_transform(tokens['title'])
print(tfidf_matrix.shape)
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_sim.shape)
indices = pd.Series(tokens.index, index=tokens['title']).drop_duplicates()
def get_recommendations(title, cosine_sim=cosine_sim):
    # index of the movie that matches the title
    idx = indices[title]

    # similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # movie indices
    movie_indices = [i[0] for i in sim_scores]

    # top 10 most similar movies
    return tokens['title'].iloc[movie_indices]


# In[45]:


get_recommendations('The Million Dollar Hotel')


# In[46]:


get_recommendations('Sleepless in Seattle')


# ## Combined model, using similarity and popularity
# 
# This model uses both similarity factor, based on movie title and popularity score, based on IMDB score.
# 
# We modify the `get_recommendation` function to return a number of 50 similar titles; we order the titles using popularity score. 
# 
# Inspiration for the following functions are from:  
# 
# * https://www.kaggle.com/fabiendaniel/film-recommendation-engine    
# * https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system/data  

# In[47]:


tfidf = TfidfVectorizer(stop_words='english',max_features=10000)
tokens = m_df[['title']]
tokens['title'] = tokens['title'].fillna('')
tfidf_matrix = tfidf.fit_transform(tokens['title'])
print(tfidf_matrix.shape)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_sim.shape)
indices = pd.Series(tokens.index, index=tokens['title']).drop_duplicates()


# In[48]:


def get_imdb_score(df, indices):
    # select the data from similarity indices
    tmp = df[df.id.isin(indices)]
    # sort the data by IMDB score
    tmp = tmp.sort_values(by='IMDB', ascending=False)
    # return title and IMDB score
    return tmp[['title','IMDB']].head(10)


# In[50]:


def get_10_recommendations_simpol(title, cosine_sim=cosine_sim):
    # index of the movie that matches the title
    idx = indices[title]

    # similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # scores of the 20 most similar movies
    sim_scores = sim_scores[1:21]
    
    # movie indices
    movie_indices = [i[0] for i in sim_scores]

    # get popularity scores
    pop_scores = get_imdb_score(m_df, movie_indices)
    
    return list(pop_scores['title'])


# In[51]:


get_10_recommendations_simpol('The Million Dollar Hotel')


# In[52]:


get_10_recommendations_simpol('Judgment Night')


# In[53]:


get_10_recommendations_simpol('Fahrenheit 9/11')


# <a href="#0"><font size="1">Go to top</font></a>
# 
# 
# # <a id="6">Conclusion</a>  
# 
# Three simple recommendation models were created, as following:
# 
# * One model, user agnostic, based on ratings averages per movie and number of votes, it is basically the IMDB rating model; this model provide a top-10 list of movies recommendation, starting from a movie selected by a user;  
# * One model, depending on simmilarities of titles;
# * One model, depending on similarities of titles and IMDB rating.
# 

# # <a id="8">References</a>  
# 
# I took inspiration from several sources. 
# 
# [1] https://medium.com/recombee-blog/machine-learning-for-recommender-systems-part-1-algorithms-evaluation-and-cold-start-6f696683d0ed  
# [2] https://www.kaggle.com/fabiendaniel/film-recommendation-engine  
# [3] https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system/data  
# 
# 
# <a href="#0"><font size="1">Go to top</font></a>
