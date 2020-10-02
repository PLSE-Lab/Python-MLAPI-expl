#!/usr/bin/env python
# coding: utf-8

# **Good Movies, Movies for your joy!**
# 
# Welcome to our Movie recommender System (Good Movies). In this EDA we will explain in details the filtering features that we offer to recommend to you a movie you will most likely adore!
# 
# ![](https://lhslance.org/wp-content/uploads/2017/12/Top-10-1-900x600.jpg)

# **Content**

# **1- EDA (Plotting, Preprocessing and Feature Selection)**
# 
# **2- Demographic Filtering**
# 
# **3- Content Based Filtering**
# 
# **4- Collaborative Filtering**
# 
# **5- Hybrid Recommender**
# 
# **6- Displaying**
# 
# **7- Conclusion**

# Sources:
# 
# * [Kaggle: Getting Started with a movie recommendation](https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system/notebook?fbclid=IwAR03LpAdwODfCMgPWtgWQBlC5dmNqJxMYcvlvEjs8uG_CTUT_PacJVCR9vQ)
#  
# * [Kaggle: movie recommender system](https://www.kaggle.com/rounakbanik/movie-recommender-systems?fbclid=IwAR2twbiQA0GToJi7YPqd72eAz1LqmvDEr9pYPcfIQvEyou0UzM73QvCYvO4)
#  

# ----------------------------------------------------------------------------------

# **1- EDA (Plotting, Preprocessing and Feature Selection)**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate
import scipy.sparse as sparse

import warnings; warnings.simplefilter('ignore')


# In[ ]:


train = pd. read_csv('../input/movies_metadata.csv')
train.head()


# In[ ]:


print (train[pd.to_numeric(train['popularity'], errors='coerce').isnull()])
print (train[pd.to_numeric(train['budget'], errors='coerce').isnull()])


# In[ ]:


train = train.drop([19729, 19730, 29502, 29503, 35586, 35587])


# In[ ]:


train['popularity'] = train[train['popularity'].notnull()]['popularity'].astype('float')
train['budget'] = train[train['budget'].notnull()]['budget'].astype('float')
train['original_language'] = np.where(train['original_language']=="en", 'english', 'other')
train['vote_average'] = train[train['vote_average'].notnull()]['vote_average'].round()


# In[ ]:


fig = sns.barplot(x="adult", y="id", data=train, estimator=len)


# Interpretation: Adult feature is not important in the dataset

# In[ ]:


sns.set(rc={'figure.figsize':(10,8.27)})
fig = sns.barplot(x="status", y="id", data=train, estimator=len)


# Interpretation: Status, video, and some other features showed to be not important in the dataset.

# In[ ]:


sns.set(rc={'figure.figsize':(9,8.27)})
fig1 = sns.barplot(x="original_language", y="id", data=train, estimator=len)


# Interpretation: 98 other languages represent only 1/4 of the data.

# In[ ]:


data = pd.concat([train['budget'], train['revenue']], axis=1)
data.plot.scatter(x='budget', y='revenue');


# Interpretation: Generally, the higher the budget, the higher the revenue.

# In[ ]:


data = pd.concat([train['vote_count'], train['popularity']], axis=1)
data.plot.scatter(x='vote_count', y='popularity');


# Interpretation: Movies with low votes have low popularity, but having high vote count_doesnt mean having higher popularity.

# In[ ]:


ax = sns.barplot(x="vote_average", y='id', data=train, estimator=len)


# Interpretation: Most movies have a rating between 4.6 and 7.5

# In[ ]:


md = pd. read_csv('../input/movies_metadata.csv')
md.head()


# **2- Demographic Filtering**

# We can use the average ratings of the movie as the score but using this won't be fair enough since a movie with 8.9 average rating and only 3 votes cannot be considered better than the movie with 7.8 as as average rating but 40 votes. So, We'll be using IMDB's weighted rating (wr) which is given as :
# 
# ![](https://image.ibb.co/jYWZp9/wr.png)

# In[ ]:


md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# C: is the mean vote across the data
# 
# m: minimum vote required to be listed (must be among the top 5% voted movies)

# In[ ]:


C = md['vote_average'].mean()
m= md['vote_count'].quantile(0.95)
print("C is %f, and m is %d"%(C,m))


# So, the mean rating for all the movies is approx 5.6 on a scale of 10
# 
# The minimum number of votes required is 434
# 
# Now we will filter out the movies that qualify
# 

# In[ ]:


qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())]
qualified.shape


# There are 2274 movies which qualify to be in this list. Now, we need to calculate our metric for each qualified movie. To do this, we will define a function, weighted_rating() and define a new feature score, of which we'll calculate the value by applying this function to our Data of qualified movies, then display the top 10 movies!

# In[ ]:


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)
qualified = qualified[['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['score'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('score', ascending=False)
qualified.head(10)


# **3- Content Based Filtering**

# To personalise our recommendations more, weare going to build an engine that computes similarity between movies based on certain metrics and suggests movies that are most similar to a particular movie that a user liked. This is known as Content Based Filtering.
# We are using a subset of all the movies available to us due to limiting computing power available to us.

# In[ ]:


links = pd.read_csv('../input/links_small.csv')
links = links[links['tmdbId'].notnull()]['tmdbId'].astype('int')
print (md[pd.to_numeric(md['id'], errors='coerce').isnull()])


# In[ ]:


md = md.drop([19730, 29503, 35587])
md['id'] = md['id'].astype('int')


# We can now define our recommendation function. These are the following steps we'll follow :
# 
# * Get the index of the movie given its title.
# * Get the list of cosine similarity scores for that particular movie with all movies. Convert it into a list of tuples where the first element is its position and the second is the similarity score.
# * Sort the mentioned list of tuples based on the similarity scores; that is, the second element.
# * Get the top 10 elements of this list. Ignore the first element as it refers to self (the movie most similar to a particular movie is the movie itself).
# * Return the titles corresponding to the indices of the top elements.

# In[ ]:


def get_recommendations(title, cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# Define cosine similarity and include more features to improve the score

# In[ ]:


credits = pd.read_csv('../input/credits.csv')
keywords = pd.read_csv('../input/keywords.csv')
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')
smd1 = md[md['id'].isin(links)]

features = ['cast', 'crew', 'keywords']
for feature in features:
    smd1[feature] = smd1[feature].apply(literal_eval)


# In[ ]:


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
smd1['director'] = smd1['crew'].apply(get_director)
smd1.head()


# In[ ]:


def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names
    return []
features = ['cast', 'keywords']
for feature in features:
    smd1[feature] = smd1[feature].apply(get_list)
smd1[['title', 'cast', 'director', 'keywords', 'genres']].head(3)


# In[ ]:


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    smd1[feature] = smd1[feature].apply(clean_data)
smd1['director'] = smd1['director'].apply(lambda x: [x,x, x])
smd1.head(3)


# In[ ]:


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast'])  + ' '.join(x['director']) + ' '.join(x['genres'])
smd1['soup'] = smd1.apply(create_soup, axis=1)
smd1[['title', 'cast', 'director', 'keywords', 'genres', 'soup']].head(3)


# In[ ]:


count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd1['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

smd1 = smd1.reset_index()
titles = smd1['title']
indices = pd.Series(smd1.index, index=smd1['title'])

indices.head()


# In[ ]:


get_recommendations('Toy Story',cosine_sim)


# **4- Collaborative Filtering**

# Collaborative Filtering is based on the idea that users similar to a certain user can be used to predict how much they will like a particular product or service those users have used/experienced but the others have not.
# 
# We will not be implementing Collaborative Filtering from scratch. Instead, we will use the Surprise library that used extremely powerful algorithms like Singular Value Decomposition (SVD) to minimise RMSE (Root Mean Square Error) and give great recommendations.

# In[ ]:


reader = Reader()
ratings = pd.read_csv('../input/ratings_small.csv')
ratings.head()


# In[ ]:


data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)
svd = SVD()
evaluate(svd, data, measures=['RMSE', 'MAE'])


# We get a mean Root Mean Sqaure Error of 0.89 which is more than good enough for our case. Let us now train on our dataset and arrive at predictions.

# In[ ]:


trainset = data.build_full_trainset()
svd.train(trainset)
ratings[ratings['userId'] == 1]


# In[ ]:


svd.predict(1, 302)


# For movie with ID 302, we get the above estimation. One startling feature of this recommender system is that it doesn't care what the movie is (or what it contains). It works purely on the basis of an assigned movie ID and tries to predict ratings based on how the other users have predicted the movie.

# **5- Hybrid Recommender**

# In this section, we will try to build a simple hybrid recommender that brings together techniques we have implemented in the content based and collaborative filter based engines. This is how it will work:
# * Input: User ID and the Title of a Movie
# * Output: Similar movies sorted on the basis of expected ratings by that particular user.

# In[ ]:


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan
    
id_map = pd.read_csv('../input/links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd1[['title', 'id']], on='id').set_index('title')
indices_map = id_map.set_index('id')


# In[ ]:


def hybrid(userId, title):
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    #print(idx)
    movie_id = id_map.loc[title]['movieId']
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    movies = smd1.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)


# In[ ]:


movies1= hybrid(1, 'Avatar')
movies1


# In[ ]:


movies2= hybrid(500, 'Avatar')
movies2


# We see that for our hybrid recommender, we get different recommendations for different users although the movie is the same. Hence, our recommendations are more personalized and tailored towards particular users.

# **6- Displaying**

# In[ ]:


metadata = pd. read_csv('../input/movies_metadata.csv')
metadata = metadata.drop([19729, 19730, 29502, 29503, 35586, 35587])
metadata['id'] = metadata[metadata['id'].notnull()]['id'].astype('int')
metadata.head()


# In[ ]:


image_data = metadata[['imdb_id', 'poster_path']]
image_data.head()


# In[ ]:


links = pd.read_csv("../input/links.csv")
links.head()


# In[ ]:


links = links[['movieId', 'imdbId']]


# In[ ]:


image_data = image_data[~ image_data.imdb_id.isnull()]


# In[ ]:


def app(x):
    try:
        return int(x[2:])
    except ValueError:
        print(x)


# In[ ]:


image_data['imdbId'] = image_data.imdb_id.apply(app)

image_data = image_data[~ image_data.imdbId.isnull()]

image_data.imdbId = image_data.imdbId.astype(int)

image_data = image_data[['imdbId', 'poster_path']]

image_data.head()


# In[ ]:


posters = pd.merge(image_data, links, on='imdbId', how='left')

posters[['id', 'poster_path']] = posters[['movieId', 'poster_path']]

posters = posters[~ posters.movieId.isnull()]

posters.movieId = posters.movieId.astype(int)

posters.head()


# In[ ]:


movies_table = pd.merge(movies1, posters, on='id', how='left')
movies_table


# In[ ]:


from IPython.display import HTML
from IPython.display import display

def display_recommendations(df):

    images = ''
    for ref in df.poster_path:
            if '.' in str(ref):
                link = 'http://image.tmdb.org/t/p/w185/' + str(ref)
                images += "<img style='width: 120px; margin: 0px;                   float: left; border: 1px solid black;' src='%s' />"               % link
    display(HTML(images))


# In[ ]:


display_recommendations(movies_table)


# **7- Conclusion**

# In this notebook, We have built 4 different recommendation engines based on different ideas and algorithms. They are as follows:
# 
# * Simple Recommender: This system used overall TMDB Vote Count and Vote Averages to build Top Movies Charts, in general and for a specific genre. The IMDB Weighted Rating System was used to calculate ratings on which the sorting was finally performed.
# * Content Based Recommender: We built two content based engines; one that took movie overview and taglines as input and the other which took metadata such as cast, crew, genre and keywords to come up with predictions. We also deviced a simple filter to give greater preference to movies with more votes and higher ratings.
# * Collaborative Filtering: We used the powerful Surprise Library to build a collaborative filter based on single value decomposition. The RMSE obtained was less than 1 and the engine gave estimated ratings for a given user and movie.
# * Hybrid Engine: We brought together ideas from content and collaborative filterting to build an engine that gave movie suggestions to a particular user based on the estimated ratings that it had internally calculated for that user.
# 
# Thank You! We hope this project has been helpful to you as much as it was enjoyable for us!
