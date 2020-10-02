#!/usr/bin/env python
# coding: utf-8

# Nowadays growth of data in world is very huge and rapidly growing by various soical memdia and other platform, To handle this large amount of data Machine learning come into picture. This technology use this data and provide the appropriate and user inetrest based data, So that user get the recommended items based on their interset.  
# 
# Three type of recommendation system are there:
# 1. Demograhic filtering: This is gemerlized recommendation system to every user, Like online movie or game recommendation. This recommendationrecommendation is come based on popularity, more probablity of user or audience likes.
# 2. Content-based filtering: Content based filtering is suggest the based on smililar items.This syetem uses the metadata, such as genre, director, anchor, descritpion etc. This system generally recommend if persion like or search particluar system.
# 3. Collabarative filtring: This system matches the person with simillar interset. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df1=pd.read_csv('../input/credits/tmdb_5000_credits.csv')
df2=pd.read_csv('../input/movies/movies_500_tmdb.csv')
df1.columns = ['id','tittle','cast','crew']
df2= df2.merge(df1,on='id')
df2.head()


# Demograhic Filtering:
# Before getting started with this -
# 
# we need a metric to score or rate movie
# Calculate the score for every movie
# Sort the scores and recommend the best rated movie to the users.
# We can use the average ratings of the movie as the score but using this won't be fair enough since a movie with 8.9 average rating and only 3 votes cannot be considered better than the movie with 7.8 as as average rating but 40 votes. So, I'll be using IMDB's weighted rating (wr) which is given as :-
# 
# where,
# 
# v is the number of votes for the movie;
# m is the minimum votes required to be listed in the chart;
# R is the average rating of the movie; And
# C is the mean vote across the whole report
# We already have v(vote_count) and R (vote_average) and C can be calculated as

# In[ ]:


C = df2['vote_average'].mean()
C


# Mean rating of all movies approx is 6 on scale of 10. Now we determine the value of m to list the movie in the chart, for that we will use 90% of all movie.

# In[ ]:


m = df2['vote_count'].quantile(0.9)
m


# Now based on this approximation we can filter out the movies that qualify the cirieria. 

# In[ ]:


q_movies = df2.copy().loc[df2['vote_count'] >= m]
q_movies.shape


# We can see here 481 movies which qualify the criteria. Now we need to calculate the metric of each qualified movie and to do this we will define the function weight_rating() with a new feature score.

# In[ ]:


def weighted_rating(x, m=m, C=C):
    v= x['vote_count']
    R = x['vote_average']
    #Calculation based on the IMDB formula
    return (v/(v+m)*R)+(m/(m+v)*C)


# In[ ]:


#Define a new feature score and caluclate the value using 'weight_rating' function
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)


# Now, let's sort the dataframe based on new feature score and output the title, vote_count, vote_average, score

# In[ ]:


#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the the top 15 movies
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)


# Hurray! Now we have made the first system. Under the trending tab we will find the movies which are very popular and obtained by sorting the dataset.
# 

# In[ ]:


pop = df2.sort_values('popularity', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.barh(pop['title'].head(6), pop['popularity'].head(6), align='center', color='green')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movie")


# Now something to keep in mind is that these demographic recommender provide a general chart of recommended movies to all the users. They are not sensitive to the interests and tastes of a particular user. This is when we move on to a more refined system- Content Basesd Filtering.

# Content based Filtering:
# Now in this we will find movies based on content which can movie overview, casst, crew, keyword etc.

# Plot description based: In this we use 'overview' feature to get description detail.And then we will find the system based on similarity score.  

# In[ ]:


df2['overview'].head(5)


# For any of you who has done even a bit of text processing before knows we need to convert the word vector of each overview. Now we'll compute Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each overview.
# 
# Now if you are wondering what is term frequency , it is the relative frequency of a word in a document and is given as (term instances/total instances). Inverse Document Frequency is the relative count of documents containing the term is given as log(number of documents/documents with term) The overall importance of each word to the documents in which they appear is equal to TF * IDF
# 
# This will give you a matrix where each column represents a word in the overview vocabulary (all the words that appear in at least one document) and each column represents a movie, as before.This is done to reduce the importance of words that occur frequently in plot overviews and therefore, their significance in computing the final similarity score.
# 
# Fortunately, scikit-learn gives you a built-in TfIdfVectorizer class that produces the TF-IDF matrix in a couple of lines. That's great, isn't it?

# In[ ]:


#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df2['overview'] = df2['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# We see that over 20,000 different words were used to describe the 4800 movies in our dataset.
# 
# With this matrix in hand, we can now compute a similarity score. There are several candidates for this; such as the euclidean, the Pearson and the cosine similarity scores. There is no right answer to which score is the best. Different scores work well in different scenarios and it is often a good idea to experiment with different metrics.
# 
# We will be using the cosine similarity to calculate a numeric quantity that denotes the similarity between two movies. We use the cosine similarity score since it is independent of magnitude and is relatively easy and fast to calculate. Mathematically, it is defined as follows:
# 
# Since we have used the TF-IDF vectorizer, calculating the dot product will directly give us the cosine similarity score. Therefore, we will use sklearn's linear_kernel() instead of cosine_similarities() since it is faster.

# In[ ]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# We are going to define a function that takes in a movie title as an input and outputs a list of the 10 most similar movies. Firstly, for this, we need a reverse mapping of movie titles and DataFrame indices. In other words, we need a mechanism to identify the index of a movie in our metadata DataFrame, given its title.

# In[ ]:


#Construct a reverse map of indices and movie titles
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()


# We are now in a good position to define our recommendation function. These are the following steps we'll follow :-
# 
# Get the index of the movie given its title.
# Get the list of cosine similarity scores for that particular movie with all movies. 
# Convert it into a list of tuples where the first element is its position and the second is the similarity score.
# Sort the aforementioned list of tuples based on the similarity scores; that is, the second element.
# Get the top 10 elements of this list. 
# Ignore the first element as it refers to self (the movie most similar to a particular movie is the movie itself).
# Return the titles corresponding to the indices of the top elements.

# In[ ]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]


# In[ ]:


get_recommendations('Batman Forever')


# While our system has done a decent job of finding movies with similar plot descriptions, the quality of recommendations is not that great. "The Dark Knight Rises" returns all Batman movies while it is more likely that the people who liked that movie are more inclined to enjoy other Christopher Nolan movies. This is something that cannot be captured by the present system.

# Now we will find the recommendation based on credits, genres, keyword.

# It goes without saying that the quality of our recommender would be increased with the usage of better metadata. That is exactly what we are going to do in this section. We are going to build a recommender based on the following metadata: the 3 top actors, the director, related genres and the movie plot keywords.
# 
# From the cast, crew and keywords features, we need to extract the three most important actors, the director and the keywords associated with that movie. Right now, our data is present in the form of "stringified" lists , we need to convert it into a safe and usable structure

# In[ ]:


# Parse the stringified features into their corresponding python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)


# Next, we'll write functions that will help us to extract the required information from each feature.

# In[ ]:


# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[ ]:


# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []


# In[ ]:


# Define new director, cast, genres and keywords features that are in a suitable form.
df2['director'] = df2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)


# In[ ]:


# Print the new features of the first 3 films
df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)


# The next step would be to convert the names and keyword instances into lowercase and strip all the spaces between them. This is done so that our vectorizer doesn't count the Johnny of "Johnny Depp" and "Johnny Galecki" as the same.

# In[ ]:


# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# In[ ]:


# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df2[feature] = df2[feature].apply(clean_data)


# We are now in a position to create our "metadata soup", which is a string that contains all the metadata that we want to feed to our vectorizer (namely actors, director and keywords).

# In[ ]:


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df2['soup'] = df2.apply(create_soup, axis=1)


# The next steps are the same as what we did with our plot description based recommender. One important difference is that we use the CountVectorizer() instead of TF-IDF. This is because we do not want to down-weight the presence of an actor/director if he or she has acted or directed in relatively more movies. It doesn't make much intuitive sense.

# In[ ]:


# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])


# In[ ]:


# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[ ]:


# Reset index of our main DataFrame and construct reverse mapping as before
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])


# In[ ]:


get_recommendations('The Dark Knight Rises', cosine_sim2)


# We see that our recommender has been successful in capturing more information due to more metadata and has given us (arguably) better recommendations. It is more likely that Marvels or DC comics fans will like the movies of the same production house. Therefore, to our features above we can add production_company . We can also increase the weight of the director , by adding the feature multiple times in the soup.
# 
# 
# 
# Collaborative Filtering
# 

# Our content based engine suffers from some severe limitations. It is only capable of suggesting movies which are close to a certain movie. That is, it is not capable of capturing tastes and providing recommendations across genres.
# 
# Also, the engine that we built is not really personal in that it doesn't capture the personal tastes and biases of a user. Anyone querying our engine for recommendations based on a movie will receive the same recommendations for that movie, regardless of who she/he is.

# In[ ]:


from surprise import Reader, Dataset, SVD, evaluate
reader = Reader()
ratings = pd.read_csv('../input/rating/ratings_small.csv')
ratings.head()


# In[ ]:


data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)


# In[ ]:


svd = SVD()
evaluate(svd, data, measures=['RMSE', 'MAE'])


# We create recommenders using demographic , content- based and collaborative filtering. While demographic filtering is very elemantary and cannot be used practically, Hybrid Systems can take advantage of content-based and collaborative filtering as the two approaches are proved to be almost complimentary. This model was very baseline and only provides a fundamental framework to start with.
# 
# I would like to mention some excellent refereces that I learned from
# 
# https://hackernoon.com/introduction-to-recommender-system-part-1-collaborative-filtering-singular-value-decomposition-44c9659c5e75
