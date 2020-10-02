#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook walks through the whole process of machine learning with its different steps and provide web services and mobile applications to recommended movies based on 3 different categories : Trendings, Item based and collaborative filetering. 
# 
# There are basically three types of recommender systems:-
# 
# **Demographic Filtering** - They offer generalized recommendations to every user, based on movie popularity and/or genre. The System recommends the same movies to users with similar demographic features. Since each user is different , this approach is considered to be too simple. The basic idea behind this system is that movies that are more popular and critically acclaimed will have a higher probability of being liked by the average audience.
# 
# **Content Based Filtering** - They suggest similar items based on a particular item. This system uses item metadata, such as genre, director, description, actors, etc. for movies, to make these recommendations. The general idea behind these recommender systems is that if a person liked a particular item, he or she will also like an item that is similar to it.
# 
# **Collaborative Filtering** - This system matches persons with similar interests and provides recommendations based on this matching. Collaborative filters do not require item metadata like its content-based counterparts.
# 
# 
# The main goal is to apply these differenet steps to the [Movies Lens 20M Dataset][1] that contains more than 27K movies and 7K users. We went more information about this dataset since we have the imdb id we gathered these from [IMDB][2] usign specific script (see below).
# 
# ![Movie IT](https://i.ibb.co/HBqd63F/web.png)
# 
# **Data Set**
# 
# [IMDB][2] is the most famous source of movies with a large number of data with about 6 million movies. We dispose a data that we had from [Movies Lens 20M Dataset][1] that contains more than 20 millions ratings on 27,000 movies and 7,000 users. 
# 
# This data contains a very few informations about the data, that is why we want to gather more information about the movies from IMDB as we have the IMDB movie Id.
# 
# After that we applied the 3 different processes on our data.
# 
# **Steps Applied**
# 
# 1. Gather the dataset
# 
# 2. Understand the data using Google Facets
#    
# 3. Prepocess the data
# 
# 4. Apply the [EDA][3](Exploratory Data Analysis) - Visualization
# 
# 5. Demographic Filtering
# 
# 5. Content Based Filtering
# 
# 6. Collaborative Filtering
# 
# 7. Export scripts
# 
# 8. Create Web service to apply the prediction
# 
# 9. Create Web application ( http://thefourtwofour.com )
# 
# [1]: https://grouplens.org/datasets/movielens/20m/
# [2]: https://www.imdb.com/
# [3]: https://en.wikipedia.org/wiki/Exploratory_data_analysis

# # 1.Gather the dataset
# 
# 
# We got the data as we have stated befor from movies lens. But this data didn't contained too much fields.
# 
# ![Dataset fields](https://i.ibb.co/kBNfSNm/data.png)
# 
# Fortunately, in an other file, this data comes with imdb ids for the movies listed.
# 
# ![Dataset imdb ids](https://i.ibb.co/3ryjQdR/data2.png)
# 
# Based on that, we created a script that gather most of the fields that we can find on IMDB, and here a snippet of that crawler.
# 
# ![IMDB crawler](https://i.ibb.co/vdYXJCv/Preprocessing.png)
# 
# The fields after that has been gathered in a file that we used for both the demographic and content based filetering.
# 
# ![Final data](https://i.ibb.co/ypR0FJT/final-data.png)
# 
# The data now, is more interesting with different fields that we could used specially for similarity as we will see next.
# 
# The [movies dataset][1] contains the following features:
# 
# * movieId - The movie id in our dataset.
# * imdbId - The movie id in IMDB.
# * imdbLink - The link to the movie in IMDB.
# * title - The movie title.
# * genres - The different movies genres such as Drama & Comedy.
# * imdbRate - The movie rate according to IMDB.
# * imdbRateCount - The count of people that rated the movie.
# * duration - The movie duration.
# * date - The movie that the year has been created.
# * poster - The movie poster picture.
# * trailer - The link to the trailer in IMDB.
# * type - The movie type for which group of people the movie is suited.
# * director - The movie director.
# * writers - The movie writers.
# * actors - The movie actors.
# 
# The [ratings dataset][2] has the following features:
# 
# * userId - The rater used it.
# * movieId - The movie id that has been rated.
# * rating - The rating out of 5.
# * timestamp - The time of the rate.
# 
# [1]: https://www.kaggle.com/younessennadj/moviesrecommendation#movies.csv
# [2]: https://www.kaggle.com/younessennadj/moviesrecommendation#ratings-1m.csv

# # 2.Understand the data using Google Facets
# 
# 
# For this part, to understand a little bit about the data, such as general information and how many data are missing we used [Google Facets][1] a very powerful tool to Understed the data which is critical to build a powerful machine learning system.
# 
# 
# **Numerical Features**
# 
# ![Numerical Feature](https://i.ibb.co/LgBMxMM/1.png)
# 
# We Observe that there is more than 27K movies in the dataset. Dates are missing for some movies, but it is so low to be so concerned about (0.26%). 
# 
# **Categorical Features**
# 
# ![Categorical Feature](https://i.ibb.co/TMPPKy8/2.png)
# 
# We Observe that only 0.06% of the directors are missing, which is again not a lot to be concerned about. Generally, we can say that the data is ready to be used.
# 
# **Grouping plotting**
# 
# In This section we tried to plot the data according to different features. The legend represents the ratigns.
# 
# * Duration by rating
# 
# ![Duration by rating](https://i.ibb.co/S3mym2c/3.png)
# 
# We Observe that most of the movies have a higher rating than 6.2. Most of the movies are in fact of around 2 hours duration or less which make sense. The duration doesn't really affect the ratings as we can observe.
# 
# * Genres by rating
# 
# ![Genres by rating](https://i.ibb.co/N9ZSvVk/5.png)
# 
# When plotting the movies as genres by rating, we observe first that they are not really grouped by one category at the time, since we still didn't apply the preprocessing to the data and we are just trying to understand better what is going on. We observer that most of the comedy movies are highly rated which make sense as it made us more happier :p.
# 
# We Observe as well that the highest cateogry is the drama movies.
# 
# 
# [1]: https://pair-code.github.io/facets/

# # 3.Preprocess the data
# 
# **Load Libraries**
# 
# Now, we get to the coding part when we explore that data using python and anaconda libraries. First of all we load the libraires.

# In[ ]:


# Define the libraries and imports
# Panda
import pandas as pd
#mat plot
import matplotlib.pyplot as plt
#Sea born
import seaborn as sns
#Num py
import numpy as np
#Word Count
import wordcloud as wc

#Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

#CountVectorizer IMPORT
from sklearn.feature_extraction.text import CountVectorizer

import os
import warnings
warnings.filterwarnings('ignore')


# We define a set of functions that would help us loading the data and preprocess some of the features that we are interested into.

# In[ ]:


# Load data from the path to the dataSet and force imdbRate to be numeric
def load_dataset_with_rate(dataSet_path):
    data = pd.read_csv(dataSet_path)
    data['imdbRate'] = pd.to_numeric(data['imdbRate'],errors='coerce')
    return data

# Load data from the path to the dataSet
def load_dataset(dataSet_path):
    data = pd.read_csv(dataSet_path)
    return data

# transform columns that have pipe as a separator
def tranform_columns(df):

    df["actors"]= df["actors"].str.split("|", n = 10, expand = False) 

    df["writers"]= df["writers"].str.split("|", n = 10, expand = False) 

    df["genres"]= df["genres"].str.split("|", n = 10, expand = False)     


# **load_dataset** is a function that would read from the csv file from a path and retrun a dataframe.
# 
# **tranfrom_columns** is a function that would tranfrom columns with pipeline to arrays. we applied for 3 columns "actors, wrtiers, genres".
# 
# **Read movies file**
# 
# Now, let's read the movies file into the dataset.

# In[ ]:


# Load dataset
df= load_dataset_with_rate("../input/movies.csv")

df.head(10)


# Now, let's transform the columns.

# In[ ]:


tranform_columns(df)

df.head(10)


# **Read ratings file**
# 
# Now, let's read the ratings file into the dataset.

# In[ ]:


# Load dataset
dr= load_dataset("../input/ratings-1m.csv")

dr.head(10)


# # 4.Visualization
# 
# In this section, we will apply the EDA to visualize the data with several plottings.
# 
# **Ratings plot**

# In[ ]:


df['imdbRate'].plot(kind='box', subplots=True)


# We observe that most of the ratings are between 6 & 8. The higher rating is around 9.9 and the lowest is around 2.0.
# 
# Let's check the ratings historgram.

# In[ ]:


df['imdbRate'].hist()


# Again, we can see that most of the movies are rated between 6 and 8.
# 
# 

# In[ ]:


p = dr.groupby('rating')['rating'].agg(['count'])

# get movie count
movie_count = df['movieId'].count()

# get customer count
cust_count = dr['userId'].nunique() 

# get rating count
rating_count = dr['userId'].count()

ax = p.plot(kind = 'barh', legend = False, figsize = (15,10))
plt.title('Total pool: {:,} Movies, {:,} users, {:,} ratings given'.format(movie_count, cust_count, rating_count), fontsize=20)
plt.axis('off')

for i in range(1,11):
    ax.text(p.iloc[i-1][0]/4, i-1, 'Rating {}: {:.0f}%'.format(i, p.iloc[i-1][0]*100 / p.sum()[0]), color = 'white', weight = 'bold')


# Again, we see that most of the movies has a rating higher than 6. 
# 
# **Movie ratings aggregated by user**
# 
# Let's plot the ratings by users.

# In[ ]:


userRatingsAggr = dr.groupby(['userId']).agg({'rating': [np.size, np.mean]})
userRatingsAggr.reset_index(inplace=True)  # To reset multilevel (pivot-like) index
userRatingsAggr['rating'].plot(kind='box', subplots=True)


# We observe that most of the movies recieved around 200 rates with an average of 3.7. The users implication varies but this is because we have only part of the uniform data, 1 million ratings only.
# 
# **Movie ratings aggregated by movies**
# 
# Let's plot the ratings by movies.

# In[ ]:


movieRatingsAggr = dr.groupby(['movieId']).agg({'rating': [np.size, np.mean]})
movieRatingsAggr.reset_index(inplace=True)
movieRatingsAggr['rating'].plot(kind='box', subplots=True)


# In this plot, we noticed that most of the movies have between 10 to 100 rates with the same average rate of 3.7.
# 
# Again, we are working on an average data.
# 
# **Movies by years**
# 
# Let's plot the movies by the year of the publication.

# In[ ]:


#Plot the count of the movies by years
def count_pie(series):
    counts=series.value_counts()
    counts=counts/counts.sum()
    labels=['' if num<0.01 else str(year) for (year,num) in counts.items()]
    f, ax = plt.subplots(figsize=(8, 8))
    explode = [0.02 if counts.iloc[i] < 100 else 0.001 for i in range(counts.size)]
    plt.pie(counts,labels=labels,autopct=lambda x:'{:1.0f}%'.format(x) if x > 1 else '',explode=explode)
    plt.show()

count_pie(df.date.dropna().apply(lambda x:str(int(x)//10*10)+'s'))    


# We observe that most of the movies are in the 2000s and the 2010s which make sense since we start having more data and the industry get more interesting.
# 
# Let's do another plot of the ratings distributed by year. First, let's define the function and then apply it.

# In[ ]:


def plotjoint(data,x,y,xlim=None,ylim=None,xscale=None,yscale=None):
    sns.set(style=None,font_scale=2)
    grid=sns.jointplot(data[x],data[y],kind="hex",color="#4CB391",height=15,ratio=10,xlim=xlim,ylim=ylim)

plotjoint(df,"date","imdbRate",xlim=(1900,2020),ylim=(4,8.5))


# Logically, most of the movies are in the 2000s, and we can see that they have more rates, and they go intensevely between 5.5 and 7.5.

# **Genres Plot**
# 
# Now let's plot some graphs about the genres.
# 
# First, let's define a function that count the genres by unique values.

# In[ ]:


# numbers of movies of different genres
def count_multiple(df,column):    
    multiple= df[column]
    multiple= multiple.apply(pd.Series.value_counts)
    multiple = multiple.sum(axis = 0, skipna = True)
    multiple = multiple.sort_values(ascending=False).nlargest(20)
    return multiple


# Let's apply this function to the categories.

# In[ ]:


genres_count = count_multiple(df,'genres')
genres_count


# As seen before, Drama followed by Comedy are the most present genres in the movies. Let's do some plotting.

# In[ ]:


f, ax = plt.subplots(figsize=(10, 6))
plt.xticks(rotation=85, fontsize=15)
genres_count.plot.bar()
plt.show()


# We can see more clearly in this bar plot that Drama and comedy are the most present comparing to other catagories.
# 
# An other presentation that we wanted to do is the words cloud. First, let's define the function and apply it to the genres.

# In[ ]:


#Wordcloud of genres and keywords
def multi_wordcloud(series):
    w=wc.WordCloud(background_color="white",margin=20,width=800,height=600,prefer_horizontal=0.7,max_words=20,scale=2)
    w.generate_from_frequencies(series)
    f, ax = plt.subplots(figsize=(16, 8))
    plt.axis('off')
    plt.imshow(w)
    plt.show()
#Apply the word cloud on the genres
multi_wordcloud(genres_count)


# Even more clearly in this word clouds, Drama and Comedy are dominants.

# **Type Plot**
# 
# Now let's plot some graphs about the types.

# In[ ]:


mean_rate_per_type = df.groupby('type')['imdbRate'].count().sort_values(ascending=False).head(15)
f, ax = plt.subplots(figsize=(10, 6))
plt.xticks(rotation=85, fontsize=15)
p = sns.barplot(x = mean_rate_per_type.index, y = mean_rate_per_type.values)
p = plt.xticks(rotation=90)
plt.show()


# We notices clearly that many of the movies doesn't go to a specific category. Most of the movies types are 'R' which means Passed only for persons 18 and over. Next 'PG-13' i.e. movies for persons of 13 or more.

# **Director Plot**
# 
# Now let's plot some graphs about the directors.

# In[ ]:


count_rate_per_director = df.groupby('director')['imdbRate'].count().sort_values(ascending=False).head(15)
multi_wordcloud(count_rate_per_director)


# We observe that John Ford and Hitchcock are the most directors who direct movies, but let's explore more and see how is that working with the mean.

# In[ ]:


count_rate_per_director = df.groupby('director')['imdbRate'].mean().sort_values(ascending=False).head(15)
f, ax = plt.subplots(figsize=(10, 6))
plt.xticks(rotation=85, fontsize=15)
p = sns.barplot(x = mean_rate_per_type.index, y = mean_rate_per_type.values)
p = plt.xticks(rotation=90)
plt.show()


# We see that the results go really different, but that still not a powerful point to judge, since a director may have only 1 movie and he performs very good. In fact, in the first figure we know more directors than the seconde one.

# **Duration Plot**
# 
# Now let's plot some graphs about the duration.
# 
# We will plot the duration with the mean. First, let's make categories of duration.

# In[ ]:


# Define categorical duration 
def categorize_duration(df,column):
    bins = (5,60,120,300,1000)
    group_names = ['short','Medium','Long','VeryLong']
    categories = pd.cut(df[column],bins,labels=group_names)
    df2 = df.copy()
    df2[column]=categories
    return df2

#Apply the function
df_duration_categories = categorize_duration(df,'duration')

#plot the graph
mean_rate_per_duration = df_duration_categories.groupby('duration')['imdbRate'].mean().sort_values(ascending=False)
f, ax = plt.subplots(figsize=(10, 6))
plt.xticks(rotation=85, fontsize=15)
p = sns.barplot(x = mean_rate_per_duration.index, y = mean_rate_per_duration.values)
p = plt.xticks(rotation=90)
plt.show()


# It looks like a positive relationship between the duration and the mean ratings. Eventhough, there is more long movies so it shouldn't be a final conclusion,one of the reasons is TV-shows with long duration.

# # 5.Demographic Filtering
# 
# They offer generalized recommendations to every user, based on movie popularity and/or genre. The System recommends the same movies to users with similar demographic features. Since each user is different , this approach is considered to be too simple. The basic idea behind this system is that movies that are more popular and critically acclaimed will have a higher probability of being liked by the average audience.
# 
# Before getting started with this :
# 
# * we need a metric to score or rate movie
# * Calculate the score for every movie
# * Sort the scores and recommend the best rated movie to the users.
# 
# We can use the average ratings of the movie as the score but using this won't be fair enough since a movie with 8.9 average rating and only 3 votes cannot be considered better than the movie with 7.8 as as average rating but 40 votes. So, we'll be using IMDB's weighted rating (wr) which is given as :
# 
# ![IMDB weight rating](https://i.ibb.co/3vKYB6M/wr.png)
# 
# where,
# 
# * v is the number of votes for the movie
# * m is the minimum votes required to be listed in the chart
# * R is the average rating of the movie
# * C is the mean vote across the whole report
# 
# Let's calculate the mean rating.

# In[ ]:


C= df['imdbRate'].mean()
C


# So, the mean rating for all the movies is approx 6.5 on a scale of 10.The next step is to determine an appropriate value for m, the minimum ratings required to be listed in the chart. We will use 90th percentile as our cutoff. In other words, for a movie to feature in the charts, it must have more ratings than at least 90% of the movies in the list.

# In[ ]:


m= df['imdbRateCount'].quantile(0.9)
m


# Now, we can filter out the movies that qualify for the chart.

# In[ ]:


q_movies = df.copy().loc[df['imdbRateCount'] >= m]
q_movies.shape


# We see that there are 2726 movies which qualify to be in this list. Now, we need to calculate our metric for each qualified movie. To do this, we will define a function, weighted_rating() and define a new feature score, of which we'll calculate the value by applying this function to our DataFrame of qualified movies:

# In[ ]:


def weighted_rating(x, m=m, C=C):
    v = x['imdbRate']
    R = x['imdbRateCount']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[ ]:


# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)


# Finally, let's sort the DataFrame based on the score feature and output the title, imdbRate, imdbRateCount and weighted rating or score of the top 10 movies.

# In[ ]:


#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
q_movies[['title', 'imdbRate', 'imdbRateCount', 'score']].head(10)


# Hurray! We have made our demographic recommender under the Trending Now tab. 

# # 6.Content based Filtering
# 
# In this recommender system the content of the movie (overview, cast, crew, keyword, tagline etc) is used to find its similarity with other movies. Then the movies that are most likely to be similar are recommended.
# 
# ![Content filtering](https://i.ibb.co/nBPFkLQ/conten.png)
# 
# The content based filtering could be done on many level of features to get the similarity at the end. For instance, some of the recommended systems used the movie description. In our case, we found out that the best features to run the similarity on are the movie metadata such as :
# 
# * Genres
# * Actors
# * Director
# * Writers
# 
# This system give a very good recommended items that are similair to the input movie.
# 
# To achieve this goal, starting from a movie, we will calculate the similarity with all other movies, and then recommand the top similair ones.
# 
# In the industry, there is many similarity algorithms, such as [Eucledian similiarity][1] and [Cosine similarity][2].
# 
# Now, let's figure out what are theses similarities.
# 
# Look at this picture, and considere this small example of 3 documents.
# 
# ![Similarity Example](https://i.ibb.co/wKkVByh/64218389-1047291562325488-3341938023230078976-n.png)
# 
# Considering this 3 doucments : "Sachin", "Dhoni" and "Dohni small". notice that Dohni small is part of the Dhoni document (i.e. every word in the small doucment exists in the big document).
# 
# Now, let's count the words occurences. We are considering 3 main words "Dhoni", "Cricket" and "Sachin". 
# 
# Next step, is to calculate the coupled 2 by 2 similariy i.e. total commun words. We can observe that the total commun words between "Sachin" and "Dhoni" document is higer than between "Dhoni" and "Dhoni Small", eventhough Dhoni Small is a part of Dhoni document, which mean is should be more similair, so a linear similarity can't give us a real estimator of the similarity. Let's look at other similarities.
# 
# * Eucledian Similarity
# 
# is a very basic similarity that counts the distance based on the formula below.
# 
# ![Eucledian Similarity](http://www.analytictech.com/mb876/handouts/image001.gif)
# 
# We observe that in the results, it still give a high similarity between "Sachin" and "Dhoni" documents, which for us still not a very good estimator.
# 
# * Cosine Similarity
# 
# Let's first talk about the formula below.
# 
# ![Cosine Similarity](https://i.ibb.co/RczrdXm/download.png)
# 
# The formula is not really expressive, let's plot the words and documents on a 3 dimension plan (see picture below).
# 
# ![Cosine Similarity](https://i.ibb.co/FhPRX4h/62530004-414250615826401-7888860659297812480-n.png)
# 
# In this plot, the dimensions are the 3 words, and we plot the number of words in each document. than we calculate the cos of the angle between the documents. It gives us an even better normalized measure than the Eucledian similarity. And we can see that clearly in the results in the first picture. We notice that the "Dhoni" and "Dhoni small" are more similair than before.
# 
# Enough talking, show me the code :D.
# 
# [1]: http://www.analytictech.com/mb876/handouts/distance_and_correlation.htm
# [2]: https://en.wikipedia.org/wiki/Cosine_similarity

# First, let's convert the instances into lowercase and strip all the spaces between them in the features we are interested into. 
# This is done so that our vectorizer doesn't count the Johnny of"Johnny Depp"and"Johnny Galecki" as the same.

# In[ ]:


#The next step would be to convert the 
#instances into lowercase and strip all the spaces between them. 
#This is done so that our vectorizer doesn't count the Johnny of"Johnny Depp"and"Johnny Galecki" as the same.

# Function to convert all strings to lower case and strip names of spaces

# we have list like Genres and actors  and we have srings like director
def clean_data(x):
    if isinstance(x, list):  
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# clean remove spaces
features = ['director', 'genres', 'actors','writers']

for feature in features:
    df[feature] = df[feature].apply(clean_data)
df.head(10)    


# Next step, We are now in a position to create our "metadata soup", which is a string that contains all the metadata that we want to feed to our vectorizer (namely actors, director ,geners and writers). so we have now column called Soup.

# In[ ]:


def create_soup(x):
    return ' '.join(x['genres']) + ' ' + ' '.join(x['actors']) + ' ' + x['director'] + ' ' + ' '.join(x['writers'])
df['soup'] = df.apply(create_soup, axis=1)
df['soup'].head()


# Next, we will vectorize the words, in 2 steps :
# * Step 1 - We fit the words, we just count the words count
# * Step 2 - Just like in MapReduce in Big Data, we will run a wordcount and then create an inverted index that for each document, show the count of each word contained in that document. Notice that the document in this application is the movie.

# In[ ]:


# Import CountVectorizer and create the count matrix
# so we want to make cosine similarity but there are two questions how we can make it and why we use it specially 
# at the first we have here 236056 movies so for each movie we want to know how many times the word from
# soup column appeared in each movie how we can make that ?
# by using CountVectorizer steps:
# 1-create instance of CountVectorizer
# 2-Fit is to (learn vocabularies) learning here means we want to know the word and repeated count for it in whole files
# like : mandiargues': 3705

# 3 - transform - > encode all movies documents as vectors so you will have each movie with all words so you can know 
# how many times this word appeared in each movie 




#Stopwords are words which do not contain enough significance to
#be used without our algorithm. We would not want these words taking up space i

count = CountVectorizer(stop_words='english')
vocab = count.fit(df['soup'])
dict(list(vocab.vocabulary_.items())[0:10])


# We see here the results for step1, the words count.

# In[ ]:


count_matrix = vocab.transform(df['soup'])
print(count_matrix)


# We see here the results from step2, the count for each word occurence in each document.

# Next, we created a function that given a movieId it will return the top similair movies. We apply first the cosine similarity on only the movie that we are concerned about againt all the other movies, this is in fact a huge optmization instead of do the whole matrix at once. Next, we sort the similarities and we show the top 10 similair movies.

# In[ ]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations_names(df,id_movie):
    cosine_sim = cosine_similarity(count_matrix[df.loc[df['movieId']==id_movie].index[0],] , count_matrix)


    
    # Get the index of the movie that matches the title
    idx = 0

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices].tolist()


# We applied that for the movieId 1 which is 'Toy Story'.

# In[ ]:


get_recommendations_names(df,1)


# Let's observe the results. We can see tha in the top similair movies 'Toy story 2' and 'Toy story 3' appeared, which make sens as they are the next parts of the movie.

# # 7.Collaborative Filtering
# 
# Our content based engine suffers from some severe limitations. It is only capable of suggesting movies which are close to a certain movie. That is, it is not capable of capturing tastes and providing recommendations across genres.
# 
# Also, the engine that we built is not really personal in that it doesn't capture the personal tastes and biases of a user. Anyone querying our engine for recommendations based on a movie will receive the same recommendations for that movie, regardless of who she/he is.
# 
# Therefore, in this section, we will use a technique called Collaborative Filtering to make recommendations to Movie Watchers.
# 
# 
# ![User Based Filetering](https://i.ibb.co/wgxXR57/15.png)
# 
# 
# * User based filtering : These systems recommend products to a user that similar users have liked. For measuring the similarity between two users we will be using the [ML Surpirse][1] a Python scikit building and analyzing recommender systems.
# 
# Surprise provides various ready-to-use prediction algorithms such as baseline algorithms, neighborhood methods, matrix factorization-based ( SVD, PMF, SVD++, NMF), and many others, that we could use directly.
# 
# [1]: http://surpriselib.com/
# 

# **Load Libraries**
# 
# First of all we load the libraires that we need out from Surpriseh.

# In[ ]:


#Surprise import
from surprise import Dataset,Reader,SVD,SVDpp,BaselineOnly,NMF,NormalPredictor,CoClustering,KNNBaseline,KNNWithMeans,KNNBasic,SlopeOne
from surprise.model_selection import cross_validate,train_test_split
from surprise.model_selection import GridSearchCV as GridSearchCV_surprise 
from surprise import accuracy

#Timer
from datetime import datetime


# Let's define some functions that would help us next.

# In[ ]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\nTime taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
        return (datetime.now() - start_time).total_seconds()

def train_and_time(algo,trainset,testset):
    start_time = timer(None)

    # Train the algorithm on the trainset, and predict ratings for the testset
    algo.fit(trainset)

    score_time.append(timer(start_time))

    predictions = algo.test(testset)

    # Then compute RMSE
    score_rmse.append(accuracy.rmse(predictions))
    # Then compute MAE
    score_mae.append(accuracy.mae(predictions))


# **timer** is a function that will calculate the time of execution for a function.
# 
# **train_and_time** is a function that would train the model from surprise, show the scores and time the execution.
# 
# Let's define some variables for scoring the errors, the algorithms and the execution time.

# In[ ]:


score_rmse = []
score_mae=[]
score_time=[]
score_algos = []
predictions =''


# Now, let's read the data to the Surprise dataset using a CSV reader and split the traing and test datasets on 20% based.

# In[ ]:


# path to dataset file
file_path = os.path.expanduser('../input/ratings-1m.csv')

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-1M dataset, each line has the following format:
# 'user item rating timestamp', separated by ',' characters.
reader = Reader(line_format='user item rating timestamp', sep=',',skip_lines=1)

data = Dataset.load_from_file(file_path, reader=reader)

# sample random trainset and testset
# test set is made of 20% of the ratings.
trainset, testset = train_test_split(data, test_size=.20)


# Now, let's apply the different model and see what we get as a result. Out of the algorithms that we will test :
# * SVM
# * NMF
# * Base Line
# * Random
# * Co-clustering
# * KNN Base line
# * KNN means
# * KNN Basic
# * Slope One
# * SVD++
# 
# Note that we only show the execution of SVD that takes more than 1 minute. All others have been tested on a dedicated server and results are shown below.

# In[ ]:


# We'll use the famous SVD algorithm.
score_algos.append('SVD')
svd = SVD()
train_and_time(svd,trainset,testset)


# ![Models_1](https://i.ibb.co/q9nc1Bc/1.png)
# 
# ![Models_2](https://i.ibb.co/Hx6fj5j/2.png)
# 
# ![Models_3](https://i.ibb.co/VJvtC0F/3.png)
# 
# For all these models that we have executed we showed the execution time and the score errors.
# 
# **Model Comparaison**
# 
# Now, let's compare these models by plotting them by :
# 
# * RMSE
# 
# ![RMSE_Code](https://i.ibb.co/DDwz5G7/4.png)
# ![RMSE_Results](https://i.ibb.co/MG6YCLH/5.png)
# 
# We can see clearly that the best model according to the RMSE is the SVD++ but not really a big margin than the SVD. Actually, the way we read this plotting, the less error the better is.
# 
# * MAE
# 
# ![MAE](https://i.ibb.co/g7PdR3N/6.png)
# 
# This plotting representing the other metric MAE, just confirm the first one, SVD++ followed by SVD are the best with a very small margin.
# 
# * Time Execution
# 
# ![Time](https://i.ibb.co/P5T0RT9/7.png)
# 
# Now, this is very important. SVD++ take very long time comparing to others especially SVD. since there is no big difference in the error margin between SVD and SVD++, we might take SVD as the best for time execution and performance mesaure.

# **Model Selection**
# 
# Base on the previous plotting, on the performance, execution time and the score errors, we have decide to use the SVD as it doesn't take a lot of time and get a pretty good score (small error) comparing to the best SVD++ that have a high downside of long time execution ( more than 1 hours comparing to 1 minute for the SVD).
# 
# **Prediction**
# 
# Last step is the prediction. Based on a user, we want to get the top movies to be recommanded based on our model.
# 
# For that, we have created a set of functions.

# In[ ]:


def get_related_movies(data,uid):
    #Get a list of all movies ids
    iids= data['movieId'].unique()
    #Get a list of iids that uid has rated
    iids_of_uid = data.loc[data['userId']==uid,'movieId']
    #Remove the iids that uid has rated from the list of all movie ids
    iids_to_pred = np.setdiff1d(iids,iids_of_uid)
    
    return iids_to_pred

def get_top_movies_prediction_for_user(algo,uid,iids_to_pred):
    predictions=[]
    for iid in iids_to_pred:
        predictions.append(algo.predict(str(uid),str(iid),4.0))
    return predictions

def get_n_top_movies(predictions,iids_to_pred,rec_num):
    pred_ratings = np.array([pred.est for pred in predictions])
    rec_items = []
    for i in range(rec_num):
        i_max=pred_ratings.argmax()
        rec_items.append(iids_to_pred[i_max])
        pred_ratings=np.delete(pred_ratings,i_max)
    return rec_items


# **get_related_movies** retrun the ids of the movies that this user haven't watch yet.
# 
# **get_top_movies_prediction_for_user** return the prediction for the movies that he haven't watch yet. Observer that these are not sorted, neither select the top recommended.
# 
# **get_n_top_movies** retrun the ids of the top recommended movies after the sort and top selection.
# 
# Then, we just apply this for one of our users.

# In[ ]:


uid = 57
iids_to_pred = get_related_movies(dr,uid)
predictions = get_top_movies_prediction_for_user(svd,uid,iids_to_pred)
print(get_n_top_movies(predictions,iids_to_pred,10))


# Those are the ids of the movies that must be recommended for this user.

# # 8.Export Scripts
# 
# 
# To prepare the field for the web api to use our models, it is useful to create easy to use scripts. 
# 
# In this section, we will show what we did with the demographic filetering, it should be the same for other parts. Basically, we just call the script and it will write the data to a file containing the movies ids that are trendings.
# 
# Here is the script :
# 
# ![Trandings script](https://i.ibb.co/GWJXZw9/21.png)
# 
# And here is the file that contains the results :
# 
# 
# 
# 
# ![Trandings movies ids](https://i.ibb.co/v3vTv3N/20.png)
# 

# # 9.Create Web servie
# 
# 
# Next step is to use these models to build a web service API that will return the requests into responses. An example of that the Trendings demographic movies ids, that you can see below :
# 
# ![Post Predict](https://i.ibb.co/SBMb7Zz/api.png)

# # 10.Web application
# 
# 
# Finnaly, using different models, the exported scripts and the Web API we have build, we created a web application with a nice User Interface similair to movies using a basic template from [W3Layouts][1]. 
# 
# * User management 
# 
# First of all, we created the user account creation and login using a database.
# 
# ![Create Account](https://i.ibb.co/sj40H2G/2.png)
# 
# 
# ![Login](https://i.ibb.co/KrNMvyg/1.png)
# 
# * Movies interface
# 
# Next, we created, a user interface that allowed the user to :
# 
# **logout**
# 
# ![UI](https://i.ibb.co/BBV2g98/3.png)
# 
# **See the trendings now**
# 
# This are basically the demographic filetering based on the popularity.
# 
# ![UI](https://i.ibb.co/3ccqCWk/4.png)
# 
# **See the recommended movies**
# 
# This are the recommended movies based on the collaborative filetering.
# 
# ![UI](https://i.ibb.co/6X1fYXR/6.png)
# 
# **See the similair movies**
# 
# This are the similair to the movies that you have watch based on the cosine similarity that we have used.
# 
# ![UI](https://i.ibb.co/hZRx0Lq/30.png)
# 
# **See the movies you have already watch**
# 
# This are the movies you have watched before and you might like to watch again.
# 
# ![UI](https://i.ibb.co/LZSMPR9/31.png)
# 
# 
# * Single Movie interface
# 
# Finnaly, we created a single movie user interface that allowed the user to :
# 
# **See movie details**
# 
# The different details + Movie trailer
# 
# ![UI](https://i.ibb.co/9N3XPMZ/32.png)
# 
# **See similair movies**
# 
# This movies that are similair to that movie.
# 
# ![UI](https://i.ibb.co/nCFS8Fd/33.png)
# 
# 
# [1]: https://w3layouts.com/

# # Refrences
# 
# This notebook has been produced using some external kernles and ressources listed below: 
# 
# [Kernel-1][1] 
# 
# [Kernel-2][2] 
# 
# [Kernel-3][3] 
# 
# [1]: https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system
# [2]: https://www.kaggle.com/bakostamas/movie-recommendation-algorithm
# [3]: https://www.kaggle.com/sjj118/movie-visualization-recommendation-prediction
