#!/usr/bin/env python
# coding: utf-8
This notebook contains two main parts: (1) an exploratory data analysis (EDA) that visualizes the content data on Netflix, and (2) a content-based recommender system.
With EDA, I try to answer questions that allow us to describe and understand the content better, such as what are the countries/directors/genres with largest number of movies/TV shows on Netflix, what are the top rated movies (using IMDb ratings), and what are the longest TV shows, etc.. 
The recommender system, based on what we enter, picks content with similar attributes on Netflix. 
# # Getting Started

# As shown below, the Netflix dataset contains useful information on type, director, cast, country, release year, duration, genre, and description. 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot

df = pd.read_csv("../input/netflix-shows/netflix_titles.csv")

df.head()


# First of all, I would like to know (1) what the proportion of movies versus TV shows is on Netflix as of Jan 2020, and (2) what the trend of adding content looks like on Netflix over the years.  

# In[ ]:


types = df['type'].value_counts().reset_index()
types = types.rename(columns = {'type' : "count", "index" : 'type'})
fig = go.Figure(data=[go.Pie(labels=types['type'], values=types['count'], hole=.3,textinfo='label+percent')],
                layout=go.Layout(title="Content Type"))
fig.update_traces(marker=dict(colors=['gold', 'Indigo']))
fig.show()


# In[ ]:


df["date_added"] = pd.to_datetime(df['date_added'])
df['year'] = df['date_added'].dt.year

movie = df[df["type"] == "Movie"]
tv = df[df["type"] == "TV Show"]

movie = movie['year'].value_counts().reset_index()
movie = movie.rename(columns = {'year' : "count", "index" : 'year'})
movie = movie.sort_values('year')

tv = tv['year'].value_counts().reset_index()
tv = tv.rename(columns = {'year' : "count", "index" : 'year'})
tv = tv.sort_values('year')


fig = go.Figure(data=[go.Scatter(x=movie['year'], y=movie["count"], name="Movies",marker=dict(color='gold')),
                      go.Scatter(x=tv['year'], y=tv["count"], name="TV Shows",marker=dict(color='Indigo'))], 
                layout=go.Layout(title="Content Added Over Years"))
fig.show()


# As shown by the results, Netflix has about 2X more movies than TV shows. Since 2016, we see an increasing number of content being added to Netflix, especially movies. 

# # Movie - Exploration

# Next I divided the exploration into two sections: Movie and TV shows. Starting with the movies on Netflix, it is fun to see what the most popular words are in the movie titles. "Love", "man", and "christmas" are the common words. 

# In[ ]:


movie = df[df["type"] == "Movie"]

from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
plt.rcParams['figure.figsize'] = (10, 10)
wordcloud = WordCloud(stopwords=STOPWORDS, background_color = 'black', width = 500,  height = 500, 
                      max_words = 100).generate(' '.join(movie['title'].str.lower()))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Most Popular Words in Movie Title',fontsize = 30)
plt.show()


# Next I will dive deeper into movies by breaking down the movie count by country, released year, guideline group, genre, and finally duration. 

# In[ ]:


movie_c = movie['country'].value_counts().reset_index()
movie_c = movie_c.rename(columns = {'country' : "count", "index" : 'country'})
movie_c = movie_c.head(10)

fig = go.Figure(go.Treemap(
    labels = movie_c['country'],
    values = movie_c["count"], parents=["","", "", "", "", "", "", "", "", "",],
    textinfo = "label+value"), layout=go.Layout(title="Top 10 Countries with Most Movies"))

fig.show()


# What directors have the most movies on Netflix?

# In[ ]:


from collections import Counter
director_split = ", ".join(movie['director'].fillna("missing")).split(", ")
director_split = Counter(director_split).most_common(11)
del director_split[0]
fig = go.Figure(data=[go.Bar(y=[_[0] for _ in director_split][::-1], 
                             x=[_[1] for _ in director_split][::-1], orientation='h',marker=dict(color='DarkTurquoise'))], 
                layout=go.Layout(title="Director with Most Movies"))
fig.show()


# In[ ]:


movie_y = movie['release_year'].value_counts().reset_index()
movie_y = movie_y.rename(columns = {'release_year' : "count", "index" : 'release_year'})
movie_y = movie_y.sort_values('release_year')

fig = go.Figure(data=[go.Bar(x=movie_y['release_year'], y=movie_y["count"],marker=dict(color='Coral'))], 
                layout=go.Layout(title="Movie by Release Year"))
fig.show()


# We can see that most movies were released in recent years, especially during the period from 2016 to 2018. What are the oldest movies on Netflix? Looking at below, they are all documentaries about war from USA. 

# In[ ]:


old_movie = movie.sort_values("release_year", ascending = True)
old_movie = old_movie[old_movie['duration'] != ""]
old_movie = old_movie[["title","country","release_year","rating","listed_in"]][:10]

fig = go.Figure(data=[go.Table(
    header=dict(values=list(old_movie.columns),
                fill_color='gold',
                align='left'),
    cells=dict(values=[old_movie.title, old_movie.country, old_movie.release_year, old_movie.rating, old_movie.listed_in],
               fill_color='white',
               align='left'))],layout=go.Layout(title="10 Oldest Movies"))
fig.show()


# In[ ]:


movie_r = movie['rating'].value_counts().reset_index()
movie_r = movie_r.rename(columns = {'rating' : "count", "index" : 'rating'})

fig = go.Figure(data=[go.Bar(x=movie_r['rating'], y=movie_r['count'])], 
                layout=go.Layout(title="Movies by Guideline Group"))
fig.update_layout(xaxis={'categoryorder':'total descending'})


# TV-MA which is for adults only has the most movies, followed by TV-14, and R and TV-PG. Thus movies on Neftlix are more adult and teen friendly. 

# In[ ]:


genre_split = ", ".join(movie['listed_in']).split(", ")
genre_split = Counter(genre_split).most_common(20)

fig = go.Figure(data=[go.Pie(labels=[_[0] for _ in genre_split][::-1], values=[_[1] for _ in genre_split][::-1], 
                textinfo='label')], layout=go.Layout(title="Movies by Genre"))
fig.show()


# The top 3 movie genres are International Movies, Dramas, and Comedies; however, these genres are not mutually exclusive. A movie is international as long as it is not from USA.  

# # Movie - IMBd Ratings

# I am interested in knowing the IMDb ratings of the movies on Netflix - what are the top rated movies by genre and country, and what are their directors and cast? Thus, I joined the Netflix dataset with the IMDb Extensive Dataset (note that the data size is smaller as 1/3 of data did not find a match). The sunburst chart shows the top 10 results from the join. 

# In[ ]:


imdb_ratings=pd.read_csv('../input/imdb-extensive-dataset/IMDb ratings.csv')
imdb_titles=pd.read_csv('../input/imdb-extensive-dataset/IMDb movies.csv')
ratings = pd.DataFrame({'Title':imdb_titles.title,
                    'Rating': imdb_ratings.weighted_average_vote})
ratings.drop_duplicates(subset=['Title','Rating'], inplace=True)
ratings.dropna()
join=ratings.merge(movie,left_on='Title',right_on='title',how='inner')
join=join.sort_values(by='Rating', ascending=False)

import plotly.express as px
top_rated=join[0:10]
fig =px.sunburst(
    top_rated,
    path=['title','country'],
    values='Rating',
    color='Rating')
fig.show()


# What are the top movies from some of the genres? I picked Dramas, Action & Adventure, and Thriller out of personal interests.  

# In[ ]:


join_drama = join[join["listed_in"].str.contains("Dramas")]
top_rated=join_drama[0:15]
top_rated = top_rated[["title","country","release_year","director","Rating"]]

fig = go.Figure(data=[go.Table(
    header=dict(values=list(top_rated.columns),
                fill_color='pink',
                align='left'),
    cells=dict(values=[top_rated.title, top_rated.country, top_rated.release_year, top_rated.director, top_rated.Rating],
               fill_color='white',
               align='left'))],layout=go.Layout(title="Top Rated Drama Movies"))
fig.show()


# In[ ]:


join_action = join[join["listed_in"].str.contains("Action")]
top_rated=join_action[0:15]
top_rated = top_rated[["title","country","release_year","director","Rating"]]

fig = go.Figure(data=[go.Table(
    header=dict(values=list(top_rated.columns),
                fill_color='orange',
                align='left'),
    cells=dict(values=[top_rated.title, top_rated.country, top_rated.release_year, top_rated.director, top_rated.Rating],
               fill_color='white',
               align='left'))],layout=go.Layout(title="Top Rated Action & Adventure Movies"))
fig.show()


# In[ ]:


join_thriller = join[join["listed_in"].str.contains("Thriller")]
top_rated=join_thriller[0:15]
top_rated = top_rated[["title","country","release_year","director","Rating"]]

fig = go.Figure(data=[go.Table(
    header=dict(values=list(top_rated.columns),
                fill_color='lightblue',
                align='left'),
    cells=dict(values=[top_rated.title, top_rated.country, top_rated.release_year, top_rated.director, top_rated.Rating],
               fill_color='white',
               align='left'))],layout=go.Layout(title="Top Rated Thriller Movies"))
fig.show()


# What are the top rated movies by country? I will select UK, Canada, and Spain as examples. 

# In[ ]:


def ratecountry(name):
    join_c = join[join["country"].fillna('missing').str.contains(name)]
    top_rated=join_c[0:10]
    trace = go.Bar(y=top_rated["title"], x=top_rated['Rating'], orientation="h", 
                   marker=dict(color="purple"))
    return trace

from plotly.subplots import make_subplots
traces = []
titles = ["United Kingdom","","Canada","","Spain"]
for title in titles:
    if title != "":
        traces.append(ratecountry(title))

fig = make_subplots(rows=1, cols=5, subplot_titles=titles)
fig.add_trace(traces[0], 1,1)
fig.add_trace(traces[1], 1,3)
fig.add_trace(traces[2], 1,5)

fig.update_layout(height=500, showlegend=False, yaxis={'categoryorder':'total ascending'})
fig.show()


# # TV Show - Exploration

# Moving on to TV shows, I will explore the title, country, release year, genre, and seasons. 

# In[ ]:


tv = df[df["type"] == "TV Show"]

plt.rcParams['figure.figsize'] = (10, 10)
wordcloud = WordCloud(stopwords=STOPWORDS, background_color = 'black', width = 500,  height = 500, 
                      max_words = 100).generate(' '.join(tv['title'].str.lower()))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Most Popular Words in TV Show Title',fontsize = 30)
plt.show()


# Similar to movies, "love" is the most popular word in titles. 

# In[ ]:


tv_c = tv['country'].value_counts().reset_index()
tv_c = tv_c.rename(columns = {'country' : "count", "index" : 'country'})
tv_c = tv_c.head(10)

fig = go.Figure(go.Treemap(
    labels = tv_c['country'],
    values = tv_c["count"], parents=["","", "", "", "", "", "", "", "", "",],
    textinfo = "label+value"), layout=go.Layout(title="Top 10 Countries with Most TV Shows"))

fig.show()


# Many TV shows are from United Kingdom, Japan, and South Korea. We can take a closer look into the top 10 genres for each of the three countries. 

# In[ ]:


def genrecountry(c):
    c = tv[tv["country"].fillna('missing').str.contains(c)]
    genre_split = ", ".join(c['listed_in']).split(", ")
    genre_split = Counter(genre_split).most_common(10)
    genre_name = [_[0] for _ in genre_split][::-1]
    genre_count = values=[_[1] for _ in genre_split][::-1]
    trace = go.Bar(y=genre_name, x=genre_count, orientation="h", 
                   marker=dict(color="Indigo"))
    return trace

traces = []
clist = ["United Kingdom","","Japan","","South Korea"]
for c in clist:
    if c != "":
        traces.append(genrecountry(c))

fig = make_subplots(rows=1, cols=5, subplot_titles=clist)
fig.add_trace(traces[0], 1,1)
fig.add_trace(traces[1], 1,3)
fig.add_trace(traces[2], 1,5)

fig.update_layout(height=500, showlegend=False, yaxis={'categoryorder':'total ascending'},title="Top 10 TV Show Genre from Each Country")
fig.show()


# While most TV shows have only 1 season, there are some TV shows with more than 10 seasons. What are they?

# In[ ]:


tv_d = tv['duration'].value_counts().reset_index()
tv_d = tv_d.rename(columns = {'duration' : "count", "index" : 'duration'})

fig = go.Figure(data=[go.Bar(x=tv_d['duration'], y=tv_d['count'],marker=dict(color='Darkgreen'))], 
                layout=go.Layout(title="TV Shows by Seasons"))
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# In[ ]:


manyseason = ['10 Seasons','11 Seasons','12 Seasons','13 Seasons','14 Seasons']
tv_long = tv[tv['duration'].isin(manyseason)]

tv_long = tv_long[["title","country","release_year","director","duration","listed_in"]]

fig = go.Figure(data=[go.Table(
    header=dict(values=list(tv_long.columns),
                fill_color='lightgreen',
                align='left'),
    cells=dict(values=[tv_long.title, tv_long.country, tv_long.release_year, tv_long.director, 
                       tv_long.duration, tv_long.listed_in],
               fill_color='white',
               align='left'))],layout=go.Layout(title="Longest TV Shows"))
fig.show()


# # Content-based Recommendation System with NLP

# Lastly, I want to build a recommendation system that is based on type, director, guideline group, genre and description. 
# https://towardsdatascience.com/how-to-build-from-scratch-a-content-based-movie-recommender-with-natural-language-processing-25ad400eb243

# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df2 = df[['title','type','director','rating','listed_in','description']]
df2.head()
df2['description'] = df2['description'].fillna('')
df2['director'] = df2['director'].fillna('')
df2['rating'] = df2['rating'].fillna('')
df2['listed_in'] = df2['listed_in'].map(lambda x: x.lower().split(','))
df2.set_index('title', inplace = True)

df2['Key_words'] = ''
columns = df2.columns
for index, row in df2.iterrows():
    words = ''
    for col in columns:
        words = words + ''.join(row[col])+ ' '
    row['Key_words'] = words
    
df2.drop(columns = [col for col in df2.columns if col!= 'Key_words'], inplace = True)


# In[ ]:


# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df2['Key_words'])

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(df2.index)
indices[:5]

# returning 10 recommended movies 
def recommendations(title, cosine_sim = cosine_sim):
    recommended_movies = []
    idx = indices[indices == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_10_indexes = list(score_series.iloc[1:11].index)
    for i in top_10_indexes:
        recommended_movies.append(list(df2.index)[i])
    return recommended_movies


# Testing the results - I will enter a few titles and see the top 10 content it recommends to me: 
# (1) "Lord of Rings: The Return of the King')"

# In[ ]:


recommendations('The Lord of the Rings: The Return of the King')


# (2)"I Am Mother"

# In[ ]:


recommendations('I Am Mother')


# The recommendations seem reasonable as most of them are from the same genre as the movie title I entered. 
