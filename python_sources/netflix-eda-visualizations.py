#!/usr/bin/env python
# coding: utf-8

# # **Netflix Data Analysis**
# 
# ![](https://hdlovewall.com/wallpaper/2015/11/sad-love-movies-on-netflix-10-background.png)
# 
# Netflix is an streaming service which is growing in India at an incredible fast rate. This is an EDA to understand growth and popularity of Netflix over last few years in India.

# In[ ]:


# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import collections
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Loading the dataset

# In[ ]:


netflix=pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")
netflix.head()


# In[ ]:


netflix["date_added"] = pd.to_datetime(netflix['date_added'])
netflix['year_added'] = netflix['date_added'].dt.year


# In[ ]:


## Shape of dataset
netflix.shape


# In[ ]:


## Columns in the dataframe
netflix.columns


# In[ ]:


## Checking for Null Values
netflix.isnull().sum()


# In[ ]:


## Check if there are any duplicate Titles
netflix.duplicated().sum()


# No duplicate Titles are there in the dataset.

# ### Null Value Treatment

# In[ ]:


## Create duplicate dataset

netflix_copy = netflix.copy()
netflix_copy.head()


# In[ ]:


netflix_copy = netflix_copy.dropna()
netflix_copy.shape


# ### Derive new columns

# In[ ]:


## Derive new columns from date which will provide the day, month and year in which they were added in the service

netflix_copy['date_added'] = pd.to_datetime(netflix['date_added'])
netflix_copy['Day_of_release'] = netflix_copy['date_added'].dt.day
netflix_copy['Month_of_release']= netflix_copy['date_added'].dt.month
netflix_copy['Year_of_release'] = netflix_copy['date_added'].dt.year

netflix_copy['Year_of_release'].astype(int);
netflix_copy['Day_of_release'].astype(int);


# ## Data Visualization

# ### Content Type

# In[ ]:


col = "type"
group_value = netflix[col].value_counts().reset_index()
group_value = group_value.rename(columns = {col : "count", "index" : col})

## plotting graph

labels = group_value.type
sizes = group_value['count']
explode=(0.1,0)

fig1, ax = plt.subplots()
ax.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%', shadow=True)
ax.axis('equal')
plt.show()


# * This shows that 2/3 content on the Netflix is Movies and 1/3 is TV Shows.

# ### Ratings Description
# 
# 1. TV-MA - This program is specifically designed to be viewed by adults and therefore may be unsuitable for children under 17
# 2. TV-14 - This program contains some material that many parents would find unsuitable for children under 14 years of age
# 3. TV-PG - This program contains material that parents may find unsuitable for younger children
# 4. R - Under 17 requires accompanying parent or adult guardian
# 5. PG-13 - Some material may be inappropriate for children under 13
# 6. NR - Not rated
# 7. PG - Some material may not be suitable for children
# 8. TV-Y7 - This program is designed for children age 7 and above
# 9. TV-G - This program is suitable for all ages
# 10. TV-Y - This program is designed to be appropriate for all children
# 11. TV-Y7-FV - This program is designed for children age 7 and above containing some fantasy violence
# 12. G - All ages admitted
# 13. UR - Unrated
# 14. NC-17- No One 17 and Under Admitted

# In[ ]:


plt.figure(figsize=(12,10))
sns.set(style="darkgrid")
ax = sns.countplot(x="rating", data=netflix, palette="Set1", order=netflix['rating'].value_counts().index[0:15])


# The most number of TV Shows and Movies are rated 'TV-MA' followed by 'TV-14' (Those can be viewed by viewere above age of 14) and then 'TV-PG' (Viewed after parental approval).
# 
# R rated programs are 4th most on the platform.

# ### Year of Most Content Release

# In[ ]:


# Make separate dataframe for Movies and shows

netflix_shows=netflix[netflix['type']=='TV Show']
netflix_movies=netflix[netflix['type']=='Movie']


# In[ ]:


plt.figure(figsize=(12,10))
sns.set(style="darkgrid")
ax = sns.countplot(y="release_year", data=netflix_movies, palette="Set2", order=netflix_movies['release_year'].value_counts().index[0:15])
plt.title('Movies released per year')
plt.xlabel('Number of movies released')


# #### 2017 is the year in which most movies (650+) were released on the platform. 

# In[ ]:


plt.figure(figsize=(12,10))
sns.set(style="darkgrid")
ax = sns.countplot(y="release_year", data=netflix_shows, palette="Set2", order=netflix_shows['release_year'].value_counts().index[0:15])
plt.title('TV Shows released per year')
plt.xlabel('Number of TV Shows released')


# #### More than 400 TV shows were released in 2019 which is most number of TV Shows released ever on the platform.

# ### Merging Netflix Dataset with IMDB Dataset
# 
# Since we don't have the "Out of 10 Ratings" in the Netflix dataset, importing IMDB dataset.
# It will provide us with ratings and new columns which can be used for further EDA.

# In[ ]:


imdb_ratings=pd.read_csv('/kaggle/input/imdb-extensive-dataset/IMDb ratings.csv',usecols=['weighted_average_vote'])
imdb_titles=pd.read_csv('/kaggle/input/imdb-extensive-dataset/IMDb movies.csv', usecols=['title','year','genre','language'])

ratings = pd.DataFrame({'Title':imdb_titles.title,
                    'Release Year':imdb_titles.year,
                    'Language': imdb_titles.language,
                    'Rating': imdb_ratings.weighted_average_vote,
                    'Genre':imdb_titles.genre})
ratings.drop_duplicates(subset=['Title','Release Year','Rating'], inplace=True)
ratings.rename(columns ={'Rating':'Out_of_10_rating'},inplace=True)
ratings.head()


# In[ ]:


## Drop NA from ratings

ratings = ratings.dropna()


# In[ ]:


## Using Inner Join to connect Netflix database with the IMDB Ratings

Netflix_Ratings = ratings.merge(netflix, left_on = 'Title', right_on = 'title', how='inner')
Netflix_Ratings.sort_values(by = 'Out_of_10_rating', ascending = False).head()


# In[ ]:


# Make separate dataframe for Movies and shows

netflix_shows=Netflix_Ratings[Netflix_Ratings['type']=='TV Show'].sort_values(by = 'Out_of_10_rating', ascending = False)
netflix_movies=Netflix_Ratings[Netflix_Ratings['type']=='Movie'].sort_values(by = 'Out_of_10_rating', ascending = False)


# ### Top 10 On Movies on Netflix

# In[ ]:


top_10_movies = netflix_movies.sort_values("Out_of_10_rating", ascending = False)
top_10_movies = top_10_movies[ top_10_movies['Release Year'] > 2000]
top_10_movies[['title', "Out_of_10_rating"]][0:10]


# ### Top 10 On TV Shows on Netflix

# In[ ]:


top_10_shows = netflix_shows.sort_values("Out_of_10_rating", ascending = False)
top_10_shows = top_10_shows[ top_10_shows['Release Year'] > 2000]
top_10_shows[['title', "Out_of_10_rating"]][0:10]


# In[ ]:


df = Netflix_Ratings[ (Netflix_Ratings['release_year']>2007) & (Netflix_Ratings['release_year']< 2020) ]

d1 = df[df["type"] == "TV Show"]
d2 = df[df["type"] == "Movie"]

col = "release_year"

vc1 = d1[col].value_counts().reset_index()
vc1 = vc1.rename(columns = {col : "count", "index" : col})
vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))
vc1 = vc1.sort_values(col)

vc2 = d2[col].value_counts().reset_index()
vc2 = vc2.rename(columns = {col : "count", "index" : col})
vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))
vc2 = vc2.sort_values(col)

trace1 = go.Scatter(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))
trace2 = go.Scatter(x=vc2[col], y=vc2["count"], name="Movies", marker=dict(color="#6ad49b"))
data = [trace1, trace2]
layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))
fig = go.Figure(data, layout=layout)
fig.show()


# Its observed that in 2018, Number of movies released on platform are way more than number of TV Shows released. But we can see a changed in this trend in 2019.
# 
# Netflix started to invest and release more TV Shows on platform in different regions as popularity of Shows is increasing.

# ### Finding popular genres

# In[ ]:


Netflix_Ratings['listed_in'] = Netflix_Ratings['listed_in'].str.split(',')
Netflix_Ratings['listed_in'].explode().nunique()


# We got 65 different categories.

# In[ ]:


Netflix_Ratings['listed_in'].explode().unique()


# There are some duplicates as they contain white spaces in them.

# In[ ]:


## Removing white spaces
genres = Netflix_Ratings['listed_in'].explode()
genres = [genre.strip() for genre in genres]
genre_count = collections.Counter(genres)
print(genre_count.most_common(5))


# In[ ]:


len(set(genres))


# There are 41 different categories after removing the duplicates.

# In[ ]:


genre_df = pd.DataFrame(genre_count.most_common(5), columns = ['Genre','Count'])
genre_df


# In[ ]:


plt = sns.barplot(x = 'Genre', y = 'Count', data = genre_df)
plt.set_xticklabels(plt.get_xticklabels(), rotation=45, ha='right')


# From 41 different categories, below are top 5 :
# 1. Dramas
# 2. International Movies
# 3. Comedies
# 4. Independent Movies
# 5. Action and Adventure

# ### Number of Movies and TV Shows each country produced

# In[ ]:


## Checking null values for country
Netflix_Ratings['country'].isna().sum()


# In[ ]:


## Create new dataframe where there are no null values for country.
country = Netflix_Ratings[Netflix_Ratings['country'].notna()]
country['country'].isna().sum()


# In[ ]:


country.head(20)


# In[ ]:


## There are more than 1 country for some rows separated by (,)
## Hence, separating these values

country['country'] =country['country'].str.split(',')

country['country'].explode().unique()


# In[ ]:


country['country'].explode().value_counts()


# United states occurs twice as it contains white character at start. Need to remove the character.

# In[ ]:


## Remove the space

countries = country['country'].explode()
countries = [country.strip() for country in countries]
country_count = collections.Counter(countries)

print(country_count.most_common(5))


# In[ ]:


# Visualize top countries
top_countries = country_count.most_common(5)
top_countries_df = pd.DataFrame(top_countries, columns=['country','count'])
top_countries_df


# In[ ]:


plt = sns.barplot(x="country", y="count",palette="Set1", data=top_countries_df)
plt.set_xticklabels(plt.get_xticklabels(), rotation=45, ha='right')


# After US, India is the second largest country where most of the content is produced.
# 
# Hence it can be said that, investing in Indian Film Industry is beneficial.

# ### When most content is released?

# In[ ]:


Netflix_Ratings['Month'] = pd.DatetimeIndex( Netflix_Ratings['date_added']).month_name()
Netflix_Ratings.head(30)


# In[ ]:


Months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
plot = sns.countplot(x="Month",order=Months, data=Netflix_Ratings)

plot.set_xticklabels(plot.get_xticklabels(), rotation=40 , ha="right")


# From plot, we can clearly say that most content is released from October to January. 
# The reason being Holidays.
# 
# We can speculate that it will be more beneficial to release content in these months.

# ### Netflix In India

# In[ ]:


India = country.explode("country")
India = India[India['country']=='India']


# In[ ]:


def content_over_years(country):
    movie_per_year=[]

    tv_shows_per_year=[]
    for i in range(2008,2020):
        h=netflix.loc[(netflix['type']=='Movie') & (netflix.year_added==i) & (netflix.country==str(country))] 
        g=netflix.loc[(netflix['type']=='TV Show') & (netflix.year_added==i) &(netflix.country==str(country))] 
        movie_per_year.append(len(h))
        tv_shows_per_year.append(len(g))



    trace1 = go.Scatter(x=[i for i in range(2008,2020)],y=movie_per_year,mode='lines+markers',name='Movies')

    trace2=go.Scatter(x=[i for i in range(2008,2020)],y=tv_shows_per_year,mode='lines+markers',name='TV Shows')

    data=[trace1,trace2]

    layout = go.Layout(title="Content added over the years in "+str(country), legend=dict(x=0.1, y=1.1, orientation="h"))

    fig = go.Figure(data, layout=layout)

    fig.show()
countries=['India']

for country in countries:
    content_over_years(str(country))


# Netflix started releasing Indian Movies from 2016. We can see spike in data in 2018. This makes sense as Netflix started investing heavily in Indian Market in same year.

# ### Popular Indian Movies

# In[ ]:


## Top 10 Hindi Movies

India[India['Language']=='Hindi'].sort_values(by=['Out_of_10_rating'], ascending=False).head(10)
top_10_shows_india = India[India['Language']=='Hindi'].sort_values(by=['Out_of_10_rating'], ascending=False)
top_10_shows_india = top_10_shows_india[ top_10_shows_india['Release Year'] >= 2000]
top_10_shows_india[['title', "Out_of_10_rating"]][0:10]


# ## Conclusion :
#    1. There are 33% TV Shows on Netflix and they are focusing on developing more TV Shows as of late.
#    2. Most of the content on Netflix 'TV-MA'.
#    3. Netflix releases large content in Holiday Seasin (October to January).
#    4. Netflix is focusing on expanding their base in India. From 2017, Netflix has started releasing Indian content on the platform.

# This is my first public notebook on Kaggle. Please suggest me further advanced ideas of data exploration to gain valuable insights.

# In[ ]:




