#!/usr/bin/env python
# coding: utf-8

# # Analysing IMDB Ratings over last 3 years 
# 
# 1. Insigth 1 - I've watched movies from a total of 51 different years of release. The earliest being 1931.
# 2. The mean Rating given to mivies by me is 7.63, as opposed to 7.78 imdb average. (Kinda fall in line with the rest)
# 3. Average number of ratings of movies watched by me 293000, (not quite the hipster I thought I was.)
# 4. Average runtime of movies rated by me, 117 mins. (Almost 2 hours, Have always been comfortable with longer movies)
# 5. Standard deviation of ratings is 1.02 , as compared to 0.66 by the rest of the group. (More varied ratings by me)
# 6. My Median rating is 8 (generally rate movies favourably), imdb is 7.8
# 7. Median runtime is 115 mins, year is 2013, num_votes is 171000.
# 8. My Mode rating (most common) is 8 (so is imdbs), most movies watched from year 2018. Mode Runtime is 122 (sont think this is of any value).
# 9. Counting the genres of movies watched reveals my most commonly watched genres include (Comedy, Drama) - 34,Comedy, (Drama, Romance)-25, (Drama)-20, (Drama, Romance)-19, (Comedy, Romance)-10
# 10. Insight 2 - It is safe to say that I like, dramas, comedies, Roamance movies or a combination of them the most.
# 11. Surprisingly I'havent even rated a single movie from the year of my birth.
# 12. Most movies watched from a single director - Christopher nolan, Hirani, Spielberg, wes anderson, fincher.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#ratings = pd.read_csv('ratings.csv', engine='python')
ratings = pd.read_csv("../input/imdb-data-of-watchedrated-movies/ratings.csv", engine = 'python')

ratings.head()
ratings.info()


# In[ ]:


# lets see how how many different years I've seen movies from

no_unique_years = ratings['Year'].unique().tolist()

print(len(no_unique_years))

#print(no_unique_years)
no_unique_years.sort()
print(no_unique_years,'\n')
print(ratings.describe(), '\n')

print(ratings.median(numeric_only=True), '\n')
ratings.mode(numeric_only=True), '\n'

genres = ratings['Genres'].unique().tolist()
print(len(genres))
#print(genres)

print(ratings['Genres'].value_counts().to_string())


# In[ ]:


# number of unique directors whose films I've watched 
directors = ratings['Directors'].unique().tolist()

# Most films by a particular director I've watched.

#print(directors.value_counts()) # value_counts works only for pandas dataframe/series

# for printing entire dataframe, convert it to string using to_string method.

print(ratings['Directors'].value_counts().to_string())


# In[ ]:


# Title analysis 
import random
from wordcloud import WordCloud, STOPWORDS

text = (str(ratings['Title'].value_counts()))

plt.subplots(figsize = (20,15))
wordcloud = WordCloud(
                        stopwords= STOPWORDS,
background_color = 'white',
width =1500,
height = 1200).generate(text)

plt.imshow(wordcloud)
plt.show()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

seperate_genre='Action','Adventure','Animation','Biography','Comedy','Crime','Drama','Fantasy','Family','History','Horror','Music','Musical','Mystery','Romance','Sci-Fi','Sport','Thriller','War','Western'

type(seperate_genre)

for genre in seperate_genre :
    n_df = ratings['Genres'].str.contains(genre).fillna(False)    # create new dataframe that contains only rows with genres listed above 
    print("The total number of movies with ", genre, '=' , len(ratings[n_df]))   # filtering for dataframe, dataframe[dataframe] 

#plt.xlabel(ratings['Genres'])

#plt.hist(gen_str.split(','),354)
ratings_1 = ratings.head(20)
plt.figure(figsize=(25,8))
ax = sns.countplot(x="Year", data=ratings)
#plt.show()
plt.savefig("Movies_years_count.png")


# In[ ]:


seperate_genre='Action','Adventure','Animation','Biography','Comedy','Crime','Drama','Fantasy','Family','History','Horror','Music','Musical','Mystery','Romance','Sci-Fi','Sport','Thriller','War','Western'
for genre in seperate_genre:
    df = ratings['Genres'].str.contains(genre).fillna(False)
    print('The total number of movies with ',genre,'=',len(ratings[df]))
    f, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Year', data=ratings[df], palette="Greens_d");
    plt.title(genre)
    compare_movies_rating = [ 'Num Votes','IMDb Rating','Year']
    for compare in compare_movies_rating:
        sns.jointplot(x='Year' , y='IMDb Rating', data=ratings[df], alpha=0.7, color='b', size=8)
        plt.title(genre)

