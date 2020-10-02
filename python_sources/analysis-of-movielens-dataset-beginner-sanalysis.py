#!/usr/bin/env python
# coding: utf-8

# ****MovieLens Dataset analysis (Beginner)****
# 
# 
# I am a newbie to the field of data science , and will be attempting to work my way through the MovieLens  dataset. Please consider upvoting if this is useful to you! :) 
# 
# Any feedback is welcome !
# 
# **Contents**
# 
#     1.Importing Libaries
#     2.Reading and Exploring the data
#     3.Data Analysis
#     4.Cleaning of data
#     5.Data Visualization
#     
#     
#   **1.First we import necessary Libaries**
#   
#   
#    
# 
# 
# 
# 
# 

# In[ ]:


import pandas as pd # pandas is a data manipulation library
import numpy as np #provides numerical arrays and functions to manipulate the arrays efficiently
import random
import matplotlib.pyplot as plt # data visualization library
from wordcloud import WordCloud, STOPWORDS #used to generate world cloud


# **2.Reading and Exploring the Data**

# In[ ]:


# lets explore movies.csv
data= pd.read_csv('../input/movies.csv')
data.shape


# In[ ]:


data.head() #displays first 5 entries 


# In[ ]:


data.info()


# In[ ]:


#number of unique movies
movies = data['movieId'].unique().tolist()
len(movies)


# In[ ]:


# lets explore ratings.CSV
ratings_data=pd.read_csv('../input/ratings.csv',sep=',')
ratings_data.shape


# In[ ]:


#summary of ratings.csv
ratings_data.describe()


# In[ ]:


#minimum rating given to a movie
ratings_data['rating'].min() 


# In[ ]:


#maximum rating given to a movie
ratings_data['rating'].max()


# **3.Cleaning of data**

# In[ ]:


# checking movies.csv
data.shape


# In[ ]:


#is any row null
data.isnull().any()


# In[ ]:


#checking ratings.csv
ratings_data.shape


# In[ ]:


#is any row null there
ratings_data.isnull().any()


# In[ ]:


#checking tags.csv
tags_data=pd.read_csv('../input/tags.csv',sep=',')
tags_data.shape


# In[ ]:


#is any row null in tags.csv
tags_data.isnull().any()


# In[ ]:


# lets drop null rows
tags_data=tags_data.dropna()


# In[ ]:


# after cleaning the data there are no more null rows in tags.csv
tags_data.isnull().any()


# In[ ]:


# number of unique tags 
unique_tags=tags_data['tag'].unique().tolist()
len(unique_tags)


# **4.Data Analysis**

# In[ ]:


# filtering to get the list of drama movies
drama_movies=data['genres'].str.contains('Drama')
data[drama_movies].head()


# In[ ]:


#total number of drama movies
drama_movies.shape


# In[ ]:


#filtering to get the list of comedy movies
comedy_movies = data['genres'].str.contains('Comedy')
data[comedy_movies].head()


# In[ ]:


#total no. of comedy movies
comedy_movies.shape


# In[ ]:


#search movie id by tag search
tag_search = tags_data['tag'].str.contains('dark')
tags_data[tag_search].head()


# In[ ]:


#displays first 5 data from a dataframe
#here rating.csv has 4 columns
ratings_data.head() 


# In[ ]:


del ratings_data['timestamp']


# In[ ]:


#displays first 5 data from a dataframe
#here ratings.csv has 3 columns
ratings_data.head() 


# In[ ]:


#displays first 5 data from a dataframe
#here movies.csv has 3 columns
data.head()


# In[ ]:


#merging two dataframes "movies.csv" and "ratings.csv"
movie_data_ratings_data=data.merge(ratings_data,on = 'movieId',how = 'inner')
movie_data_ratings_data.head(3)


# In[ ]:


#displays high rated movies
high_rated= movie_data_ratings_data['rating']>4.0
movie_data_ratings_data[high_rated].head(10)


# In[ ]:


# displays low rated movies
low_rated = movie_data_ratings_data['rating']<4.0
movie_data_ratings_data[low_rated].head()


# In[ ]:


#total number of unique movie genre
unique_genre=data['genres'].unique().tolist()
len(unique_genre)


# In[ ]:


#top 25 most rated movies
most_rated = movie_data_ratings_data.groupby('title').size().sort_values(ascending=False)[:25]
most_rated.head(25)


# In[ ]:


#slicing out columns to display only title and genres columns from movies.csv
data[['title','genres']].head()


# In[ ]:


# here we extract year from title
data['year'] =data['title'].str.extract('.*\((.*)\).*',expand = False)
data.head(5)


# In[ ]:


#define a function that counts the number of times each genre appear:
def count_word(df, ref_col, liste):
    keyword_count = dict()
    for s in liste: keyword_count[s] = 0
    for liste_keywords in df[ref_col].str.split('|'):
        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue
        for s in liste_keywords: 
            if pd.notnull(s): keyword_count[s] += 1
    # convert the dictionary in a list to sort the keywords  by frequency
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count


# In[ ]:


#here we  make census of the genres:
genre_labels = set()
for s in data['genres'].str.split('|').values:
    genre_labels = genre_labels.union(set(s))


# In[ ]:


#counting how many times each of genres occur:
keyword_occurences, dum = count_word(data, 'genres', genre_labels)
keyword_occurences


# **5.Data Visualization**

# In[ ]:


# Function that control the color of the words
def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    h = int(360.0 * tone / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)


#Finally, the result is shown as a wordcloud:
words = dict()
trunc_occurences = keyword_occurences[0:50]
for s in trunc_occurences:
    words[s[0]] = s[1]
tone = 100 # define the color of the words
f, ax = plt.subplots(figsize=(14, 6))
wordcloud = WordCloud(width=550,height=300, background_color='black', 
                      max_words=1628,relative_scaling=0.7,
                      color_func = random_color_func,
                      normalize_plurals=False)
wordcloud.generate_from_frequencies(words)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[ ]:


# lets display the same result in the histogram
fig = plt.figure(1, figsize=(18,13))
ax2 = fig.add_subplot(2,1,2)
y_axis = [i[1] for i in trunc_occurences]
x_axis = [k for k,i in enumerate(trunc_occurences)]
x_label = [i[0] for i in trunc_occurences]
plt.xticks(rotation=85, fontsize = 15)
plt.yticks(fontsize = 15)
plt.xticks(x_axis, x_label)
plt.ylabel("No. of occurences", fontsize = 24, labelpad = 0)
ax2.bar(x_axis, y_axis, align = 'center', color='r')
plt.title("Popularity of Genres",bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 30)
plt.show()


# In[ ]:




