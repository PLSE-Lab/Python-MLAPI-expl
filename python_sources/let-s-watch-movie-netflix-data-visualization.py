#!/usr/bin/env python
# coding: utf-8

# # Let's Watch Movie! Netflix Data Visualization
# ## What is the Trend of Netflix ?
# 
# This notebook is made for visualizing Netflix Dataset and figuring out the Trend of Netflix.
# 
# For example, in this notebook, following questions are answered.
# 
# * Which is more common in this Netflix dataset, **movie or TV show**?
# * What is **the most frequent word in show Title** in this dataset? (Love? Man? Woman? Peace? ...)
# * Who is **the most frequent director** in this Netflix dataset? 
# * **Which country** is the most frequent for these shows? (U.S? U.K? India? China? Korea? Japan? ...)
# * **Which day** is the most popular for publishing shows in Netflix? (Saturday? Sunday? Monday? ...)
# * **From which year**, has the total number of shows in Netflix increased? (About 2016? or more recently?...)
# * **How long** is the typical for the movie? (90 - 120 minutes or so? ...)
# * **Which show type** is the most frequent in Netflix? (Comedy? Love Romance? Action? ...)
# * **Which word** is the most frequent in movie description? (Life? Family? Love? ...)
# 
# Here is table of contents in this notebook.
# 
# [Data and Library Import](#Data_and_Library_Import)  
# [Check dataset](#Check_dataset)  
# [Data Visualization of each column](#Data-Visualization-of-each-column)  
# * [show_id](#show_id-column)  
# * [type](#type-column)
# * [Title](#Title-column)
# * [Director](#Director-column)
# * [Country](#Country-column)
# * [Date-added](#date_added-column)
# * [Year](#Year-column)
# * [Rating](#Rating-column)
# * [Duration](#duration-column)
# * [Listed_in](#listed_in-column)
# * [Description](#Description-column)  
# 
# [Future Work](#Future-Work)

# ## Data_and_Library_Import

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


# import necessary libraries for visualization.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# import sklearn text_processing library
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import nltk
from nltk.stem import PorterStemmer
from wordcloud import WordCloud

sns.set()

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Check_dataset

# Before doing anything, taking a look at dataset is the very first step of data science !!

# In[ ]:


df = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")
df.head(5)


# Next, before diving into each column data visualization, we should check each column type.  

# In[ ]:


# DataFrame Column can be found by using .dtypes method of DataFrame object
df.dtypes


# Hmm, most of the columns are object type.  
# At least, date_added column should be used by **datetime64** type, **not by object** type

# In[ ]:


# In these case, we can use to_datetime method of pandas
df["date_added"] = pd.to_datetime(df["date_added"])


# Then check the column types and make sure date_added is datetime64 [ns]

# In[ ]:


df.dtypes


# OK, date_added column can be now used as datetime

# Next, let's check about NaN data of each column.  
# By using *.info method*, we can find the number of null obeject in each column, and the total length of dataset

# In[ ]:


df.info()


# OK, length of the dataset is 6234.  
# And some columns include NaN values (ex. director, country ...)

# ## Data Visualization of each column

# Now Let's check each dataset one by one.  
# And visualize it with **matplotlib and seaborn** !

# ### show_id column

# In[ ]:


# show_id should be unique,so let's check its uniqueness 
# nunique method can show the number of unique values in dataframe
df.nunique()


# OK, show_id is unique.  
# So let's move on to next column, type!

# ### type_column

# In[ ]:


# First of all, let's check the value of type column
df["type"].head()


# The number of unique values of "type" column is 2.  
# So, it seems"type" is consisted of "Movie" and "TV Show". (No Null values in this column.)  
# Thus, this time, let's use countplot of seaborn.

# In[ ]:


sns.countplot("type", data=df)
plt.title("Show Type Count in Netflix dataset")


# In this Netflix dataset, the number of Movie is about twice as many as that of TV show. 

# ### Title column

# Next, let's take a look at title column.

# In[ ]:


df["title"]


# In title column, they are the text data type.  
# Processing text data is more difficult than processing other numerical data.  
# In this notebook, we process text data by using sklearn!  
# Let's find out the most frequent word in title using Bag of Words! 

# By the way, in jupyte notebook, if you wanna know how to use command,  
# you can find out by using ? keyword like below.   
# (For now, I commented out because the method popped up a window)

# In[ ]:


# CountVectorizer?


# In[ ]:


# Use Bag of Words, and vectorize all the words.
countvectorizer = CountVectorizer(stop_words="english")
bow = countvectorizer.fit_transform(df["title"])
bow.toarray(), bow.shape


# In[ ]:


# Get feature names
feature_names = countvectorizer.get_feature_names()

# View some feature names
feature_names[150:160]


# In[ ]:


# Create data frame (column: words in title, row: each row of original dataframe)
bow_result_df = pd.DataFrame(bow.toarray(), columns=feature_names)
bow_result_df.head()


# In[ ]:


# Let's see the word that is used for 20 times.
frequent_word_df = pd.DataFrame(bow_result_df.sum(), bow_result_df.columns)
frequent_word_df = frequent_word_df.rename(columns={0:"count"})
frequent_word_df = frequent_word_df[frequent_word_df["count"] > 20]
frequent_word_df.head(5)


# Ok, then let's visualize this result!   
# First of all, we want a sorted dataframe by count.

# In[ ]:


frequent_word_sorted_df = frequent_word_df.sort_values("count", ascending=False)
frequent_word_sorted_df.head()


# Then, let's plot the above result!! 

# In[ ]:


plt.figure(figsize=(12, 4))
sns.barplot(frequent_word_sorted_df.index, frequent_word_sorted_df["count"])
plt.xticks(rotation=60)
plt.xlabel("Word")
plt.title("Word Count of Movie Titles")


# This result is very interesting.  
# According to the above result, **"love" is used most frequently in movie titles**.  
# And man the 2nd is man, 3rd is story.

# ### Director column
# First, let's take a look at this column.  
# In this case, we can count the number of directors in the dataset. 

# In[ ]:


# How many NaN is included here?
df["director"].isnull().sum()


# In[ ]:


# pick up directors who directs more than twice.
director_df = df["director"]
director_removed_nan_df = director_df.dropna()
director_removed_nan_df.head()


# In[ ]:


# I want a dictionary which contains how many times does each director appear?
# Key: Director(s) Name, Value: Appearance Count of each directors
director_count = {}

for i in director_removed_nan_df.index:
    director_count.setdefault(director_removed_nan_df[i], 0)
    director_count[director_removed_nan_df[i]] += 1


# In[ ]:


# In the director_count dictionary, we pick up the frequent directors.
# Criteria: Appearance Count is 6 times and above.
frequent_director_count = {}

for key,value in director_count.items():
    if value >= 6:
        frequent_director_count.setdefault(key, value)


# In[ ]:


frequent_director_count


# Sort this directory by values and visualize it.

# In[ ]:


sorted_dict = sorted(frequent_director_count.items(), key=lambda x:x[1], reverse=True)
x = []
y = []
for i in range(len(sorted_dict)):
    x.append(sorted_dict[i][0])
    y.append(sorted_dict[i][1])


# In[ ]:


plt.figure(figsize=(12,4))
sns.barplot(x, y)
plt.xticks(rotation=90)
plt.yticks(np.arange(0, 20, step=1))
plt.title("Director Count in Netflix")
plt.ylabel("Count")
plt.xlabel("Director(s)")


# Oh, Raul Campos and Jan Suter are the most frequent direcors in this Netflix dataset.  
# Have you seen the movies by them?

# ### Country column
# Let's move on to country column. Which country is the most frequent in this dataset?

# In[ ]:


df["country"][0]


# Each movie or show may include multiple countries like above.  
# So use .split method and create country list in these cases.

# In[ ]:


df["country"].dropna()[0].split(",")


# In[ ]:


# Let's create the dictionary that contains how many times does each country appear?
# Key: Country Name, Value: Times that each country appears.
frequent_country = {}

for i in df["country"].dropna().index:
    country_list = df["country"].dropna()[i].split(",")
    for country in country_list:
        frequent_country.setdefault(country, 0)
        frequent_country[country] += 1


# In[ ]:


# Sort it by using sorted function in dictionary.
sorted_dict = sorted(frequent_country.items(), key=lambda x:x[1], reverse=True)
x = []
y = []
for i in range(len(sorted_dict)):
    x.append(sorted_dict[i][0])
    y.append(sorted_dict[i][1])


# In[ ]:


# Plot the result of the countries which is in top 20.
plt.figure(figsize=(12, 4))
sns.barplot(x[:20], y[:20])
plt.xticks(rotation=90)
plt.xlabel("Country")
plt.ylabel("Count")
plt.title("Country Count in Netflix")


# As you can guess, **United States** is the most frequent countries in Netflix!  
# My country, Japan, is the 6th.

# ### date_added column

# In[ ]:


df["date_added"]


# OK, in this column, it seems some NaN values are included.  
# Let's count how many NaNs is this column included at first.

# In[ ]:


df["date_added"].isnull().sum()


# OK, the total number of missing data in this column is 11.  
# Then, let's plot how many movies or TV shows are published in each day
# In such a case, **groupby** method is super useful.  
# By doing the command below, we can get series data which has the publish date as index and the count of the shows as values.

# In[ ]:


date_count_series = df.groupby("date_added")["show_id"].count()
date_count_series.head()


# OK, so far, we could prepare the object that we need to plot the graph.  
# The only thing we have to do is just plotting!!!

# In[ ]:


plt.figure(figsize=(16, 4))
plt.plot(date_count_series.index, date_count_series.values)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())


# This result is very interesting.  
# As you can see, from 2016, the number of shows in this dataset gradually increased.  
# 
# OK, so let's check more detail from 2016.

# In[ ]:


plt.figure(figsize=(16, 4))
plt.plot(date_count_series[date_count_series.index >= "2016-01-01"].index, date_count_series[date_count_series.index >= "2016-01-01"].values)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())


# Oh, this result seems interesting, too.  
# Obviously, peaks appear in some regular intervals.  
# It seems weekly, so let's check it out more precisely!

# In[ ]:


plt.figure(figsize=(16, 4))
plt.plot(date_count_series[date_count_series.index >= "2019-01-01"].index, date_count_series[date_count_series.index >= "2019-01-01"].values)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())


# Oh, it is not weekly peak, but it is rather monthly than weekly.  
# (You can find the peaks in the first day of each month.)  
# Next, which day is the most popular day for publishing shows in Netflix? (Sunday? Saturday? or Weekday?)  
# We can get the day type by using dayofweek 
# (0. Monday, 1.Tuesday, ... 6. Sunday)

# In[ ]:


date_type_df = pd.DataFrame(date_count_series)
date_type_df["day_type"] = date_count_series.index.dayofweek
date_type_df.head(5)


# In[ ]:


grouped_date_type_series = date_type_df.groupby("day_type").count()
grouped_date_type_series


# In[ ]:


day_type=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

plt.bar(x=grouped_date_type_series.index, height=grouped_date_type_series["show_id"])
plt.xticks(np.arange(7), labels=day_type)
plt.ylabel("Count")
plt.title("Day type Count in Netflix")


# To my surprise, Friday is the most typical day for publishing shows in one week.

# ### Year column

# Like what we've done in previous column, that is,  date-added column, we can use groupby to process data by release year!

# In[ ]:


release_year_series = df.groupby("release_year")["show_id"].count()
release_year_series.index = pd.to_datetime(release_year_series.index, format="%Y")
release_year_series.head(5)


# In[ ]:


plt.figure(figsize=(16, 4))
plt.plot(release_year_series.index, release_year_series.values)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(5))
plt.xticks(rotation=60)
plt.title("Publish Year in all shows")
plt.ylabel("Count")
plt.xlabel("Year")


# As you can guess, netflix supplies many movies that is published in recent years.  
# Especially, from 2015, many movies are published and supplied by Netflix.

# ### Rating column

# The type of rating column is object.  
# However, it seems there are only some rating types  
# (i.e. they have some list in this column and pick out one respectively)  
# So we have to know how many unique values are there in this column!

# In[ ]:


df["rating"].nunique()


# OK, we only have 14 unique values in this column, so plot the count below!

# In[ ]:


sns.countplot(df["rating"])
plt.xticks(rotation=90)
plt.title("Rating Count in Netflix")


# ### duration column

# In[ ]:


df["duration"]


# TV show -> Season  
# Movie -> min  
# In this notebook, I only focus on the duration time of Movie.

# In[ ]:


movie_duration_series = pd.DataFrame(df[df["type"] == "Movie"]["duration"])
movie_duration_series.head(5)


# In[ ]:


movie_duration_series = movie_duration_series.replace("(\d*) min", r"\1", regex=True)
movie_duration_series["duration"] = movie_duration_series["duration"].astype("int64")
movie_duration_series.head()


# In[ ]:


plt.hist(movie_duration_series["duration"], bins=20)
plt.xlabel("duration")
plt.ylabel("count")
plt.title("Movies duration histgram in Netflix")


# In some cases, plotting histgram with normed is better than just plotting the real count value.
# So let's normalize the above graph.  
# It is super easy! Just set density=True in hist function.

# In[ ]:


plt.hist(movie_duration_series["duration"], bins=20, density=True)
plt.xlabel("duration")
plt.ylabel("count")
plt.title("Relative Frequency Distribution of Movies duration in Netflix")


# Many movies are around 100 minutes.  
# And it seems the probability distribution is log-normal distribution.  
# We can get some statistical values by describe method!

# In[ ]:


movie_duration_series.describe()


# ### listed_in column

# In[ ]:


df["listed_in"]


# In[ ]:


# Key: show classification, Value: Count 
frequent_listed_in = {}

for i in df["listed_in"].index:
    listed_in_list = df["listed_in"][i].split(",")
    for listed_in in listed_in_list:
        frequent_listed_in.setdefault(listed_in, 0)
        frequent_listed_in[listed_in] += 1


# In[ ]:


sorted_dict = sorted(frequent_listed_in.items(), key=lambda x:x[1], reverse=True)
x = []
y = []
for i in range(len(sorted_dict)):
    x.append(sorted_dict[i][0])
    y.append(sorted_dict[i][1])


# In[ ]:


plt.figure(figsize=(12, 4))
sns.barplot(x[:20], y[:20])
plt.xticks(rotation=90)
plt.xlabel("Show Type")
plt.ylabel("Count")
plt.title("Show Type Count in Netflix")


# In Netflix, International Movies are the most popular.  
# Also, Dramas and Comedies are very popular in Netflix.  
# To my surprise, Romantic Movies are nost so many in this dataset.

# ### Description column
# Finally. let's move on to the last column, Description column.  
# Let's take a look at the first item.

# In[ ]:


df["description"][0]


# In this column, we can use Bag of Words and process the dataset.

# In[ ]:


# Use Bag of Words, and vectorize all the words.
countvectorizer = CountVectorizer(stop_words="english")
bow = countvectorizer.fit_transform(df["description"])
bow.toarray(), bow.shape


# In[ ]:


# Get feature names
feature_names = countvectorizer.get_feature_names()

# View feature names
feature_names[1500:1510]


# In[ ]:


# Create data frame (column: words in description, row: each row of original dataframe)
bow_result_df = pd.DataFrame(bow.toarray(), columns=feature_names)
bow_result_df.head()


# In[ ]:


# Let's see the word that is used for 200 times.
frequent_word_df = pd.DataFrame(bow_result_df.sum(), bow_result_df.columns)
frequent_word_df = frequent_word_df.rename(columns={0:"count"})
frequent_word_df = frequent_word_df[frequent_word_df["count"] > 200]
frequent_word_df.head(5)


# In[ ]:


frequent_word_sorted_df = frequent_word_df.sort_values("count", ascending=False)
frequent_word_sorted_df.head()


# In[ ]:


plt.figure(figsize=(12, 4))
sns.barplot(frequent_word_sorted_df.index, frequent_word_sorted_df["count"])
plt.xticks(rotation=60)
plt.xlabel("Word")
plt.title("Word Count of Movie Description")


# This is very interesting, we can guess the popular topics in Netflix by using this result.  
# Some movies and shows are about someone's life, young people, family, and world.  
# In this case, we can visualize by using word clowd.  

# In[ ]:


wordcloud = WordCloud(background_color="white")


# In[ ]:


wordcloud.generate(" ".join(df["description"]))


# In[ ]:


plt.figure(figsize=(12, 6))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Show Description WordCloud in Netflix dataset")


# By using WordCloud, we can easily visualize the trends in Netflix show description.  
# Looking at this figure, "life", "family", and "find" are most important keywords in this Netflix dataset.

# ## Future Work

# OK, we've done data visualization briefly.  
# Here I propose some future work!
# 
# - Clustering of all shows in this dataset is interesting
# - Use stemming and get more accurate result in Bag of Words is important.
# - If we have users' ratings of all dataset, we can create recommendation system.
# 
# That's all. **Let's watch Movies in Netflix and Stay Home !!**
# 
# **I appreciate your goods, all comments and feedbacks! Thank you!**
