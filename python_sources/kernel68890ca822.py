#!/usr/bin/env python
# coding: utf-8

# <p style="font-family: Arial; font-size:3.75em;color:purple; font-style:bold"><br>
# Pandas</p><br>
# 
# *pandas* is a Python library for data analysis. It offers a number of data exploration, cleaning and transformation operations that are critical in working with data in Python. 
# 
# *pandas* build upon *numpy* and *scipy* providing easy-to-use data structures and data manipulation functions with integrated indexing.
# 
# The main data structures *pandas* provides are *Series* and *DataFrames*. After a brief introduction to these two data structures and data ingestion, the key features of *pandas* this notebook covers are:
# * Generating descriptive statistics on data
# * Data cleaning using built in pandas functions
# * Frequent data operations for subsetting, filtering, insertion, deletion and aggregation of data
# * Merging multiple datasets using dataframes
# * Working with timestamps and time-series data
# 
# **Additional Recommended Resources:**
# * *pandas* Documentation: http://pandas.pydata.org/pandas-docs/stable/
# * *Python for Data Analysis* by Wes McKinney
# * *Python Data Science Handbook* by Jake VanderPlas
# 
# Let's get started with our first *pandas* notebook!

# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold"><br>
# 
# Import Libraries
# </p>

# In[ ]:


import pandas as pd


# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold">
# Introduction to pandas Data Structures</p>
# <br>
# *pandas* has two main data structures it uses, namely, *Series* and *DataFrames*. 
# 
# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold">
# pandas Series</p>
# 
# *pandas Series* one-dimensional labeled array. 
# 

# In[ ]:


# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
fruit_sales = pd.DataFrame({'Apples':pd.Series(data=[35,21],index=['2017 Sales']),'Bananas':pd.Series(data=[41,34],index=['2018 Sales'])})

q2.check()
fruit_sales


# In[ ]:


ser = pd.Series([100,2,3],['a','b','c'])
ser


# In[ ]:


ser = pd.Series(data=[100, 200, 300, 400, 500], index=['tom', 'bob', 'nancy', 'dan', 'eric'])


# In[ ]:


ser


# In[ ]:


ser.index


# In[ ]:


ser[['nancy','dan']]


# In[ ]:


ser['nancy']


# In[ ]:


ser[[4, 3, 1]]


# In[ ]:


'bob' in ser


# In[ ]:


ser


# In[ ]:


ser['dan']='suresh'


# In[ ]:


ser * 2


# In[ ]:


ser['dan']=2000


# In[ ]:


ser


# In[ ]:


ser ** 2
#ser[['tom','bob']]**2


# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold">
# pandas DataFrame</p>
# 
# *pandas DataFrame* is a 2-dimensional labeled data structure.

# <p style="font-family: Arial; font-size:1.25em;color:#2462C0; font-style:bold">
# Create DataFrame from dictionary of Python Series</p>

# In[ ]:


d = {'one' : pd.Series([100., 200., 300.], index=['apple', 'ball', 'clock']),
     'two' : pd.Series([111., 222., 333., 4444.], index=['apple', 'ball', 'cerill', 'dancy'])}


# In[ ]:


df = pd.DataFrame(d)
#print(df)
df


# In[ ]:


df.index


# In[ ]:


df.columns


# In[ ]:


pd.DataFrame(d, index=['dancy', 'ball', 'apple'])


# In[ ]:


pd.DataFrame(d, index=['dancy', 'ball', 'apple'], columns=['two','five'])


# <p style="font-family: Arial; font-size:1.25em;color:#2462C0; font-style:bold">
# Create DataFrame from list of Python dictionaries</p>

# In[ ]:


data = [{'alex': 1, 'joe': 2}, {'ema': 5, 'dora': 10, 'alice':20}]


# In[ ]:


pd.DataFrame(data)


# In[ ]:


pd.DataFrame(data, index=['orange', 'red'])


# In[ ]:


pd.DataFrame(data, columns=['joe', 'dora','alice','a'])


# <p style="font-family: Arial; font-size:1.25em;color:#2462C0; font-style:bold">
# Basic DataFrame operations</p>

# In[ ]:


df


# In[ ]:


df['one']


# In[ ]:


df['three'] = df['one'] * df['two']
df


# In[ ]:


df['flag'] = df['one'] > 250
df


# In[ ]:


three = df.pop('three')


# In[ ]:


three


# In[ ]:


df


# In[ ]:


del df['one']


# In[ ]:


df


# In[ ]:


df.insert(1, 'copy_of_onee', df['two'])
df


# In[ ]:


df['one_upper_half'] = df['two'][:2]
df


# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold">
# Case Study: Movie Data Analysis</p>
# <br>This notebook uses a dataset from the MovieLens website. We will describe the dataset further as we explore with it using *pandas*. 
# 
# ## Download the Dataset
# 
# ### Please note that **you will need to download the dataset**. 
# 
# Although the video for this notebook says that the data is in your folder, the folder turned out to be too large to fit on the edX platform due to size constraints.
# 
# Here are the links to the data source and location:
# * **Data Source:** MovieLens web site (filename: ml-20m.zip)
# * **Location:** https://grouplens.org/datasets/movielens/
# 
# Once the download completes, please make sure the data files are in a directory called **movielens** in your **Week-4-pandas** folder. 
# 
# Let us look at the files in this dataset using the UNIX command ls.
# 

# In[ ]:


# Note: Adjust the name of the folder to match your local directory

get_ipython().system('ls ./movielens')


# In[ ]:


get_ipython().system('cat ./movielens/movies.csv')


# In[ ]:





# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold">
# Use Pandas to Read the Dataset<br>
# </p>
# <br>
# In this notebook, we will be using three CSV files:
# * **ratings.csv :** *userId*,*movieId*,*rating*, *timestamp*
# * **tags.csv :** *userId*,*movieId*, *tag*, *timestamp*
# * **movies.csv :** *movieId*, *title*, *genres* <br>
# 
# Using the *read_csv* function in pandas, we will ingest these three files.

# In[ ]:


movies = pd.read_csv('../input/movielens-20m-dataset/movie.csv', sep=',')
#print(type(movies))
movies.head()


# In[ ]:


# Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970

tags = pd.read_csv('../input/movielens-20m-dataset/genome_tags.csv', sep=',')
tags.head()


# In[ ]:


#ratings = pd.read_csv('./movielens/ratings.csv', sep=',', parse_dates=['timestamp'])
ratings = pd.read_csv('../input/movielens-20m-dataset/rating.csv', sep=',')
ratings.head()


# In[ ]:


ratings


# In[ ]:


# For current analysis, we will remove timestamp (we will come back to it!)

del ratings['timestamp']
#del tags['timestamp']


# In[ ]:


row = ratings.iloc[5]
row


# <h1 style="font-size:2em;color:#2467C0">Data Structures </h1>

# In[ ]:


col = ratings.ix[5]
col


# <h1 style="font-size:1.5em;color:#2467C0">Series</h1>

# In[ ]:


#Extract 0th row: notice that it is infact a Series

row_0 = tags.iloc[5]
row_0
#type(row_0)


# In[ ]:


print(row_0)


# In[ ]:


row_0.index


# In[ ]:


row_0['tagId']


# In[ ]:


'rating' in row_0


# In[ ]:


row_0.name


# In[ ]:


row_0 = row_0.rename('first_row')
row_0.name


# <h1 style="font-size:1.5em;color:#2467C0">DataFrames </h1>

# In[ ]:


tags.head(10)


# In[ ]:


tags.index


# In[ ]:


tags.columns


# In[ ]:


# Extract row 0, 11, 2000 from DataFrame

tags.iloc[ [0,11,1127] ]


# In[ ]:


ratings


# <h1 style="font-size:2em;color:#2467C0">Descriptive Statistics</h1>
# 
# Let's look how the ratings are distributed! 

# In[ ]:


ratings['rating'].describe()


# In[ ]:


ratings.describe()


# In[ ]:


ratings['rating'].mean()


# In[ ]:


ratings.mean()


# In[ ]:


ratings.min()


# In[ ]:


ratings['rating'].max()


# In[ ]:


ratings['rating'].std()


# In[ ]:


ratings['rating'].mode()


# In[ ]:


ratings.corr()


# In[ ]:


filter_1 = ratings['rating'] > 5

filter_1.any()


# In[ ]:


filter_1


# In[ ]:


filter_2 = ratings['rating'] > 0
filter_2.all()


# <h1 style="font-size:2em;color:#2467C0">Data Cleaning: Handling Missing Data</h1>

# In[ ]:


movies.shape


# In[ ]:


#is any row NULL ?

movies.isnull().any()


# In[ ]:


ratings = pd.read_csv('../input/movielens-20m-dataset/rating.csv', sep=',')


# That's nice! No NULL values!

# In[ ]:


ratings.shape


# In[ ]:


#is any row NULL ?

ratings.isnull().any()


# That's nice! No NULL values!

# In[ ]:


tags.shape


# In[ ]:


#is any row NULL ?

tags.isnull().any()


# We have some tags which are NULL.

# In[ ]:


tags = tags.dropna()


# In[ ]:


#Check again: is any row NULL ?

tags.isnull().any()


# In[ ]:


tags.shape


# That's nice! No NULL values! Notice the number of lines have decreased.

# <h1 style="font-size:2em;color:#2467C0">Data Visualization</h1>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

ratings.hist(column='rating', figsize=(15,10))


# In[ ]:


ratings.boxplot(column='rating',figsize=(15,10))


# <h1 style="font-size:2em;color:#2467C0">Slicing Out Columns</h1>
#  

# In[ ]:


tags['tag'].head()


# In[ ]:


movies[['title','genres']].head()


# In[ ]:


ratings[1000:1010]


# In[ ]:


tags


# In[ ]:


tag_counts = tags['tag'].value_counts()
tag_counts.head(50)


# In[ ]:





# In[ ]:





# In[ ]:


tag_counts[:10].plot(kind='bar', figsize=(15,10))


# <h1 style="font-size:2em;color:#2467C0">Filters for Selecting Rows</h1>

# In[ ]:


is_highly_rated = ratings['rating'] >= 4.0

ratings[is_highly_rated][-5:]


# In[ ]:


is_animation = movies['genres'].str.contains('Animation')

movies[is_animation][5:15]

war = movie['genres'].str.contains['war']
# In[ ]:


war = movies['genres'].str.contains('War')
movies[war]


# In[ ]:


movies[is_animation].head(15)


# <h1 style="font-size:2em;color:#2467C0">Group By and Aggregate </h1>

# In[ ]:


ratings_count = ratings[['movieId','rating']].groupby('rating').count()
ratings_count


# In[ ]:


average_rating = ratings[['movieId','rating']].groupby('movieId').mean()
average_rating.tail()


# In[ ]:


movie_count = ratings[['movieId','rating']].groupby('movieId').count()
movie_count.head()


# In[ ]:


movie_count = ratings[['movieId','rating']].groupby('movieId').count()
movie_count.tail()


# <h1 style="font-size:2em;color:#2467C0">Merge Dataframes</h1>

# In[ ]:


tags.head()


# In[ ]:


tags['movieId']=tags['tagId']
tags


# In[ ]:





# In[ ]:


movies.head()


# In[ ]:


t = movies.merge(tags, on='movieId', how='inner')
t.head()


# More examples: http://pandas.pydata.org/pandas-docs/stable/merging.html

# <p style="font-family: Arial; font-size:1.75em;color:#2462C0; font-style:bold"><br>
# 
# 
# Combine aggregation, merging, and filters to get useful analytics
# </p>

# In[ ]:


avg_ratings = ratings.groupby('movieId', as_index=False).mean()
del avg_ratings['userId']
avg_ratings.head()


# In[ ]:


box_office = movies.merge(avg_ratings, on='movieId', how='inner')
box_office.tail()


# In[ ]:


is_highly_rated = box_office['rating'] >= 4.0

box_office[is_highly_rated][-5:]


# In[ ]:


is_comedy = box_office['genres'].str.contains('Comedy')

box_office[is_comedy][:5]


# In[ ]:


box_office[is_comedy & is_highly_rated][-5:]


# <h1 style="font-size:2em;color:#2467C0">Vectorized String Operations</h1>
# 

# In[ ]:


movies.head()


# <p style="font-family: Arial; font-size:1.35em;color:#2462C0; font-style:bold"><br>
# 
# Split 'genres' into multiple columns
# 
# <br> </p>

# In[ ]:


movie_genres = movies['genres'].str.split('|', expand=True)


# In[ ]:


movie_genres[:10]


# <p style="font-family: Arial; font-size:1.35em;color:#2462C0; font-style:bold"><br>
# 
# Add a new column for comedy genre flag
# 
# <br> </p>

# In[ ]:


movie_genres['isComedy'] = movies['genres'].str.contains('Comedy')


# In[ ]:


movie_genres[:10]


# <p style="font-family: Arial; font-size:1.35em;color:#2462C0; font-style:bold"><br>
# 
# Extract year from title e.g. (1995)
# 
# <br> </p>

# In[ ]:


movies['year'] = movies['title'].str.extract('.*\((.*)\).*', expand=True)


# In[ ]:


movies.tail()


# <p style="font-family: Arial; font-size:1.35em;color:#2462C0; font-style:bold"><br>
# 
# More here: http://pandas.pydata.org/pandas-docs/stable/text.html#text-string-methods
# <br> </p>

# <h1 style="font-size:2em;color:#2467C0">Parsing Timestamps</h1>

# Timestamps are common in sensor data or other time series datasets.
# Let us revisit the *tags.csv* dataset and read the timestamps!
# 

# In[ ]:


tags = pd.read_csv('../input/movielens-20m-dataset/tag.csv', sep=',')


# In[ ]:


tags.dtypes


# <p style="font-family: Arial; font-size:1.35em;color:#2462C0; font-style:bold">
# 
# Unix time / POSIX time / epoch time records 
# time in seconds <br> since midnight Coordinated Universal Time (UTC) of January 1, 1970
# </p>

# In[ ]:


tags.head(5)


# In[ ]:


tags['parsed_time'] = pd.to_datetime(tags['timestamp'], unit='s')


# <p style="font-family: Arial; font-size:1.35em;color:#2462C0; font-style:bold">
# 
# Data Type datetime64[ns] maps to either <M8[ns] or >M8[ns] depending on the hardware
# 
# </p>

# In[ ]:



tags['parsed_time'].dtype


# In[ ]:


tags.head(2)


# <p style="font-family: Arial; font-size:1.35em;color:#2462C0; font-style:bold">
# 
# Selecting rows based on timestamps
# </p>

# In[ ]:


greater_than_t = tags['parsed_time'] > '2015-02-01'

selected_rows = tags[greater_than_t]

tags.shape, selected_rows.shape


# <p style="font-family: Arial; font-size:1.35em;color:#2462C0; font-style:bold">
# 
# Sorting the table using the timestamps
# </p>

# In[ ]:


tags.sort_values(by='parsed_time', ascending=True)[:10]


# <h1 style="font-size:2em;color:#2467C0">Average Movie Ratings over Time </h1>
# ## Are Movie Ratings related to the Year of Launch?

# In[ ]:


average_rating = ratings[['movieId','rating']].groupby('movieId', as_index=False).mean()
average_rating.tail()


# In[ ]:


joined = movies.merge(average_rating, on='movieId', how='inner')
joined.head()
joined.corr()


# In[ ]:


yearly_average = joined[['year','rating']].groupby('year', as_index=False).mean()
yearly_average[:10]


# In[ ]:


yearly_average[-20:].plot(x='year', y='rating', figsize=(15,10), grid=True)


# <p style="font-family: Arial; font-size:1.35em;color:#2462C0; font-style:bold">
# 
# Do some years look better for the box office movies than others? <br><br>
# 
# Does any data point seem like an outlier in some sense?
# 
# </p>

# In[ ]:




