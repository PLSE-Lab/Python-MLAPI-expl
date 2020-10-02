#!/usr/bin/env python
# coding: utf-8

# # What is the motivation?
# 
# This project purpose to make a basic analysis on Goodreads Dataset, so that person can get familiar with this dataset. My main questions to answer are:
# 
# Q1. Top 10 most popular books
# 
# Q2. Top 10 books with highest number of occurrences
# 
# Q3. Correlation Matrix for Rating variables
# 
# Q4. What is the distribution of languages?
# 
# Q5. Is there any pattern between ratings and num_pages?
# 
# Q6. Top 10 authors with highest number of books?
# 
# Q7. Top 10 authors with top rated books?

# In[ ]:


# Importing required libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px


# In[ ]:


pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 1000)


# In[ ]:


df = pd.read_csv('../input/books.csv', error_bad_lines=False)


# Let's explore how the data looks like by looking at first 5 rows of the dataset.

# In[ ]:


df.head()


# # Data Cleaning

# In[ ]:



print(df.shape)


# ## Check missing values for each column 

# In[ ]:


# check missing values for each column 
df.isnull().sum().sort_values(ascending=False)


# In[ ]:


# check out the rows with missing values
df[df.isnull().any(axis=1)].head()


# ##  Take a look at info and summary to get a basic understanding about dataset

# In[ ]:


print(df.info())


# In[ ]:


df.describe()


# In[ ]:


df.describe(include=['object'])


# In[ ]:


print(df.columns)


# In[ ]:


#Rearrange the columns to easier reference
df = df[['bookID', 'title', 'authors', 'average_rating', 
       'language_code', '# num_pages', 'ratings_count', 'text_reviews_count', 'isbn', 'isbn13']]


# Let's see how the dataset looks like now

# In[ ]:


df.head()


# # Exploratory Data Analysis (EDA)

# ##  Q1. Top 10 most popular books

# In[ ]:


sorted_rated_df = df.sort_values(by='ratings_count', ascending=False)[:10]

fig = plt.figure(figsize=(18,10))

sns.barplot(x=sorted_rated_df['ratings_count'], y=sorted_rated_df['title'], palette="rainbow")
plt.title('The top 10 popular books')
plt.show()


# Harry Potter and Twilight have significantly higher number of ratings comparing to that of others. So we can infer that these two books tend to be the most popular ones.

# ## Q2. Top 10 books with highest number of occurrences

# In[ ]:


plt.figure(figsize=(18,10))
titles = df['title'].value_counts()[:10]
sns.barplot(x = titles, y = titles.index, palette='rainbow')
plt.title("10 most Occurring Books")
plt.xlabel("Number of occurrences")
plt.ylabel("Books")
plt.show()


# Second half of the most occuring books have similar # of appearances, while first 2 have much higher occuring count comparing to others. 

# ## Q3. Correlation Matrix for Rating variables

# Here, I was curious is there any correlation between average rating, ratings count and text reviews count?

# In[ ]:


corr_columns = ['average_rating', 'ratings_count', 'text_reviews_count']
corr_mat = df[corr_columns].corr()
corr_mat


# In[ ]:


#Plotting the Correlation matix
plt.figure(figsize=(14,14))
corr = df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(10, 650, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=55,
    horizontalalignment='right'
);


# The most obvious correlation is between ratings count and text reviews count. It's around 0.863467, which means they have a strong correlation. 

# ## Q4. What is the proportion of languages?

# In[ ]:


#Sort all languages that appear less than 200 times (< 200) into category "other" because their number are too small
language_count_df = df[['language_code']].copy()
language_count_df.head()
language_count_df.loc[language_count_df['language_code'].isin((language_count_df['language_code'].value_counts()[language_count_df['language_code'].value_counts() < 200]).index), 'language_code'] = 'other'


# In[ ]:


import plotly.graph_objects as go
labels=language_count_df.language_code.unique()
values=language_count_df['language_code'].value_counts()
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.show()


# It's visible that english US and english take up to 90% of total books count. So there's only a few other languages like spanish, french that account for only 5.5% roughly. While other languages appear very few times.

# ## Q5. Is there any pattern between ratings and num_pages?

# ## Is there a pattern between number of pages and average rating for books?

# In[ ]:


fig = px.scatter(df,x="average_rating", y="# num_pages")
fig.show()


# There's no exact pattern, but we see the majority of books lie between roughly 3.3 and 4.8 rating. Also there are books-outliers that have a huge number of pages around 6500 and the books with perfect 0 average rating.

# ## Is there a pattern between number of pages and ratings count for books?

# In[ ]:


fig = px.scatter(df,x="ratings_count", y="# num_pages")
fig.show()


# Also no exact pattern, but it's visible that books with page range from 100 to 1000 have the most amount of ratings, which shows books with page count up to 1000 are more likely to be popular. Also there outliers with a very huge ratings count and page numbers that don't necessarily show any insight.

# ## Is there any pattern between ratings count and text reviews count?

# In[ ]:


fig = px.scatter(df,y="ratings_count", x="text_reviews_count")
fig.show()


# From correlation, we learn't that these both have the strongest correlation of 0.8. But here we can exaclty see positive trend: the more ratings count, the more text reviews count. The relationship is very logical and reasonable.

# ## Q6. Top 10 authors with highest number of books?

# Let's explore some information about authors.

# In[ ]:


#Sorting the dataset by authors with highest number of books
sorted_books = df.groupby('authors')['title'].count().reset_index().sort_values('title', ascending=False)[:10]
sorted_books.head()


# In[ ]:


fig = plt.figure(figsize=(18,10))

sns.barplot(y=sorted_books['title'], x=sorted_books['authors'], palette="Set1")
plt.title('The top 10 authors with most books')
plt.show()


# Agatha Christie and Stephen King have around 65-68 number of books which is much higher than count for the rest of authors.

# ## Q7. Top 10 authors with top rated books?

# In[ ]:


sorted_authors_ = df.groupby('authors')['title'].count().reset_index().sort_values('title', ascending=False)


# In[ ]:


high_rated_author = df[df['average_rating']>=4.3]
high_rated_author = high_rated_author.groupby('authors')['title'].count().reset_index().sort_values('title', ascending = False).head(10).set_index('authors')
plt.figure(figsize=(15,10))
ax = sns.barplot(high_rated_author['title'], high_rated_author.index, palette='Set1')
ax.set_xlabel("Number of Books")
ax.set_ylabel("Authors")
for i in ax.patches:
    ax.text(i.get_width()+.3, i.get_y()+0.5, str(round(i.get_width())), fontsize = 10, color = 'k')


# While the Tolkien is the most rated author

# # What things can be done more?

# ** My EDA is very basic and makes any person ger familiar with dataset easily.** 
# 
# I believe if there were columns with **genres** and any other metrics like **sales/revenue** to see success of a books numerically, more things could be done.
# 
# But at this point, person can do more pattern recognitions, figure out other insights and do new graphs. This process is highly subjective and depends on that questions are being asked.
# 
# Also the ML application also can take place because things like recommendation engine can be formed.
# 
# Thank you.
