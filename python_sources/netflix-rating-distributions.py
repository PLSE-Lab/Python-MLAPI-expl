#!/usr/bin/env python
# coding: utf-8

# This is an analysis of the Netflix shows based on the TV Show Ratings, with respect to different Years and the User Score Ratings. If you like it, please upvote!

# ## Loading the necessary Modules ##

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
from io import StringIO
import collections as co

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/Netflix Shows.csv', encoding='cp437')
df.head()


# ## Check for Missing Values ##

# In[ ]:


print('The number of rows with Missing Values are: ')
df.isnull().any(axis=1).sum()


# ## Removing Duplicates ## 

# In[ ]:


df = df.drop_duplicates(keep="first").reset_index(drop=True)


# In[ ]:


df.info()


# **'"ratingLevel"** and **"user rating score"** seem to have missing values.

# ## Total Number of Ratings ##

# In[ ]:


print('In all, there are ',df['rating'].nunique(),'types of ratings in the dataset: ',df['rating'].unique())
#df_rating=df['rating'].unique()


# ## Total Number of Years ##

# In[ ]:


print('In all, there are ',df['release year'].nunique(),'years in the dataset: ',df['release year'].unique())
#df_year=df['release year'].unique()


# ## Data Visualization ##

# ## Which years had the most shows? ##

# In[ ]:


print("The Year-wise distribution of Netflix shows")
year_no_of_shows=df["release year"].value_counts().sort_values(ascending=False)
plt.figure(figsize=(12,4))
year_no_of_shows.plot(title='Years with the number of Netlfix Shows',kind="bar")


# The Top 5 years with **most number of shows**:
# 
#  1. 2016
#  2. 2015
#  3. 2017
#  4. 2014
#  5. 2013

# In[ ]:


print('The number of Netflix Shows in the dataset are: ',df['title'].nunique())


# ## How is the Rating distribution? ##

# In[ ]:


plt.figure(figsize=(12,12))
df.rating.value_counts().plot(kind='pie',autopct='%1.1f%%')
plt.title('Number of appearances in dataset')
plt.show()


# Most **'ratings'** in the dataset are:
# 
#  1. TV-14
#  2. PG
#  3. TV-MA
#  4. G
#  5. TV-Y
# 
# 

# ## Data Description ##

# In[ ]:


df.describe()


# ## What is the most frequent range of User Rating Scores? ##

# In[ ]:


user_rating_score=df.groupby("user rating score")['title'].count().reset_index().sort_values(by='user rating score',ascending=False).reset_index(drop=True)
plt.figure(figsize=(12,6))
sns.barplot(x='user rating score',y='title', data=user_rating_score)
plt.xticks(rotation=45)


# The above graph plot shows the distribution of User Rating Scores. Ranges of scores from **91-98** seem to be more frequent. 

# ## Year-wise Ratings ##

# In[ ]:


print('The Ratings along with their occurence in every year:')
df.groupby((['release year', 'rating'])).size()


# ## Average User Rating Score of Different Ratings by Years ##

# In[ ]:


plt.subplots(figsize=(12,10))
max_ratings=df.groupby('rating')['rating'].count()
max_ratings=max_ratings[max_ratings.values>50]
max_ratings.sort_values(ascending=True,inplace=True)
mean_shows=df[df['rating'].isin(max_ratings.index)]
piv=mean_shows.groupby(['release year','rating'])['user rating score'].mean().reset_index()
piv=piv.pivot('release year','rating','user rating score')
sns.heatmap(piv,annot=True,cmap='YlGnBu')
plt.title('Average User Score By Rating')
plt.show()


# ## What's with 'Rating Level'? ##

# In[ ]:


df1=df[df['ratingLevel'].notnull()]


#  - Removing rows with no values

# In[ ]:


string=StringIO()
df1['ratingLevel'].apply(lambda x: string.write(x))
x=string.getvalue()
string.close()
x=x.lower()
x=x.split()


# In[ ]:


words = co.Counter(nltk.corpus.words.words())
stopWords =co.Counter( nltk.corpus.stopwords.words() )
x=[i for i in x if i in words and i not in stopWords]
string=" ".join(x)
c = co.Counter(x)


# In[ ]:


most_common_10=c.most_common(10)
print('The 10 Most Common Words in ratingLevel are: ')
most_common_10


# In[ ]:


text = string
wordcloud2 = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          width=8000,
                          height=5000,
                          relative_scaling = 1.0).generate(text)
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()


# ## What are the Popular Shows? ##

# In[ ]:


wordcloud1 = WordCloud(
                          background_color='black',
                          width=8000,
                          height=5000,
                          relative_scaling = 1.0
                         ).generate(" ".join(df['title']))
plt.imshow(wordcloud1)
plt.axis('off')
plt.show()

