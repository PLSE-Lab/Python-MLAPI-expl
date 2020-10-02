#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import random
from wordcloud import WordCloud, STOPWORDS


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df=pd.read_csv("/kaggle/input/tmdb-top-10000-popular-movies-dataset/movies_tmdb_popular.csv")


# In[ ]:


df.head()


# In[ ]:


df['rel_date'] = pd.to_datetime(df['rel_date'])


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df=df.dropna()


# In[ ]:


df.info()


# In[ ]:


plt.figure(figsize=(20, 8))
plt.xlabel("Popularity")
plt.ylabel('Vote count')
plt.title("Populaity vs Vote Count (For All Films)")
plt.scatter(df['popularity'],df["vote_count"],marker=".")


# In[ ]:


text = (str(df['title']))
plt.subplots(figsize=(16,10))
wordcloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          max_words=10000,
                          width=1400,
                          height=1200
                         ).generate(text)


plt.imshow(wordcloud)
plt.title('Movie Title WordCloud')
plt.axis('off')
plt.show()


# In[ ]:


text = (str(df['overview']))
plt.subplots(figsize=(16,10))
wordcloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          max_words=100000000,
                          width=1400,
                          height=1200
                         ).generate(text)


plt.imshow(wordcloud)
plt.title('Movie Overview WordCloud')
plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:


#seeing which films have 9 and above ratings
df1=df[df["vote_average"]>=9]


# In[ ]:


df1.info()


# In[ ]:


#These are the movies having 9 and above ratings.
df1["title"]


# In[ ]:


lang_num = df['original_lang'].nunique()
print("Number of languages Films have been made in are-", lang_num)


# In[ ]:


lang_values = df['original_lang'].unique()


# In[ ]:


print("The languages are-",lang_values)


# # **English Movies**

# In[ ]:


#selecting english movies
eng=df[df["original_lang"]=="en"]


# In[ ]:


print("Number of English Movies=",len(eng))


# In[ ]:


eng.head()


# In[ ]:


print("English Movies with highest popularity are= ")
print((eng.sort_values("popularity",ascending=False).head(10))['title'])


# In[ ]:


print("English Movies with lowest popularity are= ")
print((eng.sort_values("popularity",ascending=True).head(10))['title'])


# In[ ]:


temp=(eng.sort_values(["vote_average","popularity"],ascending=[False,False]).head(10))["title"]
print("The movies with highest vote rating and sorted by popularity are= ")
print(temp.values)


# In[ ]:


print("English Movies with highest vote counts are= ")
print((eng.sort_values("vote_count",ascending=False).head(10))['title'])


# In[ ]:


print("English Movies with lowest vote counts are= ")
print((eng.sort_values("vote_count",ascending=True).head(10))['title'])


# In[ ]:


#English movies in 2019

eng_2019=eng[(eng["rel_date"].dt.year)==2019]


# In[ ]:


eng_2019.head()


# In[ ]:


print("English Movies of 2019 with highest popularity are= ")
print((eng_2019.sort_values("popularity",ascending=False).head(10))['title'])


# In[ ]:


print("English Movies of 2019 with highest vote count are= ")
print((eng_2019.sort_values("vote_count",ascending=False).head(10))['title'])


# In[ ]:


print("English Movies of 2019 with highest vote rating are= ")
print((eng_2019.sort_values("vote_average",ascending=False).head(10))['title'])


# In[ ]:





# In[ ]:


#English movies in 2016

eng_2016=eng[(eng["rel_date"].dt.year)==2016]


# In[ ]:


eng_2016.head()


# In[ ]:


print("English Movies of 2016 with highest popularity are= ")
print((eng_2016.sort_values("popularity",ascending=False).head(10))['title'])


# In[ ]:


print("English Movies of 2016 with lowest popularity are= ")
print((eng_2016.sort_values("popularity",ascending=True).head(10))['title'])


# In[ ]:


#Distribution of Vote Ratings for English movies
plt.figure(figsize=(16, 6))
sns.distplot(eng["vote_average"],kde=False)


# In[ ]:


#Vote Count of English Films
#x axis- Films
#y axis- Number of Votes

plt.figure(figsize=(16, 6))
sns.lineplot(data=eng['vote_count'], linewidth=1)


# In[ ]:





# In[ ]:





# # **Spanish Movies**

# In[ ]:


#selecting spanish movies
esp=df[df["original_lang"]=="es"]


# In[ ]:


print('Number of Spanish Movies are= ',len(esp))


# In[ ]:


esp.head()


# In[ ]:


text = (str(esp['title']))
plt.subplots(figsize=(16,10))
wordcloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          max_words=100000,
                          width=1400,
                          height=1200
                         ).generate(text)


plt.imshow(wordcloud)
plt.title('Spanish Movie Title WordCloud')
plt.axis('off')
plt.show()


# In[ ]:


text = (str(esp['overview']))
plt.subplots(figsize=(16,10))
wordcloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          max_words=100000000,
                          width=1400,
                          height=1200
                         ).generate(text)


plt.imshow(wordcloud)
plt.title('Spanish Movie Overview WordCloud')
plt.axis('off')
plt.show()


# In[ ]:


print("Spanish Movies with highest popularity are= ")
print((esp.sort_values("popularity",ascending=False).head(10))['title'])


# In[ ]:


print("Spanish Movies with highest vote count are= ")
print((esp.sort_values("vote_count",ascending=False).head(10))['title'])


# In[ ]:


print("Spanish Movies with highest Vote Score are= ")
print((esp.sort_values("vote_average",ascending=False).head(10))['title'])


# In[ ]:


#Spanish movies in 2019

esp_2019=esp[(esp["rel_date"].dt.year)==2019]


# In[ ]:


esp_2019.head()


# In[ ]:


print("Spanish Movies (2019) with highest popularity are= ")
print((esp_2019.sort_values("popularity",ascending=False).head(10))['title'])


# In[ ]:


print("Spanish Movies (2019) with highest Vote count are= ")
print((esp_2019.sort_values("vote_count",ascending=False).head(10))['title'])


# In[ ]:


print("Spanish Movies (2019) with highest Vote Rating are= ")
print((esp_2019.sort_values("vote_average",ascending=False).head(10))['title'])


# In[ ]:


#Thank You


# In[ ]:




