#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
# Any results you write to the current directory are saved as output.


# In[ ]:


#Defining the directory path 
# Read CSV file
directory = "../input/"
#
imdb_movies = pd.read_csv(directory+"imdb-movies.csv")


# ## IMDB Data Correlation

# In[ ]:


# imdb_movies.drop(columns=['id'],inplace=True)


# In[ ]:


correlation = imdb_movies.corr()
correlation


# In[ ]:


plt.figure(figsize=(14,14))
sns.heatmap(correlation,annot=True,linewidths=0.01,vmax=1,square=True,cbar=True);
get_ipython().run_line_magic('pinfo', 'sns.heatmap')


# ### Understanding
# It Defines the correlation between parameters. By analysing this graph we can clearly understand that the profit and popularity, Votecount and popularity, revene and votecount,votecount and profit are highly dependable.
# id field doesnt have any role in the dataset.

# ## Drawing bar graph based on top 10 years of most number of movies releases.

# In[ ]:


#X axis defines the year and Y axis defines its count.
imdb_movies['release_year'].value_counts().head(10).plot.bar(figsize=(12,4))


# ### Understanding
# No of movies releasescount is rapidly increased in recent years.
# 

# ## Relationship between vote_rating and budget(using last 12 records)

# In[ ]:


# visualize the relationship between vote_rating and budget(using last 12 records)
imdb_movies.head(20).boxplot(column='budget', by='vote_average',figsize=(12,8))


# ## Visualize the relationship between revenue and budget(using last 12 records)
# 
# 

# In[ ]:


imdb_movies.head(12).boxplot(column='revenue',by='budget',figsize=(12,8))


# In[ ]:


#adding new column and showing the profit details
imdb_movies['profit']=imdb_movies.revenue-imdb_movies.budget


# ## Top 10 Highest profit movies 

# In[ ]:


top10 = imdb_movies.nlargest(10,'profit')
top10.index = top10.original_title
top10[['original_title','profit']].plot.bar(figsize=(12,4))


# ## Diectors with most number of movies.

# In[ ]:



imdb_movies.director.value_counts().head(10).plot.bar(figsize=(12,4))


# ## Popularity Vs Budget

# In[ ]:


plt.figure(figsize=(12,10))
plt.title("IMDB Popularity Vs Budget")
plt.xlabel("Popularity")
plt.ylabel("Budget")
tmp=plt.scatter(imdb_movies.popularity,imdb_movies.budget,c=imdb_movies.popularity,vmin=3,vmax=10)
plt.yticks([i*2500 for i in range(11)])
plt.colorbar(tmp,fraction=.025)
plt.show()


# In[ ]:


top20ProfitMovies = imdb_movies.nlargest(20,'profit')


# ## Profit VS Average Voting
# Below graph clearly defines the correlation between the profit and Average user voting.

# In[ ]:


plt.figure(figsize=(12,10))
plt.title("Profit VS Avg Voting of Top 20 Profitable movies")
plt.xlabel('Profit')
plt.plot(top20ProfitMovies.profit,top20ProfitMovies.vote_average)
plt.legend(['Profit VS Avg Voting'],loc='lower left')
plt.yticks(range(10))
plt.show()


# ## No of Movie released in each Genres

# In[ ]:


arrayGenres = []
for i in imdb_movies.genres:
    if type(i) == str:
        for x in i.split('|'):
            arrayGenres.append(x)
            


# In[ ]:


from collections import Counter
dicGenresWithCounts = Counter(arrayGenres)


# In[ ]:


df=pd.DataFrame.from_dict([dicGenresWithCounts.values()])
df.columns=list(dicGenresWithCounts.keys())
df.plot.bar(figsize=(12,6))
plt.title("No of movies released in each Genres")
plt.xlabel("Genres")
plt.ylabel("No of movies released")


# ## Top 10 Directors 
# 

# In[ ]:


imdb_movies.director.value_counts()[:10].plot.pie(autopct='%1.1f%%',figsize=(10,10))


# ## Top 10 Actor/Actress

# In[ ]:


arrayCast = []
for i in imdb_movies.cast:
    if type(i) == str:
        for x in i.split('|'):
            arrayCast.append(x)
dicCastWithCounts = Counter(arrayCast)
df_Cast=pd.DataFrame(data={'Cast':list(dicCastWithCounts.keys()),'Count':list(dicCastWithCounts.values())}).sort_values(by='Count',ascending=False)
df_Cast.index = df_Cast.Cast
df_Cast.Count.head(10).plot.bar(figsize=(12,6))
plt.title("Top 10 Actor/Actress")
plt.xlabel("Cast")
plt.ylabel("No of movies released")
# top10.index = top10.original_title
# top10[['original_title','profit']].plot.bar(figsize=(12,4))


# ## Top 10 Acter/Actress analysis graph

# In[ ]:


df_Cast.Count.head(10).plot.pie(autopct='%1.1f%%',figsize=(10,10))


# ## IMDB movie data classification

# #### Add new column and named it as 'IsProfitable',
# #### Lets assume profit is less than 200% of budget is considered as non profitable and classify the whole dataset.

# In[ ]:



imdb_movies['IsProfitable'] = imdb_movies.profit>(imdb_movies.budget*2)


# In[ ]:


df_imp_data = imdb_movies.drop(['id','imdb_id','homepage', 'tagline','cast','keywords','overview','genres','production_companies','release_date','budget_adj','revenue_adj','release_year','original_title','vote_count','director'], axis=1)
df_imp_data
sns.pairplot(df_imp_data, hue='IsProfitable', aspect=1.5)
plt.show()


# In[ ]:




