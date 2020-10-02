#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


# In[ ]:


column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv("/kaggle/input/movielens-data/u.data",sep="\t",names=column_names)


# In[ ]:


df.head()


# In[ ]:


movie_titles = pd.read_csv("/kaggle/input/movielens-data/Movie_Id_Titles.csv")
movie_titles.head()


# In[ ]:


print(df.shape,movie_titles.shape)


# In[ ]:


df = pd.merge(df,movie_titles,on="item_id")
df.shape


# In[ ]:


df.head()


# In[ ]:


df.groupby("title").mean()["rating"].sort_values(ascending=False)


# In[ ]:


df.groupby("title").count()["rating"].sort_values(ascending=False)


# In[ ]:


ratings = pd.DataFrame(df.groupby("title").mean()["rating"])
ratings


# In[ ]:


ratings["num of ratings"] = pd.DataFrame(df.groupby("title").count()["rating"])


# In[ ]:


ratings.head()


# In[ ]:


ratings["num of ratings"].hist(bins=50)


# In[ ]:


ratings["rating"].hist(bins=50)


# In[ ]:



sns.jointplot(x="rating",y="num of ratings",data=ratings,alpha=0.5)


# In[ ]:


df.head()


# In[ ]:


moviematrix = df.pivot_table(index="user_id",columns="title",values="rating")


# In[ ]:


moviematrix.head()


# In[ ]:


ratings.sort_values("num of ratings",ascending=False).head(10)


# In[ ]:


# choosing starwars and liarliar

starwars_user_ratings = moviematrix["Star Wars (1977)"]
liarliar_user_ratings= moviematrix["Liar Liar (1997)"]


# In[ ]:


starwars_user_ratings.head()


# In[ ]:


liarliar_user_ratings.head()


# In[ ]:


#correlation of ther movieswith starwars movie

similar_to_starwars = moviematrix.corrwith(starwars_user_ratings)
similar_to_liarliar = moviematrix.corrwith(liarliar_user_ratings)


# In[ ]:


starwars_corrdf = pd.DataFrame(similar_to_starwars,columns=["Correlation"])
starwars_corrdf= starwars_corrdf.join(ratings["num of ratings"])

liarliar_corrdf = pd.DataFrame(similar_to_liarliar,columns=["Correlation"])
liarliar_corrdf=liarliar_corrdf.join(ratings["num of ratings"])

starwars_corrdf.head()


# In[ ]:


starwars_corrdf[starwars_corrdf["num of ratings"] > 50].sort_values("Correlation",ascending=False).head() #filtering out movies with very less ratings ex: 100 reviews as threshold


# In[ ]:


liarliar_corrdf[liarliar_corrdf["num of ratings"] > 65].sort_values("Correlation",ascending=False).head() #filtering out movies with very less ratings ex: 100 reviews as threshold

