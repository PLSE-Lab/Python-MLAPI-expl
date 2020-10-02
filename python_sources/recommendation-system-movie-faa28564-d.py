#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/Recommendation System.csv')


# In[ ]:


df.head()


# In[ ]:


df.columns = (['user_id','item_id',"rating","timestamp"])


# In[ ]:


df.head()


# In[ ]:


movie_titles = pd.read_csv("../input/Movie_Id_Titles")
movie_titles.head()


# # The df and movie titles are to be joined based on the item ids!! basically similay to primary key and foreign key 

# In[ ]:


df = pd.merge(df,movie_titles,on='item_id')


# In[ ]:


df.sort_values('item_id').head()


# In[ ]:


df.head()


# In[ ]:


df.groupby("title")['rating'].mean().sort_values(ascending=False).head()


# In[ ]:


df.groupby("title")['rating'].count().sort_values(ascending=False).head()


# In[ ]:


ratings = pd.DataFrame(df.groupby("title")['rating'].mean())
ratings.head()


# In[ ]:


ratings["rating_counts"]= pd.DataFrame(df.groupby("title")['rating'].count())


# In[ ]:


ratings.head()


# In[ ]:


sns.distplot(ratings["rating_counts"])


# In[ ]:


# plt.figure(figsize=(20,10))
sns.distplot(ratings["rating"],bins=50)


# In[ ]:


sns.distplot( (ratings['rating']*ratings['rating_counts'] ) )


# In[ ]:


plt.figure(figsize=(10,7))
sns.jointplot(x='rating',y="rating_counts",data=ratings,alpha=.5)


# In[ ]:


df.head()


# In[ ]:


movie_mat = df.pivot_table(values='rating',index='user_id',columns = 'title')


# In[ ]:


movie_mat.head()


# In[ ]:


ratings.sort_values("rating_counts",ascending=False).head()


# In[ ]:


star_war_user_ratin = movie_mat['Star Wars (1977)']
star_war_user_ratin.value_counts()


# In[ ]:


liar_liar_user_ratin = movie_mat['Liar Liar (1997)']
liar_liar_user_ratin.head()


# In[ ]:


liar_liar_user_ratin.value_counts()


# In[ ]:


similar_to_star_wars = movie_mat.corrwith(star_war_user_ratin)
similar_to_star_wars.head()


# In[ ]:


similar_to_liarliar = movie_mat.corrwith(liar_liar_user_ratin)
similar_to_liarliar.head()


# In[ ]:


corr_star_wards = pd.DataFrame(similar_to_star_wars,columns=['Correlation'])


# In[ ]:


corr_star_wards.dropna(inplace=True)


# In[ ]:


corr_star_wards.sort_values('Correlation',ascending=False).head()


# In[ ]:


corr_star_wards= corr_star_wards.join(ratings["rating_counts"])


# In[ ]:


corr_star_wards.head()


# In[ ]:


( corr_star_wards[corr_star_wards['rating_counts']>100] ).sort_values('Correlation',ascending=False).head()


# In[ ]:





# In[ ]:


# Finding movies similar to Liar


# In[ ]:


similar_to_liarliar.head()


# In[ ]:


corr_liarliar= pd.DataFrame(similar_to_liarliar,columns=["Correlation"])
corr_liarliar.dropna(inplace=True)
corr_liarliar.head()


# In[ ]:


corr_liarliar.sort_values("Correlation",ascending=False).head()


# In[ ]:


corr_liarliar=corr_liarliar.join(ratings["rating_counts"])


# In[ ]:


corr_liarliar.head()


# In[ ]:


corr_liarliar[corr_liarliar["rating_counts"]>100].sort_values('Correlation',ascending=False)


# In[ ]:




