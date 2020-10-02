#!/usr/bin/env python
# coding: utf-8

# 
# 
# Lets see how the social media (Facebook) influence the IMDb Ratings
# ===================================================================
# 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")


# In[ ]:


df = pd.read_csv('../input/movie_metadata.csv')
df.head()


# **There are total 28 features in the dataset:** <br><br>
#  colordirector_name<br> 
#  num_critic_for_reviews<br> 
#  duration
# <br> 
#  director_facebook_likes<br> 
#  actor_3_facebook_likes<br> 
#  actor_2_name
# <br> 
#  actor_1_facebook_likes<br> 
#  gross<br> 
#  genres<br> 
#  actor_1_name<br> 
#  movie_title
# <br> 
#  num_voted_users<br> 
#  cast_total_facebook_likes<br> 
#  actor_3_name
# <br> 
#  facenumber_in_poster<br> 
#  plot_keywords<br> 
#  movie_imdb_link
# <br> 
#  num_user_for_reviews<br> 
#  language<br> 
#  country<br> 
#  content_rating<br> 
#  budget
# <br> 
#  title_year<br> 
#  actor_2_facebook_likes<br> 
#  imdb_score<br> 
#  aspect_ratio
# <br> 
#  movie_facebook_likes

#  Lets check the number of null values in the dataset :

# In[ ]:


print(df.isnull().sum())


# **To begin with we will include three features as follows :**  <br><br>
# cast_total_facebook_likes<br>imdb_score <br>content_rating <br><br>

# In[ ]:


# content_rating has 303 null values
df = df.dropna(subset=['content_rating'])


# Lets round of the imdb_score and remove the decimal point, when we sort the 'cast_total_facebook_likes' we see that top 3 max values are  656730,303717,263584 , they huge compare to the average total likes, so we will blame them as outliers and will remove them from the data.

# In[ ]:


df['imdb_score'] = df['imdb_score'].round()
df['cast_total_facebook_likes'].sort_values(ascending=False).head(10)


# In[ ]:


df = df[(df.cast_total_facebook_likes != 656730) & (df.cast_total_facebook_likes != 303717) & (df.cast_total_facebook_likes != 263584)]


# In[ ]:


fig = plt.figure(figsize=(13, 5))
t = np.arange(0.01, 20.0, 0.01)
g = sns.lmplot('cast_total_facebook_likes', 'imdb_score',data=df,fit_reg=False,hue='content_rating',palette='muted',x_jitter=2.0,y_jitter=2.0,size=10)
g.set(xlim=(0, None))
g.set(ylim=(0, None))
x_values = [10000,30000, 50000, 70000, 90000, 110000, 130000, 150000, 170000, 190000]
plt.xticks(x_values)
plt.title('Impact of Facebook likes on IMDb score')
plt.ylabel('IMDB Rating')
plt.xlabel('Total Facebook Likes')
plt.show()

