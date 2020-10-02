#!/usr/bin/env python
# coding: utf-8

# # Conclusion from below analysis:
# 
# ### About imdb_score:
# 1. imdb_score and num_voted_users in high positive correlation;
# 2. imdb_score and num_user_for_reviews in high positive correlation;
# 3. imdb_score and movie_facebook_likes in medium poitive correlation;
# 4. imdb_score and cast_total_facebook_likes in low positive correlation;
# 5. imdb_score and title_year in negative correlation;
# 6. imdb_score and facenumber_in_poster in negative correlation;
# 7. imdb_score seems has no correlation with budget;
# 
# 
# 
# ### About gross:
# 1. gross and num_voted_users in high positive correlation;
# 2. gross and num_user_for_reviews in high positive correlation;
# 3. gross and movie_facebook_likes in high poitive correlation;
# 4. gross and cast_total_facebook_likes in medium positive correlation;
# 5. gross and imdb_score has little correlation.
# 
# 
# 
# ### About gross and Imdb score trending:
# 
# + We have more high gross movies after 1995, but the inflation rate would be a skew factor for the data.
# + Imdb score rating number is growing, but the rating score is descending.
# 
# 
# 
# ### About Genres:
# 
# + Game Show, Reality-Tv have lower imdb_score;
# + Documentary, Biography, and File-Noir have better imdb_score.
# + Documentary has low gross;
# + Action, Adventure, Fantasy, Sci-Fi have high gross.
# 
# 
# 
# ### About imdb score with actors and director:
# 
# + So the top ten actors with the highest sum of IMDB score are:
# 
# Robert De Niro, Morgan Freeman, Johnny Depp, Matt Damon, Bruce Willis, Steve Buscemi, Brad Pitt, Bill Murray,  Denzel Washington, Liam Neeson	
#     
# + And the top ten directors with the highest sum of IMDB score are:
# 
# Steven Spielberg, Woody Allen, Martin Scorsese, Clint Eastwood, Ridley Scott, Tim Burton, Steven Soderbergh, Spike Lee, Oliver Stone, Robert Zemeckis
#     
#     
#     
# ### About gross with actors and director:
# 
# + So the top ten actors with the highest sum of gross are:
# 
# Scarlett Johansson, Robert Downey Jr., Morgan Freeman, Johnny Depp, Tom Hanks, Harrison Ford, Tom Cruise, J.K. Simmons, Will Smith, Matt Damon
# 
# + And the top ten directors with the highest sum of gross are:
# 
# Steven Spielberg, Peter Jackson, Michael Bay, Tim Burton, Sam Raimi, James Cameron, Christopher Nolan, George Lucas, Joss Whedon, Robert Zemeckis
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import scipy
print('scipy: {}'.format(scipy.__version__)) # numpy
import numpy as np
print('numpy: {}'.format(np.__version__))
# matplotlib
import matplotlib 
import matplotlib.pyplot as plt
print('matplotlib: {}'.format(matplotlib.__version__)) # pandas
import pandas as pd
print('pandas: {}'.format(pd.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
import seaborn as sns
print('seaborn: {}'.format(sns.__version__))
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


movie = pd.read_csv("../input/movie_metadata.csv")
movie.head()


# In[ ]:


movie.info()


# In[ ]:


#check how many NaN
len(movie) - movie.count()


# Summary Statistics

# In[ ]:


movie.describe()


# Univariate Plots

# In[ ]:


movie.hist(figsize=(14, 26))
plt.show()


# In[ ]:


movie.plot(kind='density', figsize=(20, 35), layout=(10,4), subplots=True, sharex=False) 
plt.show()


# In[ ]:


movie.plot(kind='box', subplots=True, figsize=(20, 35), layout=(7,6), sharex=False, sharey=False) 
plt.show()


# In[ ]:


Correlations Between Attributes


# In[ ]:


#look for correlation of number_vars and rating_vars in movie data set
movie.corr(method='pearson')


# In[ ]:


corr = movie.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
plt.figure(figsize=(16, 16))
sns.heatmap(corr, vmax=1, square=True)
sns.plt


# About imdb_score:
# 
#      - imdb_score and num_voted_users in high positive correlation;
#      - imdb_score and num_user_for_reviews in high positive correlation;
#      - imdb_score and movie_facebook_likes in medium poitive correlation;
#      - imdb_score and cast_total_facebook_likes in low positive correlation;
#      - imdb_score and title_year in negative correlation;
#      - imdb_score and facenumber_in_poster in negative correlation;
#      - imdb_score seems has no correlation with budget;
#    
# - About gross:
# 
#     - gross and num_voted_users in high positive correlation;
#     - gross and num_user_for_reviews in high positive correlation;
#     - gross and movie_facebook_likes in high poitive correlation;
#     - gross and cast_total_facebook_likes in medium positive correlation;
#     - gross and imdb_score has little correlation.

# # Find the language and country influence of imdb_score

# In[ ]:


plt.figure(figsize=(60,10))
sns.boxplot(x='language', y='imdb_score', data=movie)


# In[ ]:


plt.figure(figsize=(60,10))
sns.boxplot(x='country', y='imdb_score', data=movie)


# # Genre influence

# In[ ]:


# Clean data first
genre = movie[['gross', 'imdb_score', 'genres', 'title_year', 'facenumber_in_poster']]
#using pandas clean genres data


# In[ ]:


genre[['genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7', 'genre8']]= genre['genres'].str.split('|', expand=True)
genre


# In[ ]:


genre_list = pd.melt(genre, id_vars=['imdb_score'], value_vars=['genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7', 'genre8'],
                    var_name='variable', value_name='genres')
genre_list = genre_list.dropna()
genre_list


# In[ ]:


plt.figure(figsize=(40,10))
sns.boxplot(x='genres', y='imdb_score', data=genre_list )


# In[ ]:


genre_list2 = pd.melt(genre, id_vars=['gross'], value_vars=['genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7', 'genre8'],
                    var_name='variable', value_name='genres')
genre_list2 = genre_list2.dropna()
genre_list2


# In[ ]:


plt.figure(figsize=(40,10))
sns.violinplot(x="genres", y="gross", data=genre_list2)


#  - Conclusions from above figure: 
# 1. Documentary has low gross;    
# 2. Action, Adventure, Fantasy, Sci-Fi have high gross.

# # Actor and Director Imdb Score

# In[ ]:


actor = movie[['actor_1_name', 'actor_2_name', 'actor_3_name', 'gross', 'imdb_score']]
actor


# In[ ]:


actor_list = pd.melt(actor, id_vars=['imdb_score'], value_vars=['actor_1_name', 'actor_2_name', 'actor_3_name'],
                    var_name='variable', value_name='actor_name')
actor_list


# In[ ]:


#pivot
actor_list.pivot_table(values = ['imdb_score'], index = ['actor_name'] ,aggfunc=[np.mean], margins=True)


# So the average imdb_score per actor is 6.44

# In[ ]:


score = actor_list['imdb_score'].groupby(actor_list['actor_name']).sum()
score = score.to_frame(name= 'imdb_score')


# In[ ]:


#10 actors with highest sum imdb_score
(score.sort_values('imdb_score', ascending = False)).head(10)


# In[ ]:


director_list = movie[['director_name', 'imdb_score']]
director_score= director_list['imdb_score'].groupby(director_list['director_name']).sum()
director_score= director_score.to_frame(name= 'imdb_score')


# In[ ]:


#10 directors with highest sum imdb_score
(director_score.sort_values('imdb_score', ascending = False)).head(10)


# 1. So the top ten actors with the highest sum of IMDB score are:
#     
#     Robert De Niro, Morgan Freeman, Johnny Depp, Matt Damon, Bruce Willis, Steve Buscemi, Brad Pitt, Bill Murray,  Denzel Washington, Liam Neeson	
# 
# 2. And the top ten directors with the highest sum of IMDB score are:
#     
#     Steven Spielberg, Woody Allen, Martin Scorsese, Clint Eastwood, Ridley Scott, Tim Burton, Steven Soderbergh, Spike Lee, Oliver Stone, Robert Zemeckis

# # Actor and Director Gross Ranking

# In[ ]:


actor_list_gross = pd.melt(actor, id_vars=['gross'], value_vars=['actor_1_name', 'actor_2_name', 'actor_3_name'],
                    var_name='variable', value_name='actor_name')
actor_list_gross


# In[ ]:


#10 directors with highest sum gross
director_list_gross = movie[['director_name', 'gross']]
director_score_gross= director_list_gross['gross'].groupby(director_list_gross['director_name']).sum()
director_score_gross= director_score_gross.to_frame(name= 'gross')
(director_score_gross.sort_values('gross', ascending = False)).head(10)


# 1. So the top ten actors with the highest sum of gross are:
#     
#     Scarlett Johansson, Robert Downey Jr., Morgan Freeman, Johnny Depp, Tom Hanks, Harrison Ford, Tom Cruise, J.K. Simmons, Will Smith, Matt Damon
# 
# 2. And the top ten directors with the highest sum of gross are:
#     
#     Steven Spielberg, Peter Jackson, Michael Bay, Tim Burton, Sam Raimi, James Cameron, Christopher Nolan, George Lucas, Joss Whedon, Robert Zemeckis

# In[ ]:




