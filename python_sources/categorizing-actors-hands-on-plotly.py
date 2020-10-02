#!/usr/bin/env python
# coding: utf-8

# ____
# # **Categorizing actors**
# *Fabien Daniel (August 2017)*
# ___

# This notebook is largely inspired from [Tianyi Wang's notebook](https://www.kaggle.com/tianyiwang/neighborhood-interaction-with-network-graph). This is my first use of **plotly** and I will try to get some insight on the habits of actors and the way they are perceived by spectators. I will not discuss the content of the current dataframe since this was done in many other notebooks (as, picking one up aleatory, [this one](https://www.kaggle.com/fabiendaniel/film-recommendation-engine/)).
# I will not either try to clean it and simply use it as it comes.

# First, I load the packages and the dataframe:

# In[ ]:


import matplotlib.pyplot as plt
import plotly.offline as pyo
pyo.init_notebook_mode()
from plotly.graph_objs import *
import pandas as pd
#_______________________________________________
df = pd.read_csv("../input/movie_metadata.csv")


# Now, I create the list of the movies genres:

# In[ ]:


liste_genres = set()
for s in df['genres'].str.split('|'):
    liste_genres = set().union(s, liste_genres)
liste_genres = list(liste_genres)


# I plan to determine the main genre of the actors contained in the *'actor_1_name'*  variable. To proceed with that, I perform some one hot encoding:

# In[ ]:


df_reduced = df[['actor_1_name', 'imdb_score', 'title_year']].reset_index(drop = True)
for genre in liste_genres:
    df_reduced[genre] = df['genres'].str.contains(genre).apply(lambda x:1 if x else 0)
df_reduced[:5]


# and I group according to every actors, taking the mean of all the other variables. Then I check which genre's column takes the highest value and assign the corresponding genre as the actor's favorite genre: 

# In[ ]:


df_actors = df_reduced.groupby('actor_1_name').mean()
df_actors.loc[:, 'favored_genre'] = df_actors[liste_genres].idxmax(axis = 1)
df_actors.drop(liste_genres, axis = 1, inplace = True)
df_actors = df_actors.reset_index()
df_actors[:10]


# At this point, the dataframe contains a list of actors and for each of them, we have a mean IMDB score, its mean year of activity and his favored acting style.
# 
# Then, I create a mask to account only for the actors that played in more than 5 films:

# In[ ]:


df_appearance = df_reduced[['actor_1_name', 'title_year']].groupby('actor_1_name').count()
df_appearance = df_appearance.reset_index(drop = True)
selection = df_appearance['title_year'] > 4
selection = selection.reset_index(drop = True)
most_prolific = df_actors[selection]


# Finally, I look at the percentage of films of each genre to further choose the genres I want to look at:

# In[ ]:


plt.rc('font', weight='bold')
f, ax = plt.subplots(figsize=(6, 6))
genre_count = []
for genre in liste_genres:
    genre_count.append([genre, df_reduced[genre].values.sum()])
genre_count.sort(key = lambda x:x[1], reverse = True)
labels, sizes = zip(*genre_count)
labels_selected = [n if v > sum(sizes) * 0.01 else '' for n, v in genre_count]
ax.pie(sizes, labels=labels_selected,
       autopct = lambda x:'{:2.1f}%'.format(x) if x > 1 else '',
       shadow=False, startangle=0)
ax.axis('equal')
plt.tight_layout()


# And now, the **magic of plotly** happens:

# In[ ]:


reduced_genre_list = labels[:12]
trace=[]
for genre in reduced_genre_list:
    trace.append({'type':'scatter',
                  'mode':'markers',
                  'y':most_prolific.loc[most_prolific['favored_genre']==genre,'imdb_score'],
                  'x':most_prolific.loc[most_prolific['favored_genre']==genre,'title_year'],
                  'name':genre,
                  'text': most_prolific.loc[most_prolific['favored_genre']==genre,'actor_1_name'],
                  'marker':{'size':10,'opacity':0.7,
                            'line':{'width':1.25,'color':'black'}}})
layout={'title':'Actors favored genres',
       'xaxis':{'title':'mean year of activity'},
       'yaxis':{'title':'mean IMDB score'}}
fig=Figure(data=trace,layout=layout)
pyo.iplot(fig)

