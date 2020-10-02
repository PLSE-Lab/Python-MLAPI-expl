#!/usr/bin/env python
# coding: utf-8

# * This kernel show the principal features that do anime get success
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


animes = pd.read_csv('/kaggle/input/anime-recommendations-database/anime.csv')
ratings = pd.read_csv('/kaggle/input/anime-recommendations-database/rating.csv')


# In[ ]:


animes.head()


# In[ ]:


dummies = pd.get_dummies(animes['genre'].str.get_dummies(sep=','))


# In[ ]:


import networkx as nx
matrix = np.asmatrix(dummies.corr())
G = nx.from_numpy_matrix(matrix)


# In[ ]:



def create_corr_network(G, corr_direction, min_correlation):
    H = G.copy()
    for stock1, stock2, weight in G.edges(data=True):
        if corr_direction == "positive":
            if weight["weight"] <0 or weight["weight"] < min_correlation:
                H.remove_edge(stock1, stock2)
        else:
            if weight["weight"] >=0 or weight["weight"] > min_correlation:
                H.remove_edge(stock1, stock2)
                
    edges,weights = zip(*nx.get_edge_attributes(H,'weight').items())
    weights = tuple([(1+abs(x))**2 for x in weights])
    d = nx.degree(H)
    nodelist, node_sizes = zip(*d)
    positions=nx.circular_layout(H)
    
    plt.figure(figsize=(10,10), dpi=72)

    nx.draw_networkx_nodes(H,positions,node_color='#DA70D6',nodelist=nodelist,
                           node_size=tuple([x**2 for x in node_sizes]),alpha=0.8)
    
    nx.draw_networkx_labels(H, positions, font_size=8, 
                            font_family='sans-serif')
    
    if corr_direction == "positive": edge_colour = plt.cm.GnBu 
    else: edge_colour = plt.cm.PuRd
        
    nx.draw_networkx_edges(H, positions, edge_list=edges,style='solid',
                          width=weights, edge_color = weights, edge_cmap = edge_colour,
                          edge_vmin = min(weights), edge_vmax=max(weights))
    plt.axis('off')
    plt.show()


# **Show Correlation of Genres**

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(80,80)) 

sns.set(font_scale=5)

sns.heatmap(dummies.corr(), annot_kws={"size": 16}, annot=True, fmt='.1f')


# * It's to hard understand this correlation matrix

# In[ ]:


import matplotlib.pyplot as plt

corr = dummies.corr()
stocks = corr.index.values
cor_matrix = np.asmatrix(corr)
G = nx.from_numpy_matrix(cor_matrix)
G = nx.relabel_nodes(G,lambda x: stocks[x])
G.edges(data=True)

create_corr_network(G, 'positive', 0.1)


# * So now we can see better the correlation about genres, the most strong correlation is Adventure and Action

# ***Distribution Of Rating***

# In[ ]:


import seaborn as sns
sns.set(font_scale=1)
sns.kdeplot(animes['rating'], shade=True, color="r")


# * The rating distribuition over dataset is looklike normal distribuition, the most common rating is between 6 and 8

# In[ ]:


a = animes[animes['episodes']!="Unknown"]
sns.kdeplot(a['episodes'], shade=True, color="r")


# * Animes there are among 0 and 250 episodes, and exist a gap until 1500 and 1750 episodes

# In[ ]:


sns.boxplot(animes['rating'])


# * So box plot show us that not common the animes receive rating below 4 and above 8.7, this ratings can be interpreted as an anime the not very watched

# In[ ]:


import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
sns.set(font_scale=5)

rcParams['figure.figsize'] = 50, 20
know_episodes = animes[animes['episodes']!='Unknown']
know_episodes['episodes'] = know_episodes['episodes'].astype(int)

sns.scatterplot(x="episodes", y="rating",palette="Set2",data=know_episodes.sort_values('episodes'), s=100)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()


# * How that are so much animes with low number and a great variation of rationg

# In[ ]:


a.head()


# In[ ]:


animes[animes['episodes']=='Unknown']


# In[ ]:


animes.isna().sum()


# In[ ]:


animes.head()


# In[ ]:


def count_genres(df):
    amount_of_genres = []
    df['genre'] = df['genre'].astype(str)
    for genre in df['genre']:
        if genre != 'NaN':
            count = len(genre.split(','))
            amount_of_genres.append(count)
        else:
            amount_of_genres.append(-1)
    return amount_of_genres

animes['Amount Genres'] = count_genres(animes)


# In[ ]:


sns.kdeplot(animes['Amount Genres']  ,shade=True, color="g")


# In[ ]:





# In[ ]:


cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
ax = sns.countplot(x="Amount Genres", data=animes)


# * When look for genre that are great density of 1 until 7, after that the number of genre deacrese drastically

# In[ ]:


animes.head()


# In[ ]:


ax = sns.boxplot(x="Amount Genres", y="rating", data=animes)


# * Based on genre while the number of genres of the anime grows better it is classified

# In[ ]:


ax = sns.scatterplot(x="members", y="rating", data=animes)


# * Members show how much peoples keep the anime in list, so how much people watch anime it tends to be more rated, animes with low number of members has in general low rated
# 

# In[ ]:


ax = sns.countplot(x="type",data=animes)


# In[ ]:


ax = sns.boxplot(x="type", y="rating", data=animes)


# * Ona and Movie has the most regular behavior

# In[ ]:


merged = pd.concat([animes, dummies], axis=1)
merged.head()


# In[ ]:


from tqdm import tqdm
fig, ax = plt.subplots(9, 10, figsize=(130,150))
i = 0
j=0
for genre in tqdm(dummies.columns):
    genre_ = merged[merged[genre]==1]
    sns.kdeplot(list(genre_["rating"]), shade=True, color="r", ax=ax[i][j]).set_title('Genre rating ' +genre)
    if (j % 9 ==0) and (j!=0):
        i +=1 
        j =0
    j+=1
fig.tight_layout()
fig.show()


# * Shounen, Sports, shoujos, school, Game are the genres with best ratings

# # Conclusion
# 
# * Based only in metadata to for an anime to succeed need be until 250 episodes
# * need to have up to 7 genres
# * neeb be TV or movie type
# * and need have Shounen, Sports, shoujos, school or Game genre to reach a larger audience
# 
