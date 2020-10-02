#!/usr/bin/env python
# coding: utf-8

# Enjoy the kernel. It's based on the fastai course. 
# For non-datascientist the second part of the analysis, scroll down a bit, starting at the 'best and the worst games' is probably nicest to checkout.
# 
# I've also build an app around this to games that are similar but better, see
# [over here](https://bgg.onrender.com/)
# 
# Comments of course very much welcome

# # Collaborative Filtering using Fast.ai library

# In[ ]:


import pandas as pd
import pickle
import numpy as np
from fastai.collab import *
from pprint import pprint
import matplotlib.pyplot as plt
import umap
from scipy import stats
from sklearn.neighbors import NearestNeighbors
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# the original csv from https://raw.githubusercontent.com/beefsack/bgg-ranking-historicals/master/
# The column ID is used in API calls to retrieve the game reviews
games = pd.read_csv('../input/2019-05-02.csv')
games.describe()
games.sort_values('Users rated',ascending=False,inplace=True)
games.rename(index=str, columns={"Bayes average": "Geekscore",'Name':'name'}, inplace=True)


# In[ ]:


# load the file I composed with all the reviews
reviews = pd.read_csv('../input/bgg-13m-reviews.csv',index_col=0)
print(len(reviews))
reviews.head()


# In[ ]:


games_by_all_users = reviews.groupby('name')['rating'].agg(['mean','count']).sort_values('mean',ascending=False)
games_by_all_users['rank']=games_by_all_users.reset_index().index+1
print(len(games_by_all_users))


# In[ ]:


data = CollabDataBunch.from_df(reviews, user_name='user',item_name='name',rating_name='rating',bs=100000, seed = 42)
data.show_batch()


# In[ ]:


learner = collab_learner(data, n_factors=50, y_range=(2.,10))


# In[ ]:


lr_find(learner)
learner.recorder.plot()


# In[ ]:


learner.fit_one_cycle(3, 1e-2, wd=0.15)


# In[ ]:


#learner.save('5cycles7e-2-bs100000factors50yrange0-105')
#learner.save('4cycles3e-2-bs100000factors50yrange0-105wd03')
#learner.save('4cycles2e-2-bs100000factors20yrange1-10wd01')
#learner.save('3cycles1e-2-bs100000factors50yrange2-10wd005')
#learner.load('3cycles1e-2-bs100000factors50yrange2-10wd005')
learner.model


# In[ ]:


learner.recorder.plot_losses()


# # The worst and the best games according to the model
# 
# We are only looking at the most popular games with >5000 reviews. This way the results are more recognizable.

# In[ ]:


mean_ratings = reviews.groupby('name')['rating'].mean().round(2)
top_games = games_by_all_users[games_by_all_users['count']>5000].sort_values('mean',ascending=False).index
print(len(top_games))
game_bias = learner.bias(top_games, is_item=True)
game_bias.shapemean_ratings = reviews.groupby('name')['rating'].mean()
game_ratings = [(b, i, mean_ratings.loc[i]) for i,b in zip(top_games,game_bias)]
item0 = lambda o:o[0]


# In[ ]:


sorted(game_ratings, key=item0)[:10]


# The worst games according to the model... Don't you hate tic-tac-toe? :)
# The scores to the right are the BGG averages

# In[ ]:


sorted(game_ratings, key=lambda o: o[0], reverse=True)[:15]


# The best games according to the model. Model score has a range between -0.4 and +0.4 (most scores are between 0 and 0.3). This is a score that's 'unbiased' for user preferences, such as users giving games a higher score just because they are fan of the genre.
# 
# Some surprises here, such as Dune, HeroQuest and GO that all have a much lower rating than their peers on BGG.

# # The most important dimensions that set games apart from eachother
# Now we explore the latent dimensions in the data. With PCA we reduce the variation to 3 dimensions. We try to understand what the computer has found out within the 3 dimensions by analyzing the games that score extreme on the dimensions.

# In[ ]:


game_weights = learner.weight(top_games, is_item=True)
game_weights.shape


# In[ ]:


game_pca = game_weights.pca(3)
game_pca.shape


# In[ ]:


fac0,fac1,fac2 = game_pca.t()
game_comp = [(f, i) for f,i in zip(fac0, top_games)]
print('highest on this dimension')
pprint(sorted(game_comp, key=itemgetter(0), reverse=True)[:10]) 
print('lowest on this dimension')
pprint(sorted(game_comp, key=itemgetter(0), reverse=False)[:10]) 


# This dimension seems to be about simple/old games vs complex/new games

# In[ ]:


game_comp = [(f, i) for f,i in zip(fac1, top_games)]
print('highest on this dimension')
pprint(sorted(game_comp, key=itemgetter(0), reverse=True)[:10])
print('lowest on this dimension')
pprint(sorted(game_comp, key=itemgetter(0), reverse=False)[:10])


# This dimension seems to be about friendly worker placement games vs more hardcore themed games

# In[ ]:


game_comp = [(f, i) for f,i in zip(fac2, top_games)]
print('highest on this dimension')
pprint(sorted(game_comp, key=itemgetter(0), reverse=True)[:10])
print('lowest on this dimension')
pprint(sorted(game_comp, key=itemgetter(0), reverse=False)[:10])


# This dimension seems to be about deep strategy games vs more cooperative or light games.
# 
# Now we'll plot the top 50 games on the first two dimensions

# In[ ]:


idxs = np.random.choice(len(top_games), 50, replace=False)
idxs = list(range(50))
X = fac0[idxs]
Y = fac1[idxs]
plt.figure(figsize=(15,15))
plt.scatter(X, Y)
for i, x, y in zip(top_games[idxs], X, Y):
    plt.text(x,y,i, color=np.random.rand(3)*0.7, fontsize=11)
plt.show()


# Gloomhaven is apparently a very different game from Puerto Rico :)

# # Find similar games
# Now we can also take the model in all its glory and see which games are most similar using simple nearest neighbors. The work below is also detailed in the app [over here](https://bgg.onrender.com/)

# In[ ]:


CUTOFF=2000
top_games = games_by_all_users[games_by_all_users['count']>CUTOFF].sort_values('mean',ascending=False).reset_index()
number_of_games = len(top_games)
print(number_of_games)
game_weights = learner.weight(top_games['name'], is_item=True)
game_bias = learner.bias(top_games['name'], is_item=True)
npweights = game_weights.numpy()
top_games['model_score']=game_bias.numpy()
top_games['weights_sum']=np.sum(np.abs(npweights),axis=1)

nn = NearestNeighbors(n_neighbors=number_of_games)
fitnn = nn.fit(npweights)


# In[ ]:


res = top_games[top_games['name']=='Chess']
if len(res)==1:
    distances,indices = fitnn.kneighbors([npweights[res.index[0]]])
else:
    print(res.head())
top_games.iloc[indices[0][:10]].sort_values('model_score',ascending=False)


# Bridge has a mean score of 7.46, 2735 reviews, a score by the model of 0.32
# Bridge and Go are pretty similar to Chess, and apparently even better games according to the model. Lets give it another go

# In[ ]:


res = top_games[top_games['name']=='Catan']
if len(res)==1:
    distances,indices = fitnn.kneighbors([npweights[res.index[0]]])
else:
    print(res.head())
top_games.iloc[indices[0][:10]].sort_values('model_score',ascending=False)


# These games are pretty similar to Catan right! Ok one for the road

# In[ ]:


res = top_games[top_games['name']=='Agricola']
if len(res)==1:
    distances,indices = fitnn.kneighbors([npweights[res.index[0]]])
else:
    print(res.head())
top_games.iloc[indices[0][:10]].sort_values('model_score',ascending=False)


# Well, Agricola is one of my favorite games, but apparently I should play more Puerto Rico. To me it shows that tastes are personal, since I like Agricola much more.
# 
# Well that's it, hope you liked it. Comments welcome!

# In[ ]:




