#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from fastai import *
from fastai.collab import *
from fastai.tabular import *


# In[ ]:


user,item,title = 'userId','movieId','title'
path = untar_data(URLs.ML_SAMPLE)
path


# In[ ]:


ratings = pd.read_csv(path/'ratings.csv')
print(ratings.head())
len(ratings)


# In[ ]:


data = CollabDataBunch.from_df(ratings=ratings,user_name="userId",item_name="movieId",rating_name="rating" )


# In[ ]:


y_range = [0,5.5]


# In[ ]:


learn = collab_learner(data, n_factors=50, y_range=y_range)


# In[ ]:


learn.fit_one_cycle(3, 5e-3)


# In[ ]:


import os
print(os.listdir("../input/"))


# In[ ]:


ratings = pd.read_csv("../input/u.data", delimiter='\t', header=None,
                      names=[user,item,'rating','timestamp'])
ratings.head()


# In[ ]:


movies = pd.read_csv("../input/u.item",  delimiter='|', encoding='latin-1', header=None,
                    names=[item, 'title', 'date', 'N', 'url', *[f'g{i}' for i in range(19)]])
movies.head()


# In[ ]:


len(ratings)


# In[ ]:


rating_movie = ratings.merge(movies[[item, title]])
rating_movie.head()


# In[ ]:


data = CollabDataBunch.from_df(ratings=rating_movie,user_name="userId",item_name="title",rating_name="rating" )


# In[ ]:


data.show_batch()


# In[ ]:


y_range = [0,5.5]


# In[ ]:


learn = collab_learner(data, n_factors=40, y_range=y_range, wd=1e-1)


# In[ ]:


learn.lr_find()
learn.recorder.plot(skip_end=15)


# In[ ]:


learn.fit_one_cycle(5, 5e-3)


# In[ ]:


learn.model


# In[ ]:


g = rating_movie.groupby(title)['rating'].count()
top_movies = g.sort_values(ascending=False).index.values[:1000]
top_movies[:10]


# In[ ]:


movie_bias = learn.bias(top_movies, is_item=True)
movie_bias.shape


# In[ ]:


mean_ratings = rating_movie.groupby(title)['rating'].mean()
movie_ratings = [(b, i, mean_ratings.loc[i]) for i,b in zip(top_movies,movie_bias)]


# In[ ]:


item0 = lambda o:o[0]
sorted(movie_ratings, key=item0)[:15]


# In[ ]:



sorted(movie_ratings, key=lambda o: o[0], reverse=True)[:15]


# In[ ]:


movie_w = learn.weight(top_movies, is_item=True)
movie_w.shape

movie_pca = movie_w.pca(3)
movie_pca.shape
fac0,fac1,fac2 = movie_pca.t()

torch.Size([1000, 3])

idxs = np.random.choice(len(top_movies), 50, replace=False)
idxs = list(range(50))
X = fac0[idxs]
Y = fac2[idxs]
plt.figure(figsize=(15,15))
plt.scatter(X, Y)
for i, x, y in zip(top_movies[idxs], X, Y):
    plt.text(x,y,i, color=np.random.rand(3)*0.7, fontsize=11)
plt.show()


# In[ ]:




