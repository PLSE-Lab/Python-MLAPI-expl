#!/usr/bin/env python
# coding: utf-8

# This is my workbook from [Lecture 4 ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson4-collab.ipynb) from FastAI course-v3.  
# I simply edited it with personal notes in order to better understand how it works, so credit goes to FastAI.  
# This notebook is about how to implement a *Collaborative Filter*.  
# For further info please watch the Lecture [video]( https://course.fast.ai/videos/?lesson=4).
# 
# If you are interested in other edited FastAI ipynb, you can find another one here:
# * [fastaiv1 Image Classifier](https://www.kaggle.com/gianfa/fastaiv1-image-classifier)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


from fastai.collab import *
from fastai.tabular import *


# ### Collaborative filtering example  
# collab models use data in a DataFrame of user, items, and ratings.

# In[ ]:


user,item,title = 'userId','movieId','title'


# In[ ]:


path = untar_data(URLs.ML_SAMPLE)
path


# In[ ]:


os.listdir(path)


# In[ ]:


ratings = pd.read_csv(path/'ratings.csv')
ratings.head()


# That's all we need to create and train a model:

# In[ ]:


data = CollabDataBunch.from_df(ratings, seed=42)
data


# Notice the **x** CollabList, containing  *userId* and *movieId*, being our features; the **y** contains the *rating* values, i.e. the labels.

# In[ ]:


y_range = [0,5.5]
learn = collab_learner(data, n_factors=50, y_range=y_range)
learn


# As you can see, the model resides on a [EmbeddingDotBias](https://docs.fast.ai/collab.html#EmbeddingDotBias) module, that is a simple Pytorch model (check out _source_).  
# EmbeddingDotBias wraps calls to a Pytorch [Embedding](https://pytorch.org/docs/stable/nn.html#embedding) layer, which act as a lookup table and adds a _bias_ term to the dot product result.
# 
# Here the bias term lets the layer free to better generalize the representation, catching hidden influences that are not well captured by the dataset. 
# 
# If you want to go a little more deeper about Embedding, [here](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html) you find the official Pytorch introduction to Word Embedding. 

# Now the typical fit_one_cycle()

# In[ ]:


learn.fit_one_cycle(3, 5e-3)


# ##### Let's recap a second what we have
# Our dataset contains n rows, each one 3 dimensional, which dimensions correspond to: *userId*, *movieId*, *target*.  
# The samples will be made by the *userId* and *movieId* values from dataset rows, and the labels will be the *target* values.

# In[ ]:


data.show_batch()


# If we want to predict the outcome for a specific movie, from a specific userId, we will feed the network with the userId and the movieId. That's all.

# ### Movielens 100k

# In[ ]:


folder = '../input/ml-100k/'
path = Path(folder)
os.listdir(folder)


# In[ ]:


ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,
                      names=[user,item,'rating','timestamp'])
print('ratings length: ', len(ratings))
ratings.head()


# In[ ]:


movies = pd.read_csv(path/'u.item',  delimiter='|', encoding='latin-1', header=None,
                    names=[item, 'title', 'date', 'N', 'url', *[f'g{i}' for i in range(19)]])
movies.head()


# In[ ]:


rating_movie = ratings.merge(movies[[item, title]])
rating_movie.head()


# In[ ]:


data = CollabDataBunch.from_df(rating_movie, seed=42, valid_pct=0.1, item_name=title)
data


# In[ ]:


data.show_batch()


# Check [collab_learner](https://docs.fast.ai/collab.html#collab_learner)

# In[ ]:


y_range = [0,5.5] # range of target variable
learn = collab_learner(data, n_factors=40, y_range=y_range, wd=1e-1)


# In[ ]:


learn.lr_find()
learn.recorder.plot(skip_end=15)


# In[ ]:


learn.fit_one_cycle(5, 5e-3)


# In[ ]:


#learn.save('dotprod')


# In[ ]:


learn.model


# In[ ]:


g = rating_movie.groupby(title)['rating'].count()
top_movies = g.sort_values(ascending=False).index.values[:1000]
top_movies[:10]


# Let's see how we can explore the model parameters a bit.

# ### Movie bias

# 

# In[ ]:


movie_bias = learn.bias(top_movies, is_item=True)
movie_bias.shape


# In[ ]:


mean_ratings = rating_movie.groupby(title)['rating'].mean()
movie_ratings = [(b, i, mean_ratings.loc[i]) for i,b in zip(top_movies,movie_bias)]


# In[ ]:


item0 = lambda o:o[0]


# In[ ]:


sorted(movie_ratings, key=item0)[:15]


# In[ ]:


sorted(movie_ratings, key=lambda o: o[0], reverse=True)[:15]


# ### Movie weights

# In[ ]:


movie_w = learn.weight(top_movies, is_item=True)
movie_w.shape


# In[ ]:


movie_pca = movie_w.pca(3)
movie_pca.shape


# In[ ]:


fac0,fac1,fac2 = movie_pca.t()
movie_comp = [(f, i) for f,i in zip(fac0, top_movies)]


# In[ ]:


sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]


# In[ ]:


sorted(movie_comp, key=itemgetter(0))[:10]


# In[ ]:


movie_comp = [(f, i) for f,i in zip(fac1, top_movies)]


# In[ ]:


sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]


# In[ ]:


sorted(movie_comp, key=itemgetter(0))[:10]


# In[ ]:


idxs = np.random.choice(len(top_movies), 50, replace=False)
idxs = list(range(50))
X = fac0[idxs]
Y = fac2[idxs]
plt.figure(figsize=(15,15))
plt.scatter(X, Y)
for i, x, y in zip(top_movies[idxs], X, Y):
    plt.text(x,y,i, color=np.random.rand(3)*0.7, fontsize=11)
plt.show()


# Further readings:
# * http://activisiongamescience.github.io/2016/01/11/Implicit-Recommender-Systems-Biased-Matrix-Factorization/
# * https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/2.%20Topic%20Modeling%20with%20NMF%20and%20SVD.ipynb
