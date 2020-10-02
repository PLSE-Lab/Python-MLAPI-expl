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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.collab import *
from fastai.tabular import *


# In[ ]:


ratings = pd.read_csv('../input/ratings.csv',names=['UserId','MovieId','Rating','TimeStamp'],header=0)
ratings.columns=['userId','movieId','rating','timestamp']
ratings.head()


# In[ ]:


# Load a movie metadata dataset
movies = pd.read_csv('../input/movies.csv',low_memory=False)
movies.columns=['movieId','title','genres']
movies.head()


# In[ ]:


movies_with_names = pd.merge(ratings,movies,how='inner', on='movieId')
movies_with_names.shape


# In[ ]:


movies_with_names.head(10)


# In[ ]:


import matplotlib.pyplot as plt

data_rating = ratings.rating.value_counts().sort_index(ascending=False)


# In[ ]:


x = [5,4.5,4,3.5,3,2.5,2,1.5,1,0.5]
s=['{:.1f} %'.format(val) for val in (data_rating.values / ratings.shape[0] * 100)]
plt.figure(figsize=(8,6))
plt.bar(data_rating.index,data_rating.values)
for i in range(10):
    plt.text(x[i],data_rating.values[i],s=s[i])
plt.title('Distribution Of {} Movie Lens-Ratings'.format(ratings.shape[0]))
plt.xlabel('Rating')
plt.ylabel('Count')
plt.grid(which='minor', axis='y')
plt.show()


# In[ ]:



get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.collab import *


# In[ ]:


data = CollabDataBunch.from_df(ratings, seed=42, valid_pct=0.1, user_name='userId', item_name='movieId', rating_name='rating')
data.show_batch


# In[ ]:


ratings.rating.min(), ratings.rating.max()


# In[ ]:


MODELS_PATH = '/kaggle/working/models/'
PATH='../input/'

get_ipython().run_line_magic('mkdir', '-p {MODELS_PATH}')


# In[ ]:


learn = collab_learner(data, n_factors=40, y_range=(0.5, 5), wd=1e-1, model_dir=MODELS_PATH, path=PATH)


# In[ ]:


print(learn.summary())


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10, 1e-01)


# In[ ]:


learn.save('goodmovies-2')


# In[ ]:


# load in EmbeddingDotBias model
learn = collab_learner(data, n_factors=40, y_range=(1, 5), wd=1e-1, model_dir=MODELS_PATH, path=PATH)
learn.load('goodmovies-2');


# In[ ]:


# get top movies
g = ratings.groupby('movieId').count()
top_movies = g.sort_values('rating',ascending=False).reset_index().drop('userId',axis=1).drop('timestamp',axis=1)[:3000]
# top_movies = top_movies.astype(str)
top_movies[:10]


# In[ ]:


top_movies = top_movies.astype(int)


# In[ ]:


top_movies.dtypes


# In[ ]:


top_movies_with_names = pd.merge(top_movies,movies,how='inner',on='movieId')
top_movies_with_names.shape


# In[ ]:


top_movies_with_names.head(10)


# In[ ]:


top_movies_with_names.tail(10)


# In[ ]:


top_movies = top_movies.astype(str)


# In[ ]:


top_movies_with_name = np.array(top_movies_with_names['title'])


# In[ ]:


top_movies_idx = g.sort_values('rating',ascending=False).index.values[:3000]
top_movies_idx = top_movies_idx.astype(str)


# # Movie Bias

# In[ ]:


learn.model


# In[ ]:


data.show_batch()


# In[ ]:


movie_bias = learn.bias(top_movies_idx, is_item=True)
movie_bias.shape


# In[ ]:


mean_ratings = ratings.groupby('movieId')['rating'].mean()
movie_ratings = [(b, top_movies_with_name[i], mean_ratings.loc[int(tb)]) for i, (tb, b) in enumerate(zip(top_movies_idx, movie_bias))]


# In[ ]:


item0 = lambda o:o[0]
sorted(movie_ratings, key=item0)[:15]


# In[ ]:


sorted(movie_ratings, key=item0, reverse=True)[:15]


# In[ ]:


movie_w = learn.weight(top_movies_idx, is_item=True)
movie_w.shape


# In[ ]:


movie_pca = movie_w.pca(3)
movie_pca.shape


# In[ ]:


fac0,fac1,fac2 = movie_pca.t()
movie_comp = [(f, i) for f,i in zip(fac0, top_movies_with_name)]


# In[ ]:


sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]


# In[ ]:


sorted(movie_comp, key=itemgetter(0) )[:10]


# In[ ]:


movie_comp = [(f, i) for f,i in zip(fac1, top_movies_with_name)]
sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]


# In[ ]:


sorted(movie_comp, key=itemgetter(0))[:10]


# In[ ]:


idxs = np.random.choice(len(top_movies_with_name), 50, replace=False)
idxs = list(range(50))
X = fac0[idxs]
Y = fac2[idxs]
plt.figure(figsize=(15,15))
plt.scatter(X, Y)
for i, x, y in zip(top_movies_with_name[idxs], X, Y):
    plt.text(x,y,i, color=np.random.rand(3)*0.7, fontsize=11)
plt.show()


#  # Matrix Factorisation With Keras And Gradient Descent 
# 

# In[ ]:


# Shuffle DataFrame
df_filterd = ratings.drop('timestamp', axis=1).sample(frac=1).reset_index(drop=True)

# Testingsize
n = 80000
n1 = 20000
# Split train- & testset
df_train = df_filterd[:-n]
df_test = df_filterd[-n1:]


# In[ ]:


# Create a user-movie matrix with empty values
df_p = df_filterd.pivot_table(index='userId', columns='movieId', values='rating')
print('Shape User-Movie-Matrix:\t{}'.format(df_p.shape))
df_p.sample(3)


# In[ ]:


from sklearn.metrics import mean_squared_error
# To create deep learning models
from keras.layers import Input, Embedding, Reshape, Dot, Concatenate, Dense, Dropout
from keras.models import Model


# In[ ]:


# Create user- & movie-id mapping
user_id_mapping = {id:i for i, id in enumerate(ratings['userId'].unique())}
movie_id_mapping = {id:i for i, id in enumerate(ratings['movieId'].unique())}


# Create correctly mapped train- & testset
train_user_data = df_train['userId'].map(user_id_mapping)
train_movie_data = df_train['movieId'].map(movie_id_mapping)

test_user_data = df_test['userId'].map(user_id_mapping)
test_movie_data = df_test['movieId'].map(movie_id_mapping)


# Get input variable-sizes
users = len(user_id_mapping)
movies = len(movie_id_mapping)
embedding_size = 10


##### Create model
# Set input layers
user_id_input = Input(shape=[1], name='User')
movie_id_input = Input(shape=[1], name='Movie')

# Create embedding layers for users and movies
user_embedding = Embedding(output_dim=embedding_size, 
                           input_dim=users,
                           input_length=1, 
                           name='user_embedding')(user_id_input)
movie_embedding = Embedding(output_dim=embedding_size, 
                            input_dim=movies,
                            input_length=1, 
                            name='item_embedding')(movie_id_input)

# Reshape the embedding layers
user_vector = Reshape([embedding_size])(user_embedding)
movie_vector = Reshape([embedding_size])(movie_embedding)

# Compute dot-product of reshaped embedding layers as prediction
y = Dot(1, normalize=False)([user_vector, movie_vector])

# Setup model
model = Model(inputs=[user_id_input, movie_id_input], outputs=y)
model.compile(loss='mse', optimizer='adam')


# Fit model
model.fit([train_user_data, train_movie_data],
          df_train['rating'],
          batch_size=256, 
          epochs=4,
          validation_split=0.1,
          shuffle=True)

# Test model
y_pred = model.predict([test_user_data, test_movie_data])
y_true = df_test['rating'].values

#  Compute RMSE
rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
print('\n\nTesting Result With Keras Matrix-Factorization: {:.4f} RMSE'.format(rmse))


# In[ ]:


# Create user- & movie-id mapping
user_id_mapping = {id:i for i, id in enumerate(ratings['userId'].unique())}
movie_id_mapping = {id:i for i, id in enumerate(ratings['movieId'].unique())}


# Create correctly mapped train- & testset
train_user_data = df_train['userId'].map(user_id_mapping)
train_movie_data = df_train['movieId'].map(movie_id_mapping)

test_user_data = df_test['userId'].map(user_id_mapping)
test_movie_data = df_test['movieId'].map(movie_id_mapping)


# Get input variable-sizes
users = len(user_id_mapping)
movies = len(movie_id_mapping)
embedding_size = 10


##### Create model
# Set input layers
user_id_input = Input(shape=[1], name='User')
movie_id_input = Input(shape=[1], name='Movie')

# Create embedding layers for users and movies
user_embedding = Embedding(output_dim=embedding_size, 
                           input_dim=users,
                           input_length=1, 
                           name='user_embedding')(user_id_input)
movie_embedding = Embedding(output_dim=embedding_size, 
                            input_dim=movies,
                            input_length=1, 
                            name='item_embedding')(movie_id_input)

# Reshape the embedding layers
user_vector = Reshape([embedding_size])(user_embedding)
movie_vector = Reshape([embedding_size])(movie_embedding)

# Compute dot-product of reshaped embedding layers as prediction
y = Dot(1, normalize=False)([user_vector, movie_vector])

# Setup model
model = Model(inputs=[user_id_input, movie_id_input], outputs=y)
model.compile(loss='mse', optimizer='adam')


# Fit model
model.fit([train_user_data, train_movie_data],
          df_train['rating'],
          batch_size=256, 
          epochs=10,
          validation_split=0.1,
          shuffle=True)

# Test model
y_pred = model.predict([test_user_data, test_movie_data])
y_true = df_test['rating'].values

#  Compute RMSE
rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
print('\n\nTesting Result With Keras Matrix-Factorization: {:.4f} RMSE'.format(rmse))


# 
