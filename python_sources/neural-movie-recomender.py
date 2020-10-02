#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/ml-100k/ml-100k"))


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import warnings
#warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

#Import dataset and create a new dataframe
data=pd.read_csv("../input/ml-100k/ml-100k/u.data",sep="\t",names="user_id,item_id,rating,timestamp".split(","))

#First five records of the daatframe
data.head()

#Check which of the records of the dataframe are null.
#data.isnull()

# Any results you write to the current directory are saved as output.


# In[ ]:


#Print total number of users
len(data.user_id.unique()),len(data.item_id.unique())
#Print total number of movies



# In[ ]:


#Assign a unique number between (0,No. of users) for each user.Do the same for movies.
data.user_id = data.user_id.astype('category').cat.codes.values
data.item_id = data.item_id.astype('category').cat.codes.values
data.head()


# In[ ]:


X = data[['user_id', 'item_id']].values
y = data['rating'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size=0.2,random_state=52)

X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]


# In[ ]:


min_rating = min(data['rating'])
max_rating = max(data['rating'])


# In[ ]:


from keras.models import Model
from keras.layers import Input, Reshape, Dot   
from keras.regularizers import l2
import keras
from IPython.display import SVG
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
from keras.layers.embeddings import Embedding
from keras.layers import Add, Activation, Lambda
class EmbeddingLayer:
    def __init__(self, n_items, n_factors):
        self.n_items = n_items
        self.n_factors = n_factors

    def __call__(self, x):
        x = Embedding(self.n_items, self.n_factors, embeddings_initializer='he_normal',
                  embeddings_regularizer=l2(1e-6))(x)
        x = Reshape((self.n_factors,))(x)
        return x


n_users, n_movies = len(data.user_id.unique()), len(data.item_id.unique())
n_factors = 3
user = Input(shape=(1,))
u = Embedding(n_users, n_factors, embeddings_initializer='he_normal',
              embeddings_regularizer=l2(1e-6))(user)
u = Reshape((n_factors,))(u)
ub = EmbeddingLayer(n_users, 1)(user)

movie = Input(shape=(1,))
m = Embedding(n_movies, n_factors, embeddings_initializer='he_normal',
              embeddings_regularizer=l2(1e-6))(movie)
m = Reshape((n_factors,))(m)
mb = EmbeddingLayer(n_users, 1)(movie)
   
x = Dot(axes=1)([u, m])
x = Add()([x, ub, mb])
x = Activation('sigmoid')(x)
x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)

model = Model(inputs=[user, movie], outputs=x)
opt = Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=opt)
model.summary()


# In[ ]:


model.fit(x=X_train_array, y=y_train, batch_size=64, epochs=5,verbose=1, validation_data=(X_test_array, y_test))

