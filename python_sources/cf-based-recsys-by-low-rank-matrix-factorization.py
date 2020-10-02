#!/usr/bin/env python
# coding: utf-8

# ## Collaborative Filtering Based Recommender Systems using Low Rank Matrix Factorization(User & Movie Embeddings) & Neural Network in Keras.

# In[ ]:





# ## [ Please star/upvote in case you like it. ]

# ## CONTENTS::->

# [ **1 ) Exploratory the Data**](#content1)

# [ **2 ) Preparing the Data**](#content2)

# [ **3 ) Matrix Factorization**](#content3)

# [ **4 ) Evaluating the Model Performance**](#content4)

# [ **5 ) Using a Neural Network**](#content5)

# In[ ]:





# <a id="content1"></a>
# ## 1 ) Exploring the Data

# ## 1.1 ) Importing Various Modules

# In[ ]:


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense , merge
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import ReduceLROnPlateau


from keras.layers.merge import dot
from keras.models import Model


# specifically for deeplearning.
from keras.layers import Dropout, Flatten,Activation,Input,Embedding
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import tensorflow as tf
import random as rn
from IPython.display import SVG
 
# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image


#TL pecific modules
from keras.applications.vgg16 import VGG16


# ## 1.2 ) Reading the CSV file

# In[ ]:


train=pd.read_csv(r'../input/movielens100k/ratings.csv')


# I am using the movie-lens-100K data. Note that this file contains the ratings given by our set of users to different movies. In all it contains total 100K ratings; to be exact 1000004.

# In[ ]:


df=train.copy()


# In[ ]:


df.head()


# ## 1.3 ) Exploring the dataset

# In[ ]:


df['userId'].unique()


# In[ ]:


len(df['userId'].unique())


# Note that in total we have 671 unique users whose userid range from 1->671.

# In[ ]:


df['movieId'].unique()


# In[ ]:


len(df['movieId'].unique())


# Similarly we have 9066 unique movies, Also note that as provided each user has voted for atleast 20 movies. We will see that the utility matrix thus created thus is quite sparse.

# #### Note that for 671 users and 9066 movies we can have a maximum of 671*9066 = 6083286 ratings. But note that we have only 100004 ratings with us. Hence the utility matrix has only about 1.6 % of the total values. Thus it can be concluded that it is quite sparse. This limits the use of some algorithms. Hence we will create embeddings for them later.

# In[ ]:


df['userId'].isnull().sum()


# In[ ]:


df['rating'].isnull().sum()


# In[ ]:


df['movieId'].isnull().sum()


# This confirms that none of the columns has any NULL or Nan value.

# In[ ]:


df['rating'].min() # minimum rating


# In[ ]:


df['rating'].max() # maximum rating


# <a id="content2"></a>
# ## 2 ) Preparing the data

# ## 2.1 ) Encoding the columns

# In[ ]:


df.userId = df.userId.astype('category').cat.codes.values
df.movieId = df.movieId.astype('category').cat.codes.values


# In[ ]:


df['userId'].value_counts(ascending=True)


# In[ ]:


df['movieId'].unique()


# ## 2.2 ) Creating the Utility Matrix

# In[ ]:


# creating utility matrix.
index=list(df['userId'].unique())
columns=list(df['movieId'].unique())
index=sorted(index)
columns=sorted(columns)
 
util_df=pd.pivot_table(data=df,values='rating',index='userId',columns='movieId')
# Nan implies that user has not rated the corressponding movie.


# 

# In[ ]:


util_df


# #### BREAKING IT DOWN--
# 
# 1) This is the utility matrix; for each of the 671 users arranged rowwise; each column shows the rating of the movie given by a particular user.
# 
# 2) Note that majority of the matrix is filled with 'Nan' which shows that majority of the movies are unrated by many users.
# 
# 3) For each movie-user pair if the entry is NOT 'Nan' the vaue indicates the rating given by user to that corressponding movie. 
# 
# 4) For now I am gonna fill the 'Nan' value with value '0'. But note that this just is just indicative, a 0 implies NO RATING and doesn't mean that user has rated 0 to that movie. It doesn't at all represent any rating.

# In[ ]:


util_df.fillna(0)


# ## 2.3 ) Creating Training and Validation Sets.

# In[ ]:


# x_train,x_test,y_train,y_test=train_test_split(df[['userId','movieId']],df[['rating']],test_size=0.20,random_state=42)
users = df.userId.unique()
movies = df.movieId.unique()

userid2idx = {o:i for i,o in enumerate(users)}
movieid2idx = {o:i for i,o in enumerate(movies)}


# In[ ]:


df['userId'] = df['userId'].apply(lambda x: userid2idx[x])
df['movieId'] = df['movieId'].apply(lambda x: movieid2idx[x])
split = np.random.rand(len(df)) < 0.8
train = df[split]
valid = df[~split]
print(train.shape , valid.shape)


# In[ ]:





# <a id="content3"></a>
# ## 3 ) Matrix Factorization

# #### Here comes the main part!!!      
# 
# 1) Now we move on to the crux of the notebook ie Matrix Factorization. In matrix facorization, we basically break a matrix into usually 2 smaller matrices each with smaller dimensions. these matrices are oftem called 'Embeddings'.  We can have variants of Matrix Factorizartion-> 'Low Rank MF' , 'Non-Negaive MF' (NMF) and so on..  
# 
# 2) Here I  have used the so called 'Low Rank Matrix Factorization'.  I have created  embeddings for both user as well as the item; movie in our case. The number of dimensions or the so called 'Latent Factors' in the embeddings is a hyperparameter to deal with in this implementation of Collaborative Filtering.                                                  

# ## 3.1 ) Creating the Embeddings ,Merging and Making the Model from Embeddings

# In[ ]:


n_movies=len(df['movieId'].unique())
n_users=len(df['userId'].unique())
n_latent_factors=64  # hyperparamter to deal with. 


# In[ ]:


user_input=Input(shape=(1,),name='user_input',dtype='int64')


# In[ ]:


user_embedding=Embedding(n_users,n_latent_factors,name='user_embedding')(user_input)
#user_embedding.shape


# In[ ]:


user_vec =Flatten(name='FlattenUsers')(user_embedding)
#user_vec.shape


# In[ ]:


movie_input=Input(shape=(1,),name='movie_input',dtype='int64')
movie_embedding=Embedding(n_movies,n_latent_factors,name='movie_embedding')(movie_input)
movie_vec=Flatten(name='FlattenMovies')(movie_embedding)
#movie_vec


# In[ ]:


sim=dot([user_vec,movie_vec],name='Simalarity-Dot-Product',axes=1)
model =keras.models.Model([user_input, movie_input],sim)
# #model.summary()
# # A summary of the model is shown below-->


# #### BREAKING IT DOWN--
# 
# 1) First we need to create embeddings for both the user as well as the item or movie. For this I have used the Embedding layer from keras.
# 
# 2) Specify the input expected to be embedded (Both in user and item embedding). The use a Embedding layer which expects the no of latent factors in the resulting embedding and also the no of users or items.
# 
# 3) Then we take the 'Dot-Product' of both the embeddings using the 'merge' layer. Note that 'dot-product' is just a measure of simalrity and we can use any other mode like 'mulitply' or 'cosine simalarity' or 'concatenate' etc...
# 
# 4) Lastly we make a Keras model from the specified details.
# 

# ## 3.2 ) Compiling the Model

# In[ ]:


model.compile(optimizer=Adam(lr=1e-4),loss='mse')


# Note that the metrics used is 'Mean squared Error'. Our aim is to minimize the mse on the training set ie over the values which the user has rated (100004 ratings).

# In[ ]:


train.shape
batch_size=128
epochs=50


# ## 3.3 ) Fitting on Training set & Validating on Validation Set.

# In[ ]:


History = model.fit([train.userId,train.movieId],train.rating, batch_size=batch_size,
                              epochs =epochs, validation_data = ([valid.userId,valid.movieId],valid.rating),
                              verbose = 1)


# <a id="content4"></a>
# ## 4 ) Evaluating the Model Performance

# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
import matplotlib.pyplot as plt
plt.plot(History.history['loss'] , 'g')
plt.plot(History.history['val_loss'] , 'b')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()


# In[ ]:





# <a id="content5"></a>
# ## 5 ) Using a Neural Network

# #### Now let us focus on the other main thing!!! Using a NN to matrix factorization.
# 
# 1) Note that this way is not much different from the previous approach.
# 
# 2) The main difference is that we have used Fully Connected layers as well as the Dropout layers and the BatchNormalization layers.
# 
# 3) The number of units and the number of layers etc.. are the hyperparametrs here as in a traditional neural network.
# 
# 

# ## 5.1 ) Creating the Embeddings

# ####  Note that I have used 50 latent factors as that seems to give reasonable performance. Furhter tuning and careful optimization can give even better results.

# In[ ]:


n_latent_factors=50
n_movies=len(df['movieId'].unique())
n_users=len(df['userId'].unique())


# In[ ]:


user_input=Input(shape=(1,),name='user_input',dtype='int64')
user_embedding=Embedding(n_users,n_latent_factors,name='user_embedding')(user_input)
user_vec=Flatten(name='FlattenUsers')(user_embedding)
user_vec=Dropout(0.40)(user_vec)


# In[ ]:


movie_input=Input(shape=(1,),name='movie_input',dtype='int64')
movie_embedding=Embedding(n_movies,n_latent_factors,name='movie_embedding')(movie_input)
movie_vec=Flatten(name='FlattenMovies')(movie_embedding)
movie_vec=Dropout(0.40)(movie_vec)


# In[ ]:


sim=dot([user_vec,movie_vec],name='Simalarity-Dot-Product',axes=1)


# ## 5.2 ) Specifying the Model architecture

# In[ ]:


nn_inp=Dense(96,activation='relu')(sim)
nn_inp=Dropout(0.4)(nn_inp)
# nn_inp=BatchNormalization()(nn_inp)
nn_inp=Dense(1,activation='relu')(nn_inp)
nn_model =keras.models.Model([user_input, movie_input],nn_inp)
nn_model.summary()


# #### Notice the summary of the model and also the architecture of the model which u can tune of course.

# ## 5.3 ) Compiling the Model

# In[ ]:


nn_model.compile(optimizer=Adam(lr=1e-3),loss='mse')


# In[ ]:


batch_size=128
epochs=20


# ## 5. 4) Fitting on Training set & Validating on Validation Set.

# In[ ]:


History = nn_model.fit([train.userId,train.movieId],train.rating, batch_size=batch_size,
                              epochs =epochs, validation_data = ([valid.userId,valid.movieId],valid.rating),
                              verbose = 1)


# #### Note that the validation loss is close to 0.84 which is quite decent. Also note that it has decrreased from 1.26 in the case of normal Matrix Factorization to this value here.

# ####  Similary playing with no of latent factors,  other parameters in the model architecture can give to even better results!!!!!

# In[ ]:





# ## THE END!!!

# ## [ Please star/upvote if u liked it. ]

# In[ ]:




