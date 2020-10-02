#!/usr/bin/env python
# coding: utf-8

# During the last few decades, with the rise of Youtube, Amazon, Netflix and many other such web services, recommender systems have taken more and more place in our lives. From e-commerce (suggest to buyers articles that could interest them) to online advertisement (suggest to users the right contents, matching their preferences), recommender systems are today unavoidable in our daily online journeys.
# In a very general way, recommender systems are algorithms aimed at suggesting relevant items to users (items being movies to watch, text to read, products to buy or anything else depending on industries).

# ![image](https://miro.medium.com/max/2000/1*m_Z6Da5FZ62KN2yH-x_GOQ@2x.png) 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


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



from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

import warnings; warnings.simplefilter('ignore')


# In[ ]:


df = pd.read_csv('/kaggle/input/news_articles.csv')


# In[ ]:


df.head(10)


# In[ ]:


#total articles
len(df['Article_Id'])


# In[ ]:


#unique aticles
list_of_articleid = []
q = df['Article_Id'].unique()
list_of_articleid = list_of_articleid.append(q)
list_of_articleid


# # Generating data for Number of users per Article

# In[ ]:


import scipy
import random
from scipy import stats


# In[ ]:


random.seed(15)

user_session = stats.geom.rvs(size=4831,  # Generate geometric data
                                  p=0.3)       # With success prob 0.5


# In[ ]:


user_session.size,user_session.max(),user_session.min()


# In[ ]:


user_session[:10],sum(user_session)


# In[ ]:



count_dict = {x : list(user_session).count(x) for x in user_session}
count_dict


# In[ ]:


#depicts number of users per number of sessions
    
bins = np.arange(0, 10, 1) # fixed bin size

plt.xlim([min(user_session)-1, max(user_session) +1])

plt.hist(user_session, bins=bins, alpha=0.5)
plt.title("Count of Number of users per session")
plt.xlabel('Number of sessions (bin size = 1)')
plt.ylabel('count')

plt.show()


# In[ ]:


import numpy as np

user_Id = range(1,4831)


# In[ ]:


userId_session = list(zip(user_Id,[10*i for i in user_session]))


# In[ ]:


type(userId_session), userId_session[:5]


# In[ ]:


#Calculating total number of articles served in a day in all sessions (may be clicked or not)

sum1 = 0
for i in range(len(userId_session)):
    
    sum1 += userId_session[i][1]
    
sum1


# In[ ]:


UserIDs = []

for i in range(len(userId_session)):
    
    for j in range(userId_session[i][1]):
        UserIDs.append(userId_session[i][0])


# In[ ]:


len(UserIDs)   #matches with sum1 above


# In[ ]:


UserIDs[:20]   # UserIds generated for all sessions the user opens


# In[ ]:


session_list = list(user_session)
session_list[:10]


# In[ ]:


session_Id =[]

for i in session_list:
    
    for j in range(1,i+1):
#         print j
        session_Id.append([j for i in range(10)])


# In[ ]:


session_Id = np.array(session_Id).flatten()


# In[ ]:



session_Id.shape


# In[ ]:


User_session = list(zip(UserIDs,session_Id ))


# In[ ]:



len(User_session),type(User_session)


# In[ ]:


import pandas as pd

df = pd.DataFrame(User_session, columns=['UserId', 'SessionId'])


# In[ ]:


df.tail(20)


# In[ ]:


Article_Id = list(range(4831))


#  totla article served in one day / no of unique articles = (161730/4831)

# In[ ]:


type(Article_Id)


# In[ ]:


161730/4831


# In[ ]:


Article_Id = Article_Id*int(161730/4831)  


# In[ ]:


len(Article_Id)


# to make a square matrix 

# In[ ]:


import random
for x in range(len(User_session)-len(Article_Id)):
    Article_Id.append(random.randint(1,4831))


# In[ ]:


len(Article_Id)


# > Now you can see length of user session and article_id is same

# In[ ]:


from random import shuffle
shuffle(Article_Id)


# In[ ]:


c = len(df['UserId'])


# In[ ]:


Article_Id = Article_Id[:c]


# In[ ]:


df['ArticleId_served'] = Article_Id


# In[ ]:


df.tail()


# In[ ]:


len(df['UserId'].unique())


# # creating rating 

# In[ ]:


df


# In[ ]:


p = len(df['UserId'])


# In[ ]:


import random
numLow = 1 
numHigh = 6
x = []
for i in range (0,p):
    m = random.sample(range(numLow, numHigh), 1)
    x.append(m)


# In[ ]:


x[:3]


# In[ ]:


flat_list = []
for sublist in x:
    for item in sublist:
        flat_list.append(item)


# In[ ]:


len(flat_list)


# In[ ]:


df.head()


# In[ ]:


df['rating'] = flat_list


# In[ ]:


len(df['rating'])


# In[ ]:


df.head()


# In[ ]:


# df.to_csv('file1.csv') 
# saving the dataframe 
df.to_csv('file3.csv', index=False)


#  # Creating the Utility Matrix

# In[ ]:


index=list(df['UserId'].unique())
columns=list(df['ArticleId_served'].unique())
index=sorted(index)
columns=sorted(columns)
 
util_df=pd.pivot_table(data=df,values='rating',index='UserId',columns='ArticleId_served')


# >  Nan implies that user has not rated the corressponding Article.

# In[ ]:


util_df


# UNDERSTANDING--
# 
# 1) This is the utility matrix; for each of the 4830 users arranged rowwise; each column shows the rating of the article given by a particular user.
# 
# 2) Note that majority of the matrix is filled with 'Nan' which shows that majority of the articles are unrated by many users.
# 
# 3) For each article-user pair if the entry is NOT 'Nan' the vaue indicates the rating given by user to that corressponding article.
# 
# 4) For now I am gonna fill the 'Nan' value with value '0'. But note that this just is just indicative, a **0 implies NO RATING** and doesn't mean that user has rated 0 to that article. It doesn't at all represent any rating.
# 
# RATING SCALE IS [1 2 3 4 5]

# In[ ]:


util_df.fillna(0)


# ## Creating Training and Validation Sets.

# In[ ]:


# x_train,x_test,y_train,y_test=train_test_split(df[['UserId','ArticleId_served']],df[['rating']],test_size=0.20,random_state=42)
users = df.UserId.unique()
movies = df.ArticleId_served.unique()

userid2idx = {o:i for i,o in enumerate(users)}
movieid2idx = {o:i for i,o in enumerate(movies)}


# In[ ]:


users


# In[ ]:


df['ArticleId_served'].head(70)


# In[ ]:


df['UserId'] = df['UserId'].apply(lambda x: userid2idx[x])
df['ArticleId_served'] = df['ArticleId_served'].apply(lambda x: movieid2idx[x])
split = np.random.rand(len(df)) < 0.8
train = df[split]
valid = df[~split]
print(train.shape , valid.shape)


# In[ ]:


df['ArticleId_served'].head(70)


# # Matrix Factorization
# 
# Here comes the main part!!!
# 
# 1) Now we move on to the crux of the notebook ie Matrix Factorization. In matrix facorization, we basically break a matrix into usually 2 smaller matrices each with smaller dimensions. these matrices are oftem called 'Embeddings'. We can have variants of Matrix Factorizartion-> 'Low Rank MF' , 'Non-Negaive MF' (NMF) and so on..
# 
# 2) Here I have used the so called 'Low Rank Matrix Factorization'. I have created embeddings for both user as well as the item; articles in our case. The number of dimensions or the so called 'Latent Factors' in the embeddings is a hyperparameter to deal with in this implementation of Collaborative Filtering.

# ### Creating the Embeddings ,Merging and Making the Model from Embeddings

# In[ ]:


n_article=len(df['ArticleId_served'].unique())
n_users=len(df['UserId'].unique())
n_latent_factors=64  # hyperparamter to deal with. 


# #### Input Object 
# 
# Input() is used to instantiate a Keras tensor

# In[ ]:


user_input=Input(shape=(1,),name='user_input',dtype='int64')


# In[ ]:


user_input.shape


# #### Embedding layer
# Turns positive integers (indexes) into dense vectors of fixed size.
# 
# e.g. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
# 
# **Input shape**
# 
# 2D tensor with shape: (batch_size, input_length).
# 
# **Output shape**
# 
# 3D tensor with shape: (batch_size, input_length, output_dim).

# In[ ]:


# tf.keras.layers.Embedding(
#      input_dim,
#      output_dim,
#      embeddings_initializer="uniform",
#      embeddings_regularizer=None,
#      activity_regularizer=None,
#      embeddings_constraint=None,
#      mask_zero=False,
#      input_length=None,
#      **kwargs
#  )


# In[ ]:


user_embedding=Embedding(n_users,n_latent_factors,name='user_embedding')(user_input)
user_embedding.shape


# #### Flatten layer
# Flattens the input. Does not affect the batch size.
# 
# Note: If inputs are shaped (batch,) without a feature axis, then flattening adds an extra channel dimension and output shape is (batch, 1).

# In[ ]:


user_vec =Flatten(name='FlattenUsers')(user_embedding)
user_vec.shape


# In[ ]:


article_input=Input(shape=(1,),name='article_input',dtype='int64')
article_embedding=Embedding(n_article,n_latent_factors,name='article_embedding')(article_input)
article_vec=Flatten(name='FlattenArticles')(article_embedding)
# article_vec


# In[ ]:


article_vec


# #### Dot Layer
# Layer that computes a dot product between samples in two tensors.
# 
# E.g. if applied to a list of two tensors a and b of shape (batch_size, n), the output will be a tensor of shape (batch_size, 1) where each entry i will be the dot product between a[i] and b[i].

# In[ ]:


sim=dot([user_vec,article_vec],name='Simalarity-Dot-Product',axes=1)
model =keras.models.Model([user_input, article_input],sim)
model.summary()


# UNDERSTANDING--
# 
# 1) First we need to create embeddings for both the user as well as the item or article. For this I have used the Embedding layer from keras.
# 
# 2) Specify the input expected to be embedded (Both in user and item embedding). The use a Embedding layer which expects the no of latent factors in the resulting embedding and also the no of users or items.
# 
# 3) Then we take the 'Dot-Product' of both the embeddings using the 'merge' layer. Note that 'dot-product' is just a measure of simalrity and we can use any other mode like 'mulitply' or 'cosine simalarity' or 'concatenate' etc...
# 
# 4) Lastly we make a Keras model from the specified details.
# 
# 

# ### Compiling the Model

# #### *compile* method 
# 
# Configures the model for training.
# 
# 

# In[ ]:


# Model.compile(
#     optimizer="rmsprop",
#     loss=None,
#     metrics=None,
#     loss_weights=None,
#     weighted_metrics=None,
#     run_eagerly=None,
#     **kwargs
# )


# In[ ]:


model.compile(optimizer=Adam(lr=1e-4),loss='mse')


# In[ ]:


train.shape


# In[ ]:


train.shape
batch_size=128
epochs=50


# # Fitting on Training set & Validating on Validation Set.

# #### fit method
# Trains the model for a fixed number of epochs (iterations on a dataset).
# 
# ***Returns***
# 
# A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).

# In[ ]:



# Model.fit(
#     x=None,
#     y=None,
#     batch_size=None,
#     epochs=1,
#     verbose=1,
#     callbacks=None,
#     validation_split=0.0,
#     validation_data=None,
#     shuffle=True,
#     class_weight=None,
#     sample_weight=None,
#     initial_epoch=0,
#     steps_per_epoch=None,
#     validation_steps=None,
#     validation_batch_size=None,
#     validation_freq=1,
#     max_queue_size=10,
#     workers=1,
#     use_multiprocessing=False,
# )


# In[ ]:


History = model.fit([train.UserId,train.ArticleId_served],train.rating, batch_size=batch_size,
                              epochs =epochs, validation_data = ([valid.UserId,valid.ArticleId_served],valid.rating),
                              verbose = 1)


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


# [link to part2 ](https://www.kaggle.com/bavalpreet26/recommender-system-part2)
