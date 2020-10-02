#!/usr/bin/env python
# coding: utf-8

# ## <center> JantaHack : Recommendation Systems</center>
# _______________

# <center><img src="https://datahack-prod.s3.ap-south-1.amazonaws.com/__sized__/contest_cover/cover_81npx51-thumbnail-1200x1200.png"/></center>

# ### Problem Statement

# > - Your client is a fast-growing mobile platform, for hosting coding challenges. They have a unique business model, where they crowdsource problems from various creators(authors). These authors create the problem and release it on the client's platform. The users then select the challenges they want to solve. The authors make money based on the level of difficulty of their problems and how many users take up their challenge.
# 
#  
# > 
# > - The client, on the other hand makes money when the users can find challenges of their interest and continue to stay on the platform. Till date, the client has relied on its domain expertise, user interface and experience with user behaviour to suggest the problems a user might be interested in. You have now been appointed as the data scientist who needs to come up with the algorithm to keep the users engaged on the platform.
# 
# ## What to recommend
# 
# > - The client has provided you with history of last 10 challenges the user has solved, and you need to predict which might be the next 3 challenges the user might be interested to solve. Apply your data science skills to help the client make a big mark in their user engagements/revenue.

# ## Data Description
# for data desciption, there are three files:
# 
# - ```train.csv```: It contains the set of 13 challenges that were attempted by the same user in a sequence.
# - ```challenge_data.csv```: Contains attributes related to each challenge
# - ```test.csv```: Contains the first 10 challenges solved by a new user set (not in train) in the test set. We need to predict
# 

# ![](https://raw.githubusercontent.com/AIVenture0/JanataHack-Recommendation-Systems/master/data_description1.jpg)

# ![](https://raw.githubusercontent.com/AIVenture0/JanataHack-Recommendation-Systems/master/data_description2.jpg)

# **Credit**: This version of notebook is highly inspired from ybabakhin works, For more detail check his github repo
# 
# - https://github.com/ybabakhin/mckinsey/blob/master/Mckinsey.py

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


# To print multiple output in a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# In[ ]:


os.getcwd()


# In[ ]:


# Load Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense,Embedding,LSTM,Dropout,Bidirectional,Input,BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint


# ## Reading the train and test data

# In[ ]:


train = pd.read_csv("../input/av-recommendation-systems/train_mddNHeX/train.csv")
test = pd.read_csv("../input/av-recommendation-systems/test_HLxMpl7/test.csv")

train.head()
test.head()


# In[ ]:


train[train.challenge_sequence > 10]


# In[ ]:


# Create labels
label = train[train.challenge_sequence > 10][['user_id','challenge']]
label.rename(columns={'challenge':'label'},inplace=True)


# In[ ]:


label.head()


# In[ ]:


# Treat the sequence of challenges as text
df = train[train.challenge_sequence <= 10].groupby('user_id').challenge.aggregate(lambda x: ' '.join(x)).reset_index()
df.head()


# In[ ]:


# Merge Labels
df = df.merge(label)
df.head()
df.shape


# In[ ]:



# Validation split for early stopping
df_train, df_validation = train_test_split(df.sample(frac=1,random_state=123), test_size=0.05, random_state=123)


# In[ ]:


df_train.head()
df_validation.head()

df_train.shape
df_validation.shape


# In[ ]:


# Load all the challenges
challenges = pd.read_csv('../input/av-recommendation-systems/train_mddNHeX/challenge_data.csv')

challenges.head()


# In[ ]:


len(challenges.challenge_ID.unique())


# In[ ]:


len(df_train.label.unique())


# In[ ]:


df_train.head()


# In[ ]:


# Encode challenges
encoder = LabelEncoder()
encoder.fit(challenges['challenge_ID'])

df_train['brand_id_encoded'] = encoder.transform(df_train.label)
df_validation['brand_id_encoded'] = encoder.transform(df_validation.label)

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train['challenge'])


# In[ ]:


len(tokenizer.word_index)


# In[ ]:


# Constants
NB_WORDS = len(tokenizer.word_index)+1
MAX_SEQUENCE_LENGTH = 10
N_CATEGORIES = challenges.shape[0]

# Create sequences
sequences_train = tokenizer.texts_to_sequences(df_train['challenge'])
sequences_validation = tokenizer.texts_to_sequences(df_validation['challenge'])

# Pad sequences
x_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
x_validation = pad_sequences(sequences_validation, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

# Set Labels
y_train = df_train['brand_id_encoded'].values
y_validation= df_validation['brand_id_encoded'].values


# In[ ]:


y_train
y_validation


# In[ ]:


# NN architecture
def get_model(path='',lr=0.001):
    adam = Adam(lr=lr)
    inp = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    x = Embedding(NB_WORDS,256)(inp)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(256, dropout=0.1, recurrent_dropout=0.1))(x)
    x = Dropout(0.4)(x)
    x = Dense(N_CATEGORIES, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    if path != '':
        model.load_weights(path)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


# In[ ]:


# Initialize the model
model = get_model()


# In[ ]:


# Model callbacks
path = 'best_model_weights'
es_callback = EarlyStopping(monitor='val_loss')
mc_callback = ModelCheckpoint('{}.hdf5'.format(path), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
callbacks = [es_callback,mc_callback]


# In[ ]:


# Fit the model
model.fit(x_train,
          y_train,
          epochs=100,
          batch_size=500,
          validation_data=(x_validation, y_validation),
#           callbacks = callbacks
         )


# In[ ]:


# Load best weights
# model = get_model('{}.hdf5'.format(path))

model=model


# In[ ]:


# Test preprocessing
def padding(text):
    return pad_sequences(tokenizer.texts_to_sequences(text), maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
test_text = test[test.challenge_sequence <= 10].groupby('user_id').challenge.aggregate(lambda x: ' '.join(x)).reset_index()
x_test = padding(test_text.challenge)

# Get top 3 predictions for each user
pred = model.predict(x_test,batch_size=2048)
pred = pred.argsort(axis=1)[:,-3:][:,::-1]


# In[ ]:


# Write Predictions
df_list = []
for i in range(3):
    test_11 = test_text[['user_id']]
    test_11['user_sequence'] = test_11.user_id.astype(str) + '_'+str(i+11)
    test_11['challenge'] = encoder.inverse_transform(pred[:,i])
    df_list.append(test_11[['user_sequence','challenge']])
pd.concat(df_list).to_csv('bes_submission8.csv',index=False)


# ## Note:

#  If this kernel helped you: 
#  - Do upvote 
#  - Do follow 
#  - In case you have any query use comment section.
