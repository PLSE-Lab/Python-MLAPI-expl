#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/Sentiment.csv')
df.head()


# In[ ]:


# Exploring the number of columns
df.info()


# In[ ]:


#lets see how many subjects do we have here
df['subject_matter'].unique()


# **Since the data set has many topics we can explore the sentiments for each topic**
# 
# In the next steps we will try to dive deepr and visualize the distributions of tweets against topics and confidence factors

# In[ ]:


#Using seaborn to plot a visually pleasant charts
import seaborn as se
import matplotlib as mpl
import matplotlib.pyplot as plt

#Here we are just controlling the size of the grid and the orientation of the labels in the x-axis
plt.figure(figsize=(20, 8))
plt.xticks(rotation=45)

se.countplot(x="subject_matter", data=df, palette="Greens_d");


# **We can that majoriety of the tweets don't belong to any specific topic.**
# 
# 
# > Lets look into the sentiment distribution through all the topics.

# In[ ]:


plt.figure(figsize=(20, 8))
plt.xticks(rotation=45)

se.barplot(x="subject_matter", y="sentiment_confidence", hue="sentiment", data=df);


# **We can see that the negative sentiment out-weights the positive and neutral sentiment in all the topics.**
# 
# > Except in the Gun control there is no positive conversations there,,, Surprise lol

# In[ ]:


df['tweet_location'].unique()


# In[ ]:


#Lets look at that in barplot
plt.figure(figsize=(20, 8))
plt.xticks(rotation=45)

se.countplot(x="tweet_location", data=df);


# **OK that was totally messed up, you shouldn't expect all the people to:**
# 
# * Use a real location.
# * Write the right spelling.
# * Have one way to write NYC (they will have 100 ways to do that)

# **Now lets just keep only the features that we need to train the model. Mostly  we will need the texts (tweets) and their sentiments (as our labels).
# We can try to do many other things like predicting the topic of the tweet, sentiments and so on. First lets just build an LSTM model to classify the sentiments of the tweets.**

# In[ ]:


#We extract the two columns text and sentiment from the dataframe
data = df[['text','sentiment']]

data.head()


# In[ ]:


#This code borrowed from Peter Nagy "LSTM Sentiment Analysis | Keras"

data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print(data[ data['sentiment'] == 'Positive'].size)
print(data[ data['sentiment'] == 'Negative'].size)
print(data[ data['sentiment'] == 'Neutral'].size)


for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')
    
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)


# **Training the model in the same in architecture used by Peter Nagy in "LSTM Sentiment Analysis | Keras"**

# In[ ]:


#Used the same model architecture used by Peter Nagy "LSTM Sentiment Analysis | Keras"

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# In[ ]:


Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[ ]:


batch_size = 32
model.fit(X_train, Y_train, nb_epoch = 7, batch_size=batch_size, verbose = 2)


# In[ ]:


#Retrain again in 90 epochs
model.fit(X_train, Y_train, epochs = 90, batch_size=batch_size, verbose = 2)


# **Experimenting new architecture**

# In[ ]:


embed_dim = 280
lstm_out = 210

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# In[ ]:


model.fit(X_train, Y_train, nb_epoch = 7, batch_size=batch_size, verbose = 2)


# In[ ]:


model.fit(X_train, Y_train, epochs = 90, batch_size=batch_size, verbose = 2)

