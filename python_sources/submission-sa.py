#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re


# In[ ]:


data = pd.read_csv('../input/train.tsv', sep="\t")


# In[ ]:


data.head()


# In[ ]:


# removing unnecessary columns

data = data[['tweet','label']]


# In[ ]:


data.head()


# In[ ]:


# data cleaning

stopwords = stopwords.words('english')

def tweet_cleaner(tweet):
    tweet = re.sub(r"@\w*", " ", str(tweet).lower()).strip() #removing username
    tweet = re.sub(r'https?://[A-Za-z0-9./]+', " ", str(tweet).lower()).strip() #removing links
    tweet = re.sub(r'[^a-zA-Z]', " ", str(tweet).lower()).strip() #removing sp_char
    tw = []
    
    for text in tweet.split():
        if text not in stopwords:
            tw.append(text)
    
    return " ".join(tw)


# In[ ]:


data.tweet = data.tweet.apply(lambda x: tweet_cleaner(x))


# In[ ]:


data.head()


# In[ ]:


# text to sequence

tokenizer = Tokenizer(num_words=3000)
tokenizer.fit_on_texts(data['tweet'].values)
X = tokenizer.texts_to_sequences(data['tweet'].values)
X = pad_sequences(X)


# In[ ]:


X.shape


# In[ ]:


Y = pd.get_dummies(data['label']).values


# In[ ]:


Y


# In[ ]:


# model

model = Sequential()
model.add(Embedding(3000, 200,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# In[ ]:


model.fit(X, Y, epochs = 4, batch_size=32, verbose = 2)


# In[ ]:


def sentiment(text):
    
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=96)
    score = model.predict([x_test])[0]
    
    if np.argmax(score) == 2:
        a = "POSITIVE"
    elif np.argmax(score) == 0:
        a = "NEGATIVE"
    elif np.argmax(score) == 1:
        a = "NEUTRAL"
    return print(a)


# In[ ]:


sentiment("I hate running.")

