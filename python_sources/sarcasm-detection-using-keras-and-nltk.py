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


import pandas as pd
import numpy as np
import nltk


# In[ ]:


dataset = pd.read_json('../input/Sarcasm_Headlines_Dataset.json',lines=True)


# In[ ]:


data = dataset.iloc[:,1:]


# In[ ]:


X = []
Y = []
for i in range(len(dataset)):
  X.append(dataset.iloc[i,0])
  Y.append([dataset.iloc[i,1]])


# We Are now going to remove all the stopwords and also Lemmatize the tokens using nltk WordNetLemmatizer.
# The tokens are generated from sentences using RegexpTokenizer from nltk

# In[ ]:


from nltk.corpus import stopwords
import string
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer


# In[ ]:


stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


# In[ ]:


tokenizer_nltk = RegexpTokenizer('[0-9]*\.[0-9]*|\w+')


# In[ ]:


for i in range(len(data)): 
    tokens = tokenizer_nltk.tokenize(data.iloc[i,0])
    tokens = [lemmatizer.lemmatize(x) for x in tokens]
    tok = [x for x in tokens if ((x not in stop) and (x not in string.punctuation))]
    data.iloc[i,0] = ' '.join(tok)


# In[ ]:


data


# In[ ]:


X = []
Y = []
for i in range(len(dataset)):
  X.append(data.iloc[i,0])
  Y.append([data.iloc[i,1]])
y = np.array(Y)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


Xs_train, Xs_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42,shuffle=True)


# In[ ]:


import keras


# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[ ]:


tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(Xs_train)

X_train = tokenizer.texts_to_sequences(Xs_train)
X_test =  tokenizer.texts_to_sequences(Xs_test)

vocab_size = len(tokenizer.word_index) + 1


# In[ ]:


print(vocab_size)


# In[ ]:


print(Xs_train[0])
print(X_train[0])


# In[ ]:


maxlen = 32
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# In[ ]:


from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Dropout, Flatten ,LSTM
from keras.layers import Conv1D,GlobalMaxPool1D,MaxPooling1D


# In[ ]:


from keras.models import Sequential
from keras import layers

embedding_dim = 128

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(LSTM(embedding_dim, return_sequences=True))
model.add(layers.Conv1D(128, 3, activation='relu'))
#model.add(layers.MaxPooling1D(3,strides=2))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.Nadam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

