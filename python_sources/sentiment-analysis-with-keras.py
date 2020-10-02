#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding

from keras.layers import Dense,LSTM
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer


# In[ ]:


train = pd.read_csv('/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep="\t")
test = pd.read_csv('/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep="\t")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


print (train["Phrase"][0])
print (train["Phrase"][1])
print (train.shape)


# In[ ]:




import seaborn as sns

sns.countplot(x='Sentiment', data=train)


# # X and y

# In[ ]:



tokenizer = Tokenizer()

full_text = list(train['Phrase'].values) + list(test['Phrase'].values)
tokenizer.fit_on_texts(full_text)

X_train = tokenizer.texts_to_sequences(train['Phrase'].values)
X_test = tokenizer.texts_to_sequences(test['Phrase'].values)


# In[ ]:


max_len = 50
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)
len(X_train)


# In[ ]:


X_train = np.array(X_train)
X_test = np.array(X_test)


# In[ ]:


X_train.shape


# In[ ]:


y = train['Sentiment']


# # Model

# In[ ]:


vocab_size = len(tokenizer.word_index) + 1
vocab_size


# * LSTM layer take a 3D tensor with shape (batch_size, timesteps, input_dim).
# * in this case the correct shape is generated from the embedding layer
# * in this case the timesteps is 50 words
# 
# How embedding works in Keras
# input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
# output_dim: int >= 0. Dimension of the dense embedding. indicates the size of the embedding vectors
# each input integer is used as the index to access a table that contains all posible vectors. That is the reason why it needs to specify the size of the vocabulary as the first argument (so the table can be initialized).Once the network has been trained, we can get the weights of the embedding layer,and can be thought as the table used to map integers to embedding vectors.the underlying automatic differentiation engines (e.g., Tensorflow or Theano) manage to optimize these vectors associated to each input integer just like any other parameter.
# 
# it also possible to use pretained embedding https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

# In[ ]:


# TEST
vocabulary_size = len(tokenizer.word_counts)
vocabulary_size = vocabulary_size + 1
seq_len = X_train.shape[1]
model_test = Sequential()
model_test.add(Embedding(input_dim = vocabulary_size, output_dim = 2, input_length=seq_len))
model_test.compile('rmsprop', 'mse')
output_array = model_test.predict(X_train)
print (output_array.shape)
out1 = pd.DataFrame(output_array[0])
out1.tail()


# ### continue ...

# In[ ]:


vocabulary_size = len(tokenizer.word_counts)
vocabulary_size = vocabulary_size + 1
seq_len = X_train.shape[1]

model = Sequential()
model.add(Embedding(vocabulary_size, 25, input_length=seq_len))
model.add(LSTM(150, return_sequences=True)) # to stack LSTM we need return seq 
model.add(LSTM(150))
model.add(Dense(150, activation='relu'))

model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


# # Train

# In[ ]:


from keras.utils import to_categorical
y = to_categorical(y, num_classes=5)

y.shape


# In[ ]:


# fit model
model.fit(X_train, y, batch_size=256, epochs=10,verbose=1)


# # Submission

# In[ ]:


sub = pd.read_csv('/kaggle/input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv')
sub.head()


# In[ ]:


pred = model.predict(X_test, batch_size = 256, verbose = 1)
pred[0]


# In[ ]:


import matplotlib.pyplot as plt
predictions = np.round(np.argmax(pred, axis=1)).astype(int)
plt.hist(predictions, normed=False, bins=5)


# In[ ]:


sub['Sentiment'] = predictions
sub.to_csv("submission.csv", index=False)


# In[ ]:




