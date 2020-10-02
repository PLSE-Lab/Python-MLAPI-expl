#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# **Import all necessary libraries here**

# In[2]:


import os
print(os.listdir("../input"))

this function is to clean the sequence
# In[3]:


def clean_str(string):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


# In[5]:


# reading data
df = pd.read_csv('../input/text-analysis/all_tickets-1551435513304.csv')
df = df.dropna()
df = df.reset_index(drop=True)
print('Shape of dataset ',df.shape)
print(df.columns)


# In[ ]:


np.unique(df.urgency)


# In[ ]:


df.head()


# In[ ]:


print(df.body.shape)


# In[ ]:


from bs4 import BeautifulSoup
import re


# In[ ]:


texts = []
labels = []


for idx in range(df.body.shape[0]):
    text = BeautifulSoup(df.body[idx])
    texts.append(clean_str(str(text.get_text().encode())))

for idx in df['urgency']:
    labels.append(idx)


# In[ ]:


texts[0:10]


# In[ ]:


sum(np.isnan(labels))


# In[ ]:


MAX_NB_WORDS = 20000


# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer


# In[ ]:


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Number of Unique Tokens',len(word_index))


# In[ ]:


texts[3]
len(texts[3])


# In[ ]:


word_index


# In[ ]:


print(sequences[5])
print(labels[5])


# In[ ]:


MAX_SEQUENCE_LENGTH = 1000


# In[ ]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical


# In[ ]:


data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of Data Tensor:', data.shape)
print('Shape of Label Tensor:', labels.shape)


# In[ ]:


print("length of the target column = ",len(labels))
unique, counts = np.unique(np.argmax(labels,axis=1), return_counts=True)

print(np.asarray((unique, counts)).T)


# In[ ]:


from sklearn.utils import shuffle

X, Y = shuffle(data,labels, random_state=123)


# In[ ]:


Y[0]


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=123)


# In[ ]:


embeddings_index = {}
f = open('../input/glove-embedding-weights/glove.6B.100d.txt',encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# In[ ]:


print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))


# In[ ]:


len(embeddings_index)


# In[ ]:


from tensorflow.keras.layers import Embedding


# In[ ]:


EMBEDDING_DIM=100


# In[ ]:


embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,trainable=True)


# In[ ]:


from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint


# In[ ]:


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_cov1= Conv1D(256, 5, activation='relu')(embedded_sequences)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(5)(l_cov2)
l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
l_flat = Flatten()(l_pool3)
l_dense = Dense(64, activation='relu')(l_flat)
preds = Dense(4, activation='softmax')(l_dense)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("Simplified convolutional neural network")
model.summary()
cp=ModelCheckpoint('model_cnn.hdf5',monitor='val_acc',verbose=1,save_best_only=True)


# check the model_cnn.h5 above file........................................................

# In[ ]:


history=model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=15, batch_size=128,callbacks=[cp])


# In[ ]:


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()


# In[ ]:


plot_graphs(history,'loss')


# In[ ]:


plot_graphs(history,'acc')


# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import LSTM,Bidirectional


# In[ ]:


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
lstm1=Bidirectional(LSTM(64))(embedded_sequences)
dense1=Dense(64, activation='relu')(lstm1)
dense2=Dense(4, activation='softmax')(dense1)


# In[ ]:


model = Model(sequence_input, dense2)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


# In[ ]:


model.summary()


# In[ ]:


history = model.fit(X_train,y_train, epochs=10,
                    validation_data=[X_test,y_test])


# In[ ]:


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))

