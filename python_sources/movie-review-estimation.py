#!/usr/bin/env python
# coding: utf-8

# In this kernel we will create a recurrent neural network with LSTM layer for predicting positive or negative estimations based on movie reviews.

# In[ ]:


import keras as k
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import SpatialDropout1D
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
import re
import io
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Creating new data frame with two columns: first column for movie reviews, and the second one for estimations.

# In[ ]:


df = pd.DataFrame(columns = ['review', 'estimation'])


# Gathering all reviews from different folders into one csv file. Assigning "1" to positive reviews and "0" to negative reviews:

# In[ ]:


path = "../input/movie-review/txt_sentoken/txt_sentoken"
pos_reviews = os.listdir(path + '/pos/')
for i in range(len(pos_reviews)):
    with io.open(path+'/pos/'+pos_reviews[i], "r") as f:
        text = f.read().lower()
        df = df.append({'review':text, 'estimation': 1}, ignore_index=True)
        
neg_reviews = os.listdir(path + '/neg/')
for i in range(len(pos_reviews)):
    with io.open(path+'/neg/'+neg_reviews[i], "r") as f:
        text = f.read().lower()
        df = df.append({'review':text, 'estimation': 0}, ignore_index=True)


# Removing all characters that are not letters or numbers:

# In[ ]:


df['review'] = df['review'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))   


# Defining our vocabulary with 40000 most common words and assigning a unique index (token) to each. 0 is a reserved index that won't be assigned to any word. Then replacing words with tokens, leaving out those not present in the vocabulary. Finally, creating an array containing token sequences with paddings and an extra dimension holding a number of timesteps.

# In[ ]:


tokenizer = k.preprocessing.text.Tokenizer(num_words=40000, split=' ')
tokenizer.fit_on_texts(df['review'].values)
X = tokenizer.texts_to_sequences(df['review'].values)
X = k.preprocessing.sequence.pad_sequences(X)
sequence_dict = tokenizer.word_index;


# In[ ]:


Y = df['estimation'].values


# Setting parameters:

# In[ ]:


output_dim = 30
lstm_units = 30
dropoutLSTM = 0.5
batch_size = 128
epochs = 30
optimizer = k.optimizers.Adam(lr=0.01, decay=0.01)


# Setting up the model:

# In[ ]:


model = Sequential()
model.add(Embedding(40000, output_dim = output_dim, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.5))
model.add(LSTM(lstm_units, recurrent_dropout = dropoutLSTM))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer=optimizer, metrics = ['accuracy'])


# Fitting model:

# In[ ]:


history = model.fit(X, Y, validation_split=0.1, batch_size = batch_size, epochs = epochs)


# Saving  model:

# In[ ]:


model.save('model.h5')


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# The main problem of this model is that accuracy on the train subset is much bigger than that on the test subset. Unfortunately, our dataset is too small to avoid overfitting even by means of various regularization technics such as reducing network nodes and dropout.

# Another posible way to fix this problem is to use another algorithms of word embedding. Here we will use GloVe model with 100 dimensional vectors:

# In[ ]:


embeddings_index = dict();
with open('../input/glove6b/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;


# Creating embedding matrix:

# In[ ]:


vocab_size = len(sequence_dict);
embeddings_matrix = np.zeros((vocab_size+1, 100));
for word, i in sequence_dict.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;


# Setting up, fitting and plotting the new model:

# In[ ]:


lstm_units = 30
dropoutLSTM = 0.5
batch_size = 64
epochs = 30
optimizer = k.optimizers.Adam(lr=0.01, decay=0.01)


# In[ ]:


model = Sequential()
model.add(Embedding(embeddings_matrix.shape[0], output_dim = 100, input_length = X.shape[1], weights=[embeddings_matrix], trainable=False))
model.add(SpatialDropout1D(0.5))
model.add(LSTM(lstm_units, recurrent_dropout = dropoutLSTM))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer=optimizer, metrics = ['accuracy'])


# In[ ]:


history = model.fit(X, Y, validation_split=0.1, batch_size = batch_size, epochs = epochs)


# In[ ]:


model.save('model_GloVe.h5')


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# We can see that with GloVe embedding the variance becomes lower.

# Adding second LSTM layer and tweaking hyperparameters:

# In[ ]:


lstm_units1 = 150
lstm_units2 = 100
dropoutLSTM = 0.3
dense_units = 30
batch_size = 32
epochs = 30
optimizer = k.optimizers.Adam(lr=0.01, decay=0.0001, clipnorm=1)


# In[ ]:


model = Sequential()
model.add(Embedding(embeddings_matrix.shape[0], output_dim = 100, input_length = X.shape[1], weights=[embeddings_matrix], trainable=False))
model.add(LSTM(lstm_units1, recurrent_dropout = dropoutLSTM, return_sequences=True))
model.add(SpatialDropout1D(0.1))
model.add(LSTM(lstm_units2, recurrent_dropout = dropoutLSTM))
model.add(Dense(dense_units, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer=optimizer, metrics = ['accuracy'])


# In[ ]:


history = model.fit(X, Y, validation_split=0.1, batch_size = batch_size, epochs = epochs)


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


model.save('model_improved.h5')


# Finally, we've reached 100% on the test subset. Training accuracy is lower due to dropout regularization that uses only a fraction of neurons to estimate train accuracy. <a href="https://keras.io/getting-started/faq/#why-is-the-training-loss-much-higher-than-the-testing-loss">Keras FAQ: Why is the training loss much higher than the testing loss?!</a>
