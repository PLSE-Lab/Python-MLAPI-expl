#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.layers import Bidirectional


from numpy import asarray
from numpy import zeros

import random


# Preparing Training Data obtained from XML files but saved as a pickle.

# In[ ]:


import pickle
with open('/kaggle/input/patents-data/abstracts_data.pkl', 'rb') as f:
    abstract_data = pickle.load(f)

del abstract_data['A']    
del abstract_data['F']    
del abstract_data['G']    
del abstract_data['H']    

del abstract_data['D']    
del abstract_data['E']    

categories = list(abstract_data.keys())


i = 0
labels = []
for category in categories:
    for x in range(9996):
        labels.append(i)
    i+=1

from keras.utils import np_utils
labels = np_utils.to_categorical(labels)    
labels = array(labels)

data = []

for category in categories:
    for d in abstract_data[category][0:9996]:
        data.append(d)
        


# In[ ]:


for category in categories:
    print(category+':'+str(len(abstract_data[category])))


# Preparing Testing Data obtained from XML files but saved as a pickle.

# In[ ]:


with open('/kaggle/input/patents-data/abstracts_testing_data.pkl', 'rb') as f:
    abstract_testing_data = pickle.load(f)

del abstract_testing_data['A']    
del abstract_testing_data['F']    
del abstract_testing_data['G']    
del abstract_testing_data['H']  
    
del abstract_testing_data['D']
del abstract_testing_data['E']    

categories = list(abstract_testing_data.keys())

i = 0
testing_labels = []
for category in categories:
    for x in range(300):
        testing_labels.append(i)
    i+=1

from keras.utils import np_utils
testing_labels = np_utils.to_categorical(testing_labels)    
testing_labels = array(testing_labels)


testing_data = []
for category in categories:
    for d in abstract_testing_data[category][0:300]:
        testing_data.append(d)


# Tokenzing and padding the training and testing data.

# In[ ]:


l=0
total_data=testing_data+data
for x in total_data:
    l=l+len(x)
length_long_sentence = (l)/(len(total_data))


# In[ ]:


word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(data+testing_data)

vocab_length = len(word_tokenizer.word_index) + 1
vocab_length

embedded_sentences = word_tokenizer.texts_to_sequences(data)

embedded_testing_sentences = word_tokenizer.texts_to_sequences(testing_data)


from nltk.tokenize import word_tokenize

word_count = lambda sentence: len(word_tokenize(sentence))
longest_sentence = max(data+testing_data, key=word_count)
#length_long_sentence = len(word_tokenize(longest_sentence))
#length_long_sentence = 230
length_long_sentence = int(length_long_sentence + 10)
padded_sentences = pad_sequences(embedded_sentences, length_long_sentence, padding='post')

padded_testing_sentences = pad_sequences(embedded_testing_sentences, length_long_sentence, padding='post')


# Importing and preparing the Glove embedding of each word in our dictionay.

# In[ ]:


embeddings_dict = {}
with open("/kaggle/input/patents-data/glove.6B.100d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector


embedding_matrix = zeros((vocab_length, 100))
for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


# Setting up the model which utilizes Glove embedding.

# In[ ]:


model = Sequential()
embedding_layer = Embedding(vocab_length, 100, weights=[embedding_matrix], input_length=length_long_sentence, trainable=False)
model.add(embedding_layer)
#model.add(Flatten())
model.add(Bidirectional(LSTM(100)))
#model.add(SimpleRNN(100))
#model.add(Dense(50, activation = 'relu'))
model.add(Dense(len(categories), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())


# Training the model.

# In[ ]:


history = model.fit(padded_sentences, labels,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)


# Verificaion of the model.

# In[ ]:


scores = model.evaluate(padded_testing_sentences, testing_labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# Loading the embedding of Prof Ralf.

# In[ ]:


patent_embeddings_dict = {}
with open("/kaggle/input/patents-data/patent-100.vec/patent-100.vec", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        try:
            vector = np.asarray(values[1:101], "float32")
            patent_embeddings_dict[word] = vector
        except ValueError:
            pass
        
patent_embedding_matrix = zeros((vocab_length, 100))
for word, index in word_tokenizer.word_index.items():
    embedding_vector = patent_embeddings_dict.get(word)
    if embedding_vector is not None:
        patent_embedding_matrix[index] = embedding_vector


# In[ ]:


model = Sequential()
embedding_layer = Embedding(vocab_length, 100, weights=[patent_embedding_matrix], input_length=length_long_sentence, trainable=False)
model.add(embedding_layer)
#model.add(Flatten())
model.add(Bidirectional(LSTM(100)))
#model.add(SimpleRNN(100))
#model.add(Dense(50, activation = 'relu'))
model.add(Dense(len(categories), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())


# In[ ]:


history = model.fit(padded_sentences, labels,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)


# In[ ]:


scores = model.evaluate(padded_testing_sentences, testing_labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# BERT

# In[ ]:


get_ipython().system('pip install keras-bert')
get_ipython().system('pip install keras-rectified-adam')

get_ipython().system('wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip')
get_ipython().system('unzip -o uncased_L-12_H-768_A-12.zip')


# In[ ]:


import codecs
import tensorflow as tf
from tqdm import tqdm
from chardet import detect
import keras
from keras_radam import RAdam
from keras.optimizers import Adam
from keras import backend as K
from keras_bert import load_trained_model_from_checkpoint


# In[ ]:


SEQ_LEN = 150
BATCH_SIZE = 50
EPOCHS = 5
LR = 1e-4


# In[ ]:


pretrained_path = 'uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')


# In[ ]:


model = load_trained_model_from_checkpoint(
      config_path,
      checkpoint_path,
      training=True,
      trainable=True,
      seq_len=SEQ_LEN,
  )


# In[ ]:


import codecs
from keras_bert import Tokenizer
token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
        
tokenizer = Tokenizer(token_dict)


# In[ ]:


i = 0
labels = []
for category in categories:
    for x in range(9996):
        labels.append(i)
    i+=1

i = 0
testing_labels = []
for category in categories:
    for x in range(300):
        testing_labels.append(i)
    i+=1

data_indices = []
for d in data:
    ids, segments = tokenizer.encode(d, max_len=SEQ_LEN)
    data_indices.append(ids)

test_data_indices = []
for d in testing_data:
    ids, segments = tokenizer.encode(d, max_len=SEQ_LEN)
    test_data_indices.append(ids)    

items = list(zip(data_indices, labels))
test_items = list(zip(test_data_indices, testing_labels))

np.random.shuffle(items)
np.random.shuffle(test_items)


# In[ ]:


indices_train, sentiments_train = zip(*items)
indices_train = np.array(indices_train)
train_x, train_y = [indices_train, np.zeros_like(indices_train)], np.array(sentiments_train)

indices_test, sentiments_test = zip(*test_items)
indices_test = np.array(indices_test)
test_x, test_y = [indices_test, np.zeros_like(indices_test)], np.array(sentiments_test)


# In[ ]:


inputs = model.inputs[:2]
dense = model.get_layer('NSP-Dense').output
outputs = keras.layers.Dense(units=2, activation='softmax')(dense)

model = keras.models.Model(inputs, outputs)
model.compile(
  Adam(learning_rate =LR),
  loss='sparse_categorical_crossentropy',
  metrics=['sparse_categorical_accuracy'],
)


# In[ ]:


history = model.fit(
    train_x,
    train_y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2
)


# In[ ]:


predicts = model.predict(test_x, verbose=True).argmax(axis=-1)
score = (np.sum(test_y == predicts) / test_y.shape[0])

print("\n%s: %.2f%%" % ('Accuracy', score*100))

