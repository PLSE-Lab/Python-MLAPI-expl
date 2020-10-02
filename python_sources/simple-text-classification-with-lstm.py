#!/usr/bin/env python
# coding: utf-8

# # Simple text classification with LSTM
# 
# In this approach, I chose to use a **Long Short Term Memory** (aka. LSTM) model as it is a good model to make predictions in numeric series. You can read below some additional descriptions of this approach.
# 
# Please, note that the data that I use in this notebook is already cleaned and ready to be used as input.

# In[ ]:


import copy
import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as sk_train_test_split

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer


# In[ ]:


def get_categories(df):
    return df['category'].unique()


# In[ ]:


# Load the Keras tokenizer
# Note that it will use only the most "num_words" used words
def load_tokenizer(X_data, num_words=150000):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(X_data)
    return tokenizer


# In[ ]:


def data_to_sequences(X_data, tokenizer, max_sequence_length):
    X_data = tokenizer.texts_to_sequences(X_data)
    X_data = sequence.pad_sequences(X_data, maxlen=max_sequence_length)
    return X_data


# In[ ]:


def train_test_split(X_data, Y_data, tokenizer, max_sequence_length):
    X_data = data_to_sequences(X_data, tokenizer, max_sequence_length)
    
    Y_data = Y_data.astype(np.int32)
    X_train, X_test, Y_train, Y_test = sk_train_test_split(X_data, Y_data, test_size=0.3)
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    return X_train, X_test, Y_train, Y_test


# ## Prepare dataset
# 
# Since the LSTM layer process sequences of number, I needed to **transform the texts into numbers**, so I used Keras Tokenizer to transform them. The Tokenizer contains a dictionary word-number, so every text we tokenize will be the same.

# In[ ]:


df = pd.read_csv('../input/bbc-articles-cleaned/tfidf_dataset.csv')
df.head()


# In[ ]:


X_data = df[['text']].to_numpy().reshape(-1)
Y_data = df[['category']].to_numpy().reshape(-1)


# I must to something similar with the categories: I only can **predict classes in a numeric format**, I had to transform each category into a number.

# In[ ]:


category_to_id = {}
category_to_name = {}

for index, c in enumerate(Y_data):
    if c in category_to_id:
        category_id = category_to_id[c]
    else:
        category_id = len(category_to_id)
        category_to_id[c] = category_id
        category_to_name[category_id] = c
    
    Y_data[index] = category_id

# Display dictionary
category_to_name


# I have **padded the short texts** with zeros and, in case that there are texts longer than a certain length (I fixed this value to $1000$ words), cut them (so all texts are the same length).

# In[ ]:


MAX_SEQUENCE_LENGTH = 1000

n_texts = len(X_data)
print('Texts in dataset: %d' % n_texts)

n_categories = len(get_categories(df))
print('Number of categories: %d' % n_categories)

print('Loading tokenizer...')
tokenizer = load_tokenizer(X_data)

print('Loading train dataset...')
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, tokenizer, MAX_SEQUENCE_LENGTH)

print('Done!')


# ## Create model and train
# 
# I have used the loss function $SparseCategoricalCrossentropy$ since I have a multi-class problem. This model returns **a vector of probabilities** where each value is the probability for each *category* (hence I must pick the highest one).
# 
# In the first layer of the model, I have used a **pre-trained set of embedding vectors** (so I should get more accurate results). These embeddings have been trained over Wikipedia 2014 database using the top $400K$ words (source: http://nlp.stanford.edu/data/glove.6B.zip). However, some words might not be present in that embedding set, so I have initialized the vectors (of those "missing words") with zeros.
# 
# Since the classes are not highly unbalanced, I have used **accuracy** to measure each model and find the best one. In other cases, you should consider to use a different metric, e.g. F1-score.

# In[ ]:


def load_embedding_matrix(tokenizer):
    embedding_dim = 100
    embeddings_index = {}

    f = open('../input/glove6b/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix, embedding_dim


# In[ ]:


def create_lstm_model(tokenizer, input_length, n_categories):
    word_index = tokenizer.word_index
    embedding_matrix, embedding_dim = load_embedding_matrix(tokenizer)

    model = Sequential()
    model.add(Embedding(input_dim=len(word_index) + 1,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=input_length,
                        trainable=True))
    model.add(SpatialDropout1D(0.3))
    model.add(LSTM(64,
                   activation='tanh',
                   dropout=0.2,
                   recurrent_dropout=0.5))
    model.add(Dense(n_categories, activation='softmax'))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model


# In this version of the notebook, I did not add the function to tunne the parameters (**hyperparameter optimization**), I just wrote the best parameter values that I found. If you are interested in it the process to obtain them, take a look to my github repository: https://github.com/DimasDMM/text-classification

# In[ ]:


EPOCHS = 10

model = create_lstm_model(tokenizer, MAX_SEQUENCE_LENGTH, n_categories)
history = model.fit(X_train,
                    Y_train,
                    epochs=EPOCHS,
                    validation_data=(X_test, Y_test),
                    verbose=1)


# ## Evaluation

# In[ ]:


def plot_confusion_matrix(X_test, Y_test, model):
    Y_pred = model.predict_classes(X_test)
    con_mat = tf.math.confusion_matrix(labels=Y_test, predictions=Y_pred).numpy()

    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    label_names = list(range(len(con_mat_norm)))

    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=label_names, 
                              columns=label_names)

    figure = plt.figure(figsize=(10, 10))
    sns.heatmap(con_mat_df, cmap=plt.cm.Blues, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

x_labels = range(1, EPOCHS + 1)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(x_labels, acc, color='b', linestyle='-', label='Training acc')
plt.plot(x_labels, val_acc, color='b', linestyle='--', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_labels, loss, color='b', linestyle='-', label='Training acc')
plt.plot(x_labels, val_loss, color='b', linestyle='--', label='Validation acc')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[ ]:


plot_confusion_matrix(X_test, Y_test, model)


# In[ ]:


scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))


# The global accuracy (i.e. taking into account all categories) looks pretty good for this selection of model and parameters: $95.96\%$.
# 
# If you take a closer look into the confussion matrix, you can see that most of mismatches are in the categories $2$ and $4$ which, according to the dictionary of category-ID, are `tech` and `business`.

# In[ ]:


category_to_name


# In[ ]:




