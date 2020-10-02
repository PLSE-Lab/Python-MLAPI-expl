#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="https://camo.githubusercontent.com/200d24b84fb905e680fa1ebaa71af582e3d6e24e/68747470733a2f2f64327776666f7163396779717a662e636c6f756466726f6e742e6e65742f636f6e74656e742f75706c6f6164732f323031392f30362f576562736974652d5446534465736b746f7042616e6e65722e706e67" width=800><br></center>
# 
# 
# ## I decided to create this notebook while working on [Tensorflow in Practice Specialization](https://www.coursera.org/specializations/tensorflow-in-practice) on Coursera. I highly recommend this course, especially for beginners. Most of the ideas here belong to this course.

# ## Import necessary libraries

# In[ ]:


import tensorflow as tf
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# # 1. Explore the dataset

# In[ ]:


df = pd.read_json('../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json', lines=True)
df.head()


# In[ ]:


df.info()


# In[ ]:


sentences = df['headline']
labels = df['is_sarcastic']


# ## 1.1 Target value counts

# In[ ]:


plt.figure(figsize=(10,6))
sns.set(style="darkgrid")
sns.countplot(labels)


# ## 1.2 Sentence lengths distribution
# 
# ### This'll be useful for deciding the maxlen parameter in pad_sequence()

# In[ ]:


# Check sentence lengths
sentences_lengths = sentences.apply(lambda x: len(x))

plt.figure(figsize=(15,6))
plt.xlim(0, 150)

ax = sns.distplot(sentences_lengths, hist=False, color="r")
ax.set(xlabel='Sentence Lengths')


# ## 1.3 Train-Validation Split

# In[ ]:


from sklearn.model_selection import train_test_split

train_sentences, val_sentences, train_labels, val_labels = train_test_split(sentences, labels, test_size=0.2, random_state=0)

print(train_sentences.shape)
print(val_sentences.shape)
print(train_labels.shape)
print(val_labels.shape)


# ## Hyperparameters

# In[ ]:


# Tokenize and pad
vocab_size = 10000
oov_token = '<00V>'
max_length = 120
padding_type = 'post'
trunc_type = 'post'
embedding_dim = 16
num_epochs = 10


# ## Tokenize and Pad

# In[ ]:


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

val_sequences = tokenizer.texts_to_sequences(val_sentences)
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


# # 2. Build Models 

# # 2.1 Only Embedding 

# ## GlobalAveragePooling1D layer vs Flatten layer
# 
# ### Flattening simply converts a multi-dimensional object to one-dimensional by re-arranging the elements.
# 
# ### GlobalAveragePooling is a methodology used for better representation of your vector. It can be 1D/2D/3D. It uses a parser window which moves across the object and pools the data by averaging it (GlobalAveragePooling) or picking max value (GlobalMaxPooling).
# 
# ## 2.1.1 GlobalAveragePooling1D

# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(train_padded, 
                    train_labels, 
                    validation_data=(val_padded, val_labels), 
                    epochs=num_epochs, 
                    verbose=2)


# ## 2.1.2 Flatten

# In[ ]:


model2 = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model2.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model2.summary()
history_flatten = model2.fit(train_padded, 
                    train_labels, 
                    validation_data=(val_padded, val_labels), 
                    epochs=num_epochs, 
                    verbose=2)


# ## The second model with flatten layer slightly slower than the first model but has done a better job.

# # 2.2 LSTM

# In[ ]:


model_lstm = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_lstm.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model_lstm.summary()
history_lstm = model_lstm.fit(train_padded, 
                    train_labels, 
                    validation_data=(val_padded, val_labels), 
                    epochs=num_epochs, 
                    verbose=2)


# # 2.3 Multiple Layer LSTM

# In[ ]:


model_mul_lstm = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_mul_lstm.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model_mul_lstm.summary()
history_mul_lstm = model_mul_lstm.fit(train_padded, 
                    train_labels, 
                    validation_data=(val_padded, val_labels), 
                    epochs=num_epochs, 
                    verbose=2)

