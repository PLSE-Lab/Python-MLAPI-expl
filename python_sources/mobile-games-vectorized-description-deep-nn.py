#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import math
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# pd.set_option('display.max_colwidth', 0)


# In[ ]:


train = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv')
train.head()


# In[ ]:


train['User Rating Count'] = train['User Rating Count'].fillna(np.mean(train['User Rating Count']))
train['Average User Rating'] = train['Average User Rating'].fillna(np.mean(train['Average User Rating']))
train['User Rating Count'] = train['User Rating Count'].apply(lambda x: int(math.log(x,10)))
train['User Rating Count'] = train['User Rating Count'].apply(lambda x: 1 if x>=3 else 0)

print(train.groupby('User Rating Count')['ID'].nunique())


# <div style = "font-size:20px;line-height:25px;">Log of **User Rating Count** plotted against **Average User Rating**.<br>
# The figure above shows there is a correlation between how popular a game is vs it's rating
# </div>

# In[ ]:


import matplotlib.pyplot as plt
x = np.arange(1,3)
y = train.groupby(train['User Rating Count'])['Average User Rating'].mean()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)
plt.show()


# In[ ]:


train['Price'] = train['Price'].fillna(np.mean(train['Price']))
train['Price'] = train['Price'].apply(lambda x: 5.99 if x>=5.99 else x)
print(train.groupby('Price')['ID'].nunique())


# <div style = "font-size:20px;line-height:25px;">**Log** of **Price** plotted against **Average User Rating**.<br>
# The figure above shows there is a **loose** correlation between how popular a game is vs it's rating</div>

# In[ ]:


x = train.sort_values('Price',ascending=[True])
x = x['Price'].unique()
y = train.groupby(train['Price'])['Average User Rating'].mean()
print(x)
print(train.groupby(train['Price'])['Average User Rating'].mean())

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)
plt.show()


# In[ ]:


import tensorflow as tf
print(tf.__version__)


# <div style = "font-size:20px;line-height:23px;">Vectorized the **Description** and fed into Deep Neural Network to see if they play a role in the number of downloads(**User Rating Count**).
# Relation not found. I encourage someone to play with hyperparameters to further test this hypothesis.</div>

# In[ ]:


training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []
train_length = int(len(train)/2)

training_sentences = (train['Description'][:train_length])
training_labels = train['User Rating Count'][:train_length]
testing_sentences = (train['Description'][train_length:])
testing_labels = train['User Rating Count'][train_length:]# training_labels_final = np.array(training_labels)

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)


# In[ ]:


vocab_size = 10000
embedding_dim = 100
max_length = 150
trunc_type='post'
oov_tok = "<OOV>"


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)


# In[ ]:


reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[1]))
print(training_sentences[1])


# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='tanh')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[ ]:


num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))


# In[ ]:




