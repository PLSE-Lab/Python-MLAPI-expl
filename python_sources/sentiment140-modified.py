#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import tensorflow as tf
import csv
import random
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers


embedding_dim = 100
max_length = 16
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size=1599999
#Your dataset size here. Experiment using smaller values (i.e. 16000), but don't forget to train on at least 160000 to see the best effects
test_portion=.1

corpus = []


# In[ ]:



# Note that I cleaned the Stanford dataset to remove LATIN1 encoding to make it easier for Python CSV reader
# You can do that yourself with:
# iconv -f LATIN1 -t UTF8 training.1600000.processed.noemoticon.csv -o training_cleaned.csv
# I then hosted it on my site to make it easier to use in this notebook

get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/training_cleaned.csv     -O /tmp/training_cleaned.csv')


# In[ ]:


import pandas as pd
data = pd.read_csv('../input/training_cleaned.csv')
data.shape


# In[ ]:


num_sentences = 0

with open("../input/sentiment140-processed-glove100d/training_cleaned.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
      # Your Code here. Create list items where the first item is the text, found in row[5], and the second is the label. Note that the label is a '0' or a '4' in the text. When it's the former, make
      # your label to be 0, otherwise 1. Keep a count of the number of sentences in num_sentences
        list_item=[]
        list_item.append(row[5])
        list_item.append(0 if row[0]=='0' else 1)
        num_sentences = num_sentences + 1
        corpus.append(list_item)


# In[ ]:


print(num_sentences)
print(len(corpus))
print(corpus[1])

# Expected Output:
# 1600000
# 1600000
# ["is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!", 0]


# In[ ]:


sentences=[]
labels=[]
random.shuffle(corpus)
for x in range(training_size):
    sentences.append(corpus[x][0])
    labels.append(corpus[x][1])

tokenizer = Tokenizer(oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
vocab_size=len(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

split = int(test_portion * training_size)

test_sequences = padded[:split]
training_sequences = padded[split:]
test_labels = labels[:split]
training_labels = labels[split:]


# In[ ]:


print(vocab_size)
print(word_index['i'])
# Expected Output
# 138858
# 1


# In[ ]:


# Note this is the 100 dimension version of GloVe from Stanford
# I unzipped and hosted it on my site to make this notebook easier
'''!wget --no-check-certificate     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt     -O /tmp/glove.6B.100d.txt'''
embeddings_index = {};
with open('../input/sentiment140-processed-glove100d/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;


# In[ ]:


print(len(embeddings_matrix))
print(embeddings_matrix.shape)
# Expected Output
# 138859


# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),
    tf.keras.layers.Conv1D(32, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.LSTM(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[ ]:


num_epochs = 15
history = model.fit(training_sequences, training_labels, epochs=num_epochs, validation_data=(test_sequences, test_labels), verbose=2)

print("Training Complete")


# In[ ]:


model.save('/kaggle/working/my_model.h5',save_format='h5')

import os
os.chdir('/kaggle/working')

from IPython.display import FileLink
FileLink('my_model.h5')


# In[ ]:


import os
print(os.listdir())
print(os.getcwd())


# In[ ]:


new_model = tf.keras.models.load_model('../input/sentiment140-trained-model/my_model.h5')
new_model.summary()


# In[ ]:


num_epochs = 3
history = new_model.fit(training_sequences, training_labels, epochs=num_epochs, validation_data=(test_sequences, test_labels), verbose=2)

print("Training Complete")

new_model.save('/kaggle/working/my_model.h5', save_format='h5')


# In[ ]:


os.chdir('/kaggle/working/')


# In[ ]:




