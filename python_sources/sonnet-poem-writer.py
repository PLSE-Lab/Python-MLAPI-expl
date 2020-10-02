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


from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as np_utils


# In[ ]:


sonnet = open('/kaggle/input/testingg/sonnet.txt', 'r', encoding = 'utf-8').read()
sonnet = sonnet.lower()
sonnet = sonnet.split("\n")
print(sonnet[:10])


# In[ ]:


print(len(sonnet))


# In[ ]:


import string
def clean_text(text):
    txt = "".join(ch for ch in text if ch not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt

corpus = [clean_text(line) for line in sonnet]
corpus[:7]


# In[ ]:


t = Tokenizer()
t.fit_on_texts(corpus)


# In[ ]:


total_words = len(t.word_index) + 1
input_sequences = []
#Generate Sequences
for line in corpus:
    tokens = t.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        n_grams = tokens[: i+1]
        input_sequences.append(n_grams)
input_sequences[:10]


# In[ ]:


#pad all the sequences to same length
max_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen = max_len, padding = 'pre'))
x_train, y_train = input_sequences[:, :-1], input_sequences[:, -1]
y_train = np_utils.to_categorical(y_train, num_classes = total_words)
x_train.shape, y_train.shape


# In[ ]:


total_words, max_len


# In[ ]:


import tensorflow as tf
model = tf.keras.Sequential() # define your model normally
model.add(Embedding(total_words, 10, input_length = max_len - 1))
model.add(LSTM(512, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(total_words, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
model.summary()


# In[ ]:


callbacks = [ModelCheckpoint("poem-sonnet.h5", monitor = 'loss', mode = 'min', save_best_only = True)]
history = model.fit(x_train[:15360], y_train[:15360], epochs = 1000, batch_size = 128, callbacks = callbacks)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])


# In[ ]:


def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = t.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list)
        
        output_word = ""
        for word,index in t.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()


# In[ ]:


print ("1. ",generate_text("Julius", 20, model, max_len))
print ("2. ",generate_text("Thou", 20, model, max_len))
print ("3. ",generate_text("King is", 20, model, max_len))
print ("4. ",generate_text("Death of", 20, model, max_len))
print ("5. ",generate_text("The Princess", 20, model, max_len))
print ("6. ",generate_text("Thanos", 20, model, max_len))


# In[ ]:


print ("6. ",generate_text("Hello my dear", 200, model, max_len))


# In[ ]:



print ("6. ",generate_text("read my the", 200, model, max_len))


# In[ ]:


print ("6. ",generate_text("Once upon a Time", 200, model, max_len))


# In[ ]:


model.load_weights('/kaggle/working/poem-sonnet.h5')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
print ("6. ",generate_text("Once upon a Time", 200, model, max_len))
model.save("poem-writer.h5")

