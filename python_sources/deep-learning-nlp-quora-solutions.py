#!/usr/bin/env python
# coding: utf-8

# ## Quora Insincere Question Classification
# ---
# >  ### Outline of the notebook
# > 1. [**Load Library**](#1)
# > 1. [**Define the sequence of words**](#2)
# > 1. [**Setup Train data**](#3)
# > 1. [**Some Question Statistics**](#4)
# > 1. [**Embedding Datasetup**](#5)
# > 1. [**Value to Embedding**](#6)
# > 1. [**Data Providers**](#7)
# > 1. [**Model Training**](#8)
# > 1. [**Inference of result**](#9)
# > 1. [**Submission**](#10)

# # 1. Load Library<a id="1"></a>

# In[ ]:


# forked from : https://www.kaggle.com/mihaskalic/lstm-is-all-you-need-well-maybe-embeddings-also
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split


# # 2.Define the sequence of words<a id="2"></a>

# In[ ]:


SEQ_LEN = 28  # magic number - length to truncate sequences of words


# # 3.Setup Train data <a id="3"></a>

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
train_df, val_df = train_test_split(train_df, test_size=0.07)


# # 4.Some Question Statistics <a id="4"> </a>

# In[ ]:


#minor eda: average question length (in words) is 12  , majority are under 12 words
train_df.question_text.str.split().str.len().describe()


# # 5.Embedding Datasetup <a id="5"> </a>

# In[ ]:


### Unclear why fails to open [encoding error], format is same as for glove. Will Debug, Dan:
### f = open('../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt')

# embedding setup
# Source https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# 
embeddings_index = {}
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
# f = open('../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt')
for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# # 6. Value to Embedding<a id="6"></a> 

# In[ ]:


# Convert values to embeddings
def text_to_array(text):
    empyt_emb = np.zeros(300)
    text = text[:-1].split()[:SEQ_LEN]
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    embeds+= [empyt_emb] * (SEQ_LEN - len(embeds))
    return np.array(embeds)

# train_vects = [text_to_array(X_text) for X_text in tqdm(train_df["question_text"])]
val_vects = np.array([text_to_array(X_text) for X_text in tqdm(val_df["question_text"][:3000])])
val_y = np.array(val_df["target"][:3000])


# # 7.Data Providers <a id="7"> </a>

# In[ ]:


# Data providers
batch_size = 128

def batch_gen(train_df):
    n_batches = math.ceil(len(train_df) / batch_size)
    while True: 
        train_df = train_df.sample(frac=1.)  # Shuffle the data.
        for i in range(n_batches):
            texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]
            text_arr = np.array([text_to_array(text) for text in texts])
            yield text_arr, np.array(train_df["target"][i*batch_size:(i+1)*batch_size])


# # 8.Model Training <a id="8"> </a>

# In[ ]:


from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Bidirectional, CuDNNGRU
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


model = Sequential()
model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True),input_shape=(SEQ_LEN, 300)))
model.add(Bidirectional(CuDNNGRU(128, return_sequences=True),input_shape=(SEQ_LEN, 300)))
model.add(Bidirectional(CuDNNGRU(64)))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[ ]:


mg = batch_gen(train_df)
model.fit_generator(mg, epochs=25,
                    steps_per_epoch=1000,
                    validation_data=(val_vects, val_y),
                    verbose=True)


# In[ ]:


# Plot training & validation accuracy values
plt.style.use("fivethirtyeight")
plt.figure(figsize=(12,5))
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(12,5))
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# # 9.Inference of result <a id = "9"></a>

# In[ ]:


# prediction part
batch_size = 256
def batch_gen(test_df):
    n_batches = math.ceil(len(test_df) / batch_size)
    for i in range(n_batches):
        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]
        text_arr = np.array([text_to_array(text) for text in texts])
        yield text_arr

test_df = pd.read_csv("../input/test.csv")

all_preds = []
for x in tqdm(batch_gen(test_df)):
    all_preds.extend(model.predict(x).flatten())


# # 10.Submission<a id="10"> </a>

# In[ ]:


y_te = (np.array(all_preds) > 0.5).astype(np.int)

submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})
submit_df.to_csv("submission.csv", index=False)

