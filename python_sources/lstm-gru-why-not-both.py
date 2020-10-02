#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm

import math
from sklearn.model_selection import train_test_split


# # Setup

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
train_df, val_df = train_test_split(train_df, test_size=0.1)


# In[ ]:


# embdedding setup
# Source https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
embeddings_index = {}
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[ ]:


# Convert values to embeddings
def text_to_array(text):
    empyt_emb = np.zeros(300)
    text = text[:-1].split()[:100]
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    embeds+= [empyt_emb] * (100 - len(embeds))
    return np.array(embeds)

# train_vects = [text_to_array(X_text) for X_text in tqdm(train_df["question_text"])]
val_vects = np.array([text_to_array(X_text) for X_text in tqdm(val_df["question_text"][:5000])])
val_y = np.array(val_df["target"][:5000])


# In[ ]:


val_vects.shape


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


# # Training

# In[ ]:


from keras.models import Sequential, Input
from keras.models import Model
from keras.layers import concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, CuDNNLSTM, CuDNNGRU, Dense, Bidirectional, SpatialDropout1D, Conv1D


# In[ ]:


dr = 0.25
units = 96

inp = Input(shape = ((100, 300)))
x1 = SpatialDropout1D(dr)(inp)

x = Bidirectional(CuDNNGRU(units, return_sequences = True))(x1)
x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    
y = Bidirectional(CuDNNLSTM(units, return_sequences = True))(x1)
y = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(y)
    
avg_pool1 = GlobalAveragePooling1D()(x)
max_pool1 = GlobalMaxPooling1D()(x)
    
avg_pool2 = GlobalAveragePooling1D()(y)
max_pool2 = GlobalMaxPooling1D()(y)
    
    
x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])

x = Dense(1, activation = "sigmoid")(x)
model = Model(inputs = inp, outputs = x)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


mg = batch_gen(train_df)
model.fit_generator(mg, epochs=30,
                    steps_per_epoch=1000,
                    validation_data=(val_vects, val_y),
                    verbose=True)


# # Inference

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


# In[ ]:


all_preds_val = []
for x in tqdm(batch_gen(val_df)):
    all_preds_val.extend(model.predict(x).flatten())


# In[ ]:


val_y = val_df["target"].values


# In[ ]:


from sklearn.metrics import f1_score


# In[ ]:


y_val = (np.array(all_preds_val) > 0.1).astype(np.int)
f1_score(val_y, y_val)


# In[ ]:


0.6764157867991644


# In[ ]:


0.6762409528694363


# In[ ]:


0.6733616435211752


# In[ ]:


0.6676628479716642


# In[ ]:


f1_score(val_y, y_val)


# In[ ]:


score = 0
thresh = .5
for i in np.arange(0.1, 0.991, 0.01):
    y_val = (np.array(all_preds_val) > i).astype(np.int)
    temp_score = f1_score(val_y, y_val)
    if(temp_score > score):
        score = temp_score
        thresh = i

print("CV: {}, Threshold: {}".format(score, thresh))


# In[ ]:


y_te = (np.array(all_preds) > thresh).astype(np.int)

submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})
submit_df.to_csv("submission.csv", index=False)


# In[ ]:




