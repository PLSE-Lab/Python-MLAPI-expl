#!/usr/bin/env python
# coding: utf-8

# # Amazon Reviews Dataset
# 
# This dataset contains several million reviews of Amazon products, with the reviews separated into two classes for positive and negative reviews. The two classes are evenly balanced here.
# 
# This is a large dataset, and the version that I am using here only has the text as a feature with no other metadata. This makes this an interesting dataset for doing NLP work. It is data written by users, so it's like that there are various typos, nonstandard spellings, and other variations that you may not find in curated sets of published text.
# 
# In this notebook, I will do some very simple text processing and then try out two fairly unoptimized deep learning models:
# 1. A convolutional neural net
# 2. A recurrent neural net
# These models should achieve results that are within a couple percent of state of the art at predicting the binary sentiment of the reviews.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import models, layers, optimizers
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import bz2
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import re

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Reading the text
# 
# The text is held in a compressed format. Luckily, we can still read it line by line. The first word gives the label, so we have to convert that into a number and then take the rest to be the comment.

# In[ ]:


def get_labels_and_texts(file):
    labels = []
    texts = []
    for line in bz2.BZ2File(file):
        x = line.decode("utf-8")
        labels.append(int(x[9]) - 1)
        texts.append(x[10:].strip())
    return np.array(labels), texts
train_labels, train_texts = get_labels_and_texts('../input/train.ft.txt.bz2')
test_labels, test_texts = get_labels_and_texts('../input/test.ft.txt.bz2')


# ## Text Preprocessing
# 
# The first thing I'm going to do to process the text is to lowercase everything and then remove non-word characters. I replace these with spaces since most are going to be punctuation. Then I'm going to just remove any other characters (like letters with accents). It could be better to replace some of these with regular ascii characters but I'm just going to ignore that here. It also turns out if you look at the counts of the different characters that there are very few unusual characters in this corpus.

# In[ ]:


import re
NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')
def normalize_texts(texts):
    normalized_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        normalized_texts.append(no_non_ascii)
    return normalized_texts
        
train_texts = normalize_texts(train_texts)
test_texts = normalize_texts(test_texts)


# ## Train/Validation Split
# Now I'm going to set aside 20% of the training set for validation.

# In[ ]:


from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, random_state=57643892, test_size=0.2)


# Keras provides some tools for converting text to formats that are useful in deep learning models. I've already done some processing, so now I will just run a Tokenizer using the top 12000 words as features.

# In[ ]:


MAX_FEATURES = 12000
tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(train_texts)
train_texts = tokenizer.texts_to_sequences(train_texts)
val_texts = tokenizer.texts_to_sequences(val_texts)
test_texts = tokenizer.texts_to_sequences(test_texts)


# ## Padding Sequences
# In order to use batches effectively, I'm going to need to take my sequences and turn them into sequences of the same length. I'm just going to make everything here the length of the longest sentence in the training set. I'm not dealing with this here, but it may be advantageous to have variable lengths so that each batch contains sentences of similar lengths. This might help mitigate issues that arise from having too many padded elements in a sequence. There are also different padding modes that might be useful for different models.
# 

# In[ ]:


MAX_LENGTH = max(len(train_ex) for train_ex in train_texts)
train_texts = pad_sequences(train_texts, maxlen=MAX_LENGTH)
val_texts = pad_sequences(val_texts, maxlen=MAX_LENGTH)
test_texts = pad_sequences(test_texts, maxlen=MAX_LENGTH)


# ## Convolutional Neural Net Model
# 
# I'm just using fairly simple models here. This CNN has an embedding with a dimension of 64, 3 convolutional layers with the first two having match normalization and max pooling and the last with global max pooling. The results are then passed to a dense layer and then the output.

# In[ ]:


def build_model():
    sequences = layers.Input(shape=(MAX_LENGTH,))
    embedded = layers.Embedding(MAX_FEATURES, 64)(sequences)
    x = layers.Conv1D(64, 3, activation='relu')(embedded)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(3)(x)
    x = layers.Conv1D(64, 5, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(5)(x)
    x = layers.Conv1D(64, 5, activation='relu')(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu')(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=sequences, outputs=predictions)
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return model
    
model = build_model()


# In[ ]:


model.fit(
    train_texts, 
    train_labels, 
    batch_size=128,
    epochs=2,
    validation_data=(val_texts, val_labels), )


# Once this finishes training, we should find that we get an accuracy of around 94% for this model.

# In[ ]:


preds = model.predict(test_texts)
print('Accuracy score: {:0.4}'.format(accuracy_score(test_labels, 1 * (preds > 0.5))))
print('F1 score: {:0.4}'.format(f1_score(test_labels, 1 * (preds > 0.5))))
print('ROC AUC score: {:0.4}'.format(roc_auc_score(test_labels, preds)))


# ## Recurrent Neural Net Model
# For an RNN model I'm also going to use a simple model. This has an embedding, two GRU layers, followed by 2 dense layers and then the output layer. I'm using the CuDNNGRU rather than GRU because the former will run much faster (over a factor of 10 I think on Kaggle's servers.

# In[ ]:


def build_rnn_model():
    sequences = layers.Input(shape=(MAX_LENGTH,))
    embedded = layers.Embedding(MAX_FEATURES, 64)(sequences)
    x = layers.CuDNNGRU(128, return_sequences=True)(embedded)
    x = layers.CuDNNGRU(128)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(100, activation='relu')(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=sequences, outputs=predictions)
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return model
    
rnn_model = build_rnn_model()


# In[ ]:


rnn_model.fit(
    train_texts, 
    train_labels, 
    batch_size=128,
    epochs=1,
    validation_data=(val_texts, val_labels), )


# And we should find that this model will end up with an accuracy similar to the CNN model. I haven't bothered to set the seeds, but it can go as high as 95%.

# In[ ]:


preds = rnn_model.predict(test_texts)
print('Accuracy score: {:0.4}'.format(accuracy_score(test_labels, 1 * (preds > 0.5))))
print('F1 score: {:0.4}'.format(f1_score(test_labels, 1 * (preds > 0.5))))
print('ROC AUC score: {:0.4}'.format(roc_auc_score(test_labels, preds)))


# ## What else could we do?
# 
# There are lots of things I haven't tried here. I think the original data from Amazon has other fields that could be added to the model. Additionally, we haven't added any global features from the samples such as length, character level features, and more. We could even attempt to run character-level deep learning models, which might be able to reduce sensitivity to misspellings. In online reviews, character level features could be quite important as users could intentionally misspell things to avoid moderation. However, these models are already performing at well over 90% so at this point any gains are going to be pretty small.
