#!/usr/bin/env python
# coding: utf-8

# One of the more underappreciated aspects of Kaggle competitions is that we can "repurpose" them for all sorts of different tasks that go beyond the socope of the original context. Not all of the models that we build are equally generalizable, but some can be used for a wide variety of purposes. 
# 
# In this kernel I'd like to see how do models built on [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/) perform on non-competition "real world" data. Here I will just use one model that was built inside of a [kernel](https://www.kaggle.com/tunguz/bi-gru-lstm-cnn-poolings-fasttext). The kernel scores in the 0.984x AUC range. It's a respectable score, but well below the top solutions that scored in the 0.988x range. I have used this approach to find out [how toxic are Hillary Clinton and Donald Trump tweets](https://www.kaggle.com/tunguz/how-toxic-are-hillary-and-trump-tweets/), with some interesting insights. 
# 
# First, let's load up the Python libraries.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import time
start_time = time.time()
from sklearn.model_selection import train_test_split
import sys, os, re, csv, codecs, numpy as np, pandas as pd
np.random.seed(32)
os.environ["OMP_NUM_THREADS"] = "4"
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import GRU, BatchNormalization, Conv1D, MaxPooling1D

import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Now we'll load the Quora datasets:

# In[ ]:


train = pd.read_csv('../input/quora-insincere-questions-classification/train.csv').fillna(' ')
test = pd.read_csv('../input/quora-insincere-questions-classification/test.csv').fillna(' ')
test_qid = test['qid']
train_qid = train['qid']
train_target = train['target'].values

train_text = train['question_text']
test_text = test['question_text']

all_text = pd.concat([train_text, test_text])


# In[ ]:


embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"


# We will embed words from these tweets into a word-vector space using one of the previously trained word embeddings. Here we use a 300-dimensional vector space that comes curtesy of FastText. Unfortunately, this embedding is not available for the Quora compatition, but as we are using this kernel just for the educational purposes, that will be fine. We will also limit the length of text to 220 words. This is an overkill for questions, but for general purpose it is rather small text length. The original was aimed at much longer text sizes, and this was a reasonable length for those purposes. The best embedding that we used in Toxic limited length to 900 words.

# In[ ]:


embed_size = 300
max_features = 130000
max_len = 220

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train_text = train_text.str.lower()
test_text = test_text.str.lower()
all_text = all_text.str.lower()


# In order for our pretrained models to work, we need to transform the text here into the appropriate vectorized format.

# In[ ]:


tk = Tokenizer(num_words = max_features, lower = True)
tk.fit_on_texts(all_text)
all_text = tk.texts_to_sequences(all_text)
train_text = tk.texts_to_sequences(train_text)
test_text = tk.texts_to_sequences(test_text)


# We also need to pad the tweets that are less than 220 words, which is essentially all of them.

# In[ ]:


train_pad_sequences = pad_sequences(train_text, maxlen = max_len)
test_pad_sequences = pad_sequences(test_text, maxlen = max_len)
all_pad_sequences = pad_sequences(all_text, maxlen = max_len)


# In[ ]:


def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))


# In[ ]:


word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[ ]:


model = load_model("../input/bi-gru-lstm-cnn-poolings-fasttext/best_model.hdf5")


# In[ ]:


train_pred = model.predict(train_pad_sequences, batch_size = 1024, verbose = 1)
test_pred = model.predict(test_pad_sequences, batch_size = 1024, verbose = 1)
#all_pred = model.predict(all_pad_sequences, batch_size = 1024, verbose = 1)


# Let's see what's the maximum probability for this model:

# In[ ]:


train_pred.max()


# In[ ]:


test_pred.max()


# In other words, at nearly 1.0 probability the model seems pretty confident about the "toxicity" of some of the tweets.
# 
# Now let's put the predictions into a dataframe, so we can have a better view of them and how they relate to the actual tweets.

# In[ ]:


toxic_predictions_train = pd.DataFrame(columns=list_classes, data=train_pred)
toxic_predictions_test = pd.DataFrame(columns=list_classes, data=test_pred)
toxic_predictions_train['question_text'] = train['question_text'].values
toxic_predictions_test['question_text'] = test['question_text'].values
toxic_predictions_train['qid'] = train_qid
toxic_predictions_test['qid'] = test_qid


# In[ ]:


toxic_predictions_train.head()


# In[ ]:


toxic_predictions_test.head()


# In[ ]:


toxic_predictions_train[list_classes].describe()


# In[ ]:


toxic_predictions_test[list_classes].describe()


# The worst 'toxic' train questions:

# In[ ]:


print(toxic_predictions_train.sort_values(by=['toxic'], ascending=False)['question_text'].head(10).values)


# In[ ]:


print(toxic_predictions_train.sort_values(by=['severe_toxic'], ascending=False)['question_text'].head(10).values)


# In[ ]:


print(toxic_predictions_train.sort_values(by=['obscene'], ascending=False)['question_text'].head(10).values)


# In[ ]:


print(toxic_predictions_train.sort_values(by=['threat'], ascending=False)['question_text'].head(10).values)


# In[ ]:


print(toxic_predictions_train.sort_values(by=['insult'], ascending=False)['question_text'].head(10).values)


# In[ ]:


print(toxic_predictions_train.sort_values(by=['identity_hate'], ascending=False)['question_text'].head(10).values)


# In other words, the model seems not to work too well. The fact that soem of these are marked almost 1.0 on probablity scale shows the limitations of this model. It is very likely that Quora employs a very good set of toxicity-detection tools on their site, so the kinds of questions we get in this competition will most likley be already heavily vetted for inapropriate content. 
