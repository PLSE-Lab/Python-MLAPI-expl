#!/usr/bin/env python
# coding: utf-8

# # Getting Started with the competition

# This notebook is an introduction to the ongoing Jigsaw Multilingual Toxic Comment Challenge. The challenge contains of comments from previous two competitions of Jigsaw Toxic comment classification. The train dataset contains of comments in English whereas validation and test set contain of comments in different languages. The task is to predict whether the comment is toxic or not.
# We will be using DistilBert along with Keras for this competition since it is faster than BERT and performance is roughly the same as that of BERT. 
# 
# Read following articles to get a headstart about Bert and how it works:
# * http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
# * https://stackabuse.com/text-classification-with-bert-tokenizer-and-tf-2-0-in-python/
# 
# HuggingFace has an awesome documentation. Do read the documentation to understand various concepts and code below.
# * https://huggingface.co/transformers/quickstart.html
# 
# **Acknowledgements**
# * https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
# * https://www.kaggle.com/tanulsingh077/deep-learning-for-nlp-zero-to-transformers-bert (Do give this notebook a read as @Mr_KnowNothing has curated a list of amazing articles and tutorials for anyone who is getting started with NLP and his notebook has given me the headstart to this competition.)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import transformers
from tokenizers import BertWordPieceTokenizer
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam


# # TPU Config

# For understanding TPU refer to following links:
# * https://www.tensorflow.org/guide/tpu
# * https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/TPUStrategy
# * https://www.tensorflow.org/guide/data_performance
# 

# In[ ]:


#Detect hardware, return appropriate distribution strategy
try:
   # TPU detection. No parameters necessary if TPU_NAME environment variable is
   # set: this is always the case on Kaggle.
   tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
   print('Running on TPU ', tpu.master())
except ValueError:
   tpu = None

if tpu:
   tf.config.experimental_connect_to_cluster(tpu)
   tf.tpu.experimental.initialize_tpu_system(tpu)
   strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
   # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
   strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[ ]:


#IMP DATA FOR CONFIG

AUTO = tf.data.experimental.AUTOTUNE


# Configuration
EPOCHS = 3
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 192


# In[ ]:


def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    '''
    Function for fast encoding
    '''
    all_ids = []
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = list(texts[i:chunk_size+i])
        encs = tokenizer.batch_encode_plus(text_chunk, max_length=maxlen, pad_to_max_length = True)
        all_ids.extend(encs['input_ids'])
        
    return np.array(all_ids)


# In[ ]:


train = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
validation = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/test.csv')


# # Tokenization

# In[ ]:


# Refer the HuggingFace Documention
fast_tokenizer = transformers.DistilBertTokenizerFast.from_pretrained('distilbert-base-multilingual-cased')


# In[ ]:


x_train = fast_encode(train.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_valid = fast_encode(validation.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_test = fast_encode(test.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)


# In[ ]:


y_train = train.toxic.values
y_valid = validation.toxic.values


# # Creating Data Objects
# 
# To understand how to create the data objects refer to the links provided under TPU config section

# In[ ]:


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train,y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid,y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)


# # Building the model to start the training

# In[ ]:


def build_model(transformer, maxlen=512):
    input_word_ids = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32, name='input_word_ids')
    sequence_output = transformer(input_word_ids)[0]
    
    clf_output = sequence_output[:,0,:]
    out = tf.keras.layers.Dense(1, activation='sigmoid')(clf_output)
    
    model = tf.keras.models.Model(inputs = input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:


with strategy.scope():
    transformer_layer = (
        transformers.TFDistilBertModel
        .from_pretrained('distilbert-base-multilingual-cased')
    )
    model = build_model(transformer_layer, maxlen=MAX_LEN)
    
model.summary()


# ## Training on English Data

# In[ ]:


n_steps = x_train.shape[0]//BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch = n_steps,
    validation_data = valid_dataset,
    epochs = EPOCHS
)


# Since we have trained the model on training data, now we will train the model on validation data because it contains comments in different languages.

# In[ ]:


n_steps_valid = x_valid.shape[0]//BATCH_SIZE
valid_history = model.fit(
    valid_dataset.repeat(),
    steps_per_epoch = n_steps_valid,
    epochs = EPOCHS*2
)


# # Predicting the results

# In[ ]:


sub = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
sub['toxic'] = model.predict(test_dataset, verbose=1)
sub.to_csv('submission.csv', index=False )

