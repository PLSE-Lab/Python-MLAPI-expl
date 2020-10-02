#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# Hello All,
# 
# I'm a beginner in NLP and machine learning that I have recently started to explore. I have learned Keras and tensorflow and I wanted to impliment a bert kernal in keras, so I found out this github repository https://github.com/CyberZHG/keras-bert which is also mentioned in https://www.kaggle.com/httpwwwfszyc/bert-keras-with-warmup-and-excluding-wd-parameters/notebook, I packed this in my dataset https://www.kaggle.com/gauravs90/keras-bert-by-cyberzhg-github, I have to create this dataset as existing ones did not include the tokenizer and AdamWarmup.
# 
# Thanks for Jon Mischo (https://www.kaggle.com/supertaz) for uploading BERT Models + Scripts :)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Copy All the files to working directory**

# In[ ]:


get_ipython().system("cp -r '../input/keras-bert-by-cyberzhg-github/keras_bert/keras_bert' '/kaggle/working'")
get_ipython().system("cp -r '../input/keras-bert-by-cyberzhg-github/keras_bert/keras_pos_embd' '/kaggle/working'")
get_ipython().system("cp -r '../input/keras-bert-by-cyberzhg-github/keras_bert/keras_embed_sim' '/kaggle/working'")
get_ipython().system("cp -r '../input/keras-bert-by-cyberzhg-github/keras_bert/keras_position_wise_feed_forward' '/kaggle/working'")
get_ipython().system("cp -r '../input/keras-bert-by-cyberzhg-github/keras_bert/keras_layer_normalization' '/kaggle/working'")
get_ipython().system("cp -r '../input/keras-bert-by-cyberzhg-github/keras_bert/keras_self_attention' '/kaggle/working'")
get_ipython().system("cp -r '../input/keras-bert-by-cyberzhg-github/keras_bert/keras_multi_head' '/kaggle/working'")
get_ipython().system("cp -r '../input/keras-bert-by-cyberzhg-github/keras_bert/keras_transformer' '/kaggle/working'")


# In[ ]:


import tensorflow as tf
import keras as keras
import keras.backend as K
from keras.models import load_model

from keras_bert import load_trained_model_from_checkpoint, load_vocabulary
from keras_bert import Tokenizer
from keras_bert import AdamWarmup, calc_train_steps

import pandas as pd
import numpy as np
from tqdm import tqdm
import gc


# In[ ]:


SEQ_LEN = 64
BATCH_SIZE = 128
EPOCHS = 1
LR = 1e-4

pretrained_path = '../input/pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

DATA_COLUMN = 'comment_text'
LABEL_COLUMN = 'target'


# **Get Tokenizer**

# In[ ]:


token_dict = load_vocabulary(vocab_path)
tokenizer = Tokenizer(token_dict)


# **Load and Convert to data that BERT understand**

# In[ ]:


def convert_data(data_df):
    global tokenizer
    indices, targets = [], []
    for i in tqdm(range(len(data_df))):
        ids, segments = tokenizer.encode(data_df[DATA_COLUMN][i], max_len=SEQ_LEN)
        indices.append(ids)
        targets.append(data_df[LABEL_COLUMN][i])
    items = list(zip(indices, targets))
    np.random.shuffle(items)
    indices, targets = zip(*items)
    indices = np.array(indices)
    return [indices, np.zeros_like(indices)], np.array(targets)


# In[ ]:


def load_data(path):
    data_df = pd.read_csv(path, nrows=10000)
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_x, data_y = convert_data(data_df)
    return data_x, data_y


# In[ ]:


train_x, train_y = load_data('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
gc.collect()


# **Load Model from Checkpoint**

# In[ ]:


model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training=True,
    trainable=True,
    seq_len=SEQ_LEN,
)


# **Create Keras Model and compile it**

# In[ ]:


inputs = model.inputs[:2]
dense = model.layers[-3].output
outputs = keras.layers.Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                             name = 'real_output')(dense)

decay_steps, warmup_steps = calc_train_steps(
    train_y.shape[0],
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
)

model = keras.models.Model(inputs, outputs)
model.compile(
    AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=LR),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# In[ ]:


model.summary()


# In[ ]:


sess = K.get_session()
uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
init_op = tf.variables_initializer(
    [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables]
)
sess.run(init_op)


# In[ ]:


model.fit(
        train_x,
        train_y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )


# **Load the test data**

# In[ ]:


def convert_test(test_df):
    global tokenizer
    indices = []
    for i in tqdm(range(len(test_df))):
        ids, segments = tokenizer.encode(test_df[DATA_COLUMN][i], max_len=SEQ_LEN)
        indices.append(ids)
    indices = np.array(indices)
    return [indices, np.zeros_like(indices)]

def load_test(path):
    data_df = pd.read_csv(path, nrows=5000)
    data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
    data_x = convert_test(data_df)
    return data_x


# In[ ]:


test_x = load_test('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
gc.collect()


# **Predict and Submit**

# In[ ]:


prediction = model.predict(test_x)
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id', nrows=5000)
submission['prediction'] = prediction
submission.reset_index(drop=False, inplace=True)
submission.to_csv('submission.csv', index=False)

