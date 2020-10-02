#!/usr/bin/env python
# coding: utf-8

# 
# # Notes:
# 
# **Kaggle Sources**
#  - https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta
#      - forked this notebook
#  - https://www.kaggle.com/shonenkov/tpu-training-super-fast-xlmroberta
#      - grabed external data from this notebook, i.e. https://www.kaggle.com/shonenkov/open-subtitles-toxic-pseudo-labeling
#      
# **External Sources and More Additions**
#  - Used [eda_nlp](https://github.com/jasonwei20/eda_nlp) to create an augmented version of the unintended bias dataset, then downsampled this data to have balanced dataset.  More information can be found here (minus the downsampling):  https://www.kaggle.com/yeayates21/jigsaw-bias-toxicity-eda-nlp-aug16-alpha005
#  - pickled encoded data for faster runtime
#  - some light manual hyperparameter tuning
#  - scored the test set with each "model.fit" run as "checkpoint predictions" and blended the checkpoint predictions (I didn't checkpoint the models, but that could be easily added).
# 
# -----------------------------------------------------------------------
# 
# #### Acknowledgements
# 
#  - [@alexshonenkov](https://www.kaggle.com/shonenkov)
#  - [@xhlulu](https://www.kaggle.com/xhlulu)
# 

# In[ ]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
import pickle


# # Helper Functions

# In[ ]:


def build_model(transformer, max_len=512):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=0.000009), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# # Configs

# In[ ]:


# Detect hardware, return appropriate distribution strategy
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


AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 192
MODEL = 'jplu/tf-xlm-roberta-large'


# # Tokenizer

# In[ ]:


# First load the real tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)


# # Load Data

# In[ ]:


get_ipython().system('ls /kaggle/input/jigsawtpuxlmrobertacopypickledata')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfilename = "/kaggle/input/jigsawtpuxlmrobertacopypickledata/jigsaw_multilingual_x_train.pkl"\nx_train = pickle.load(open(filename, \'rb\')) # load data example\nfilename = "/kaggle/input/jigsawtpuxlmrobertacopypickledata/jigsaw_multilingual_x_trainOA.pkl"\nx_trainOA = pickle.load(open(filename, \'rb\')) # load data example\nfilename = "/kaggle/input/jigsawtpuxlmrobertacopypickledata/jigsaw_multilingual_x_trainA.pkl"\nx_trainA = pickle.load(open(filename, \'rb\')) # load data example\nfilename = "/kaggle/input/jigsawtpuxlmrobertacopypickledata/jigsaw_multilingual_x_valid.pkl"\nx_valid = pickle.load(open(filename, \'rb\')) # load data example\nfilename = "/kaggle/input/jigsawtpuxlmrobertacopypickledata/jigsaw_multilingual_x_test.pkl"\nx_test = pickle.load(open(filename, \'rb\')) # load data example\n\nfilename = "/kaggle/input/jigsawtpuxlmrobertacopypickledata/jigsaw_multilingual_y_train.pkl"\ny_train = pickle.load(open(filename, \'rb\')) # load data example\nfilename = "/kaggle/input/jigsawtpuxlmrobertacopypickledata/jigsaw_multilingual_y_trainOA.pkl"\ny_trainOA = pickle.load(open(filename, \'rb\')) # load data example\nfilename = "/kaggle/input/jigsawtpuxlmrobertacopypickledata/jigsaw_multilingual_y_trainA.pkl"\ny_trainA = pickle.load(open(filename, \'rb\')) # load data example\nfilename = "/kaggle/input/jigsawtpuxlmrobertacopypickledata/jigsaw_multilingual_y_valid.pkl"\ny_valid = pickle.load(open(filename, \'rb\')) # load data example')


# # TF Datasets

# In[ ]:


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

train_datasetOA = (
    tf.data.Dataset
    .from_tensor_slices((x_trainOA, y_trainOA))
    .repeat()
    .shuffle(x_trainOA.shape[0])
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

train_datasetA = (
    tf.data.Dataset
    .from_tensor_slices((x_trainA, y_trainA))
    .repeat()
    .shuffle(x_trainOA.shape[0])
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)


# # Load Model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'with strategy.scope():\n    transformer_layer = TFAutoModel.from_pretrained(MODEL)\n    model = build_model(transformer_layer, max_len=MAX_LEN)\nmodel.summary()')


# # Train

# #### Train on English training data

# In[ ]:


n_steps = x_train.shape[0] // BATCH_SIZE

model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=1
)


# In[ ]:


checkpointPredictions1 = model.predict(test_dataset, verbose=1)
print(checkpointPredictions1[:10])


# #### Train on augmented english data

# In[ ]:


n_steps = x_trainA.shape[0] // BATCH_SIZE

model.fit(
    train_datasetA,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=1
)


# In[ ]:


checkpointPredictions2 = model.predict(test_dataset, verbose=1)
print(checkpointPredictions2[:10])


# #### Train on multilingual validation training data

# In[ ]:


n_steps = x_valid.shape[0] // BATCH_SIZE

model.fit(
    valid_dataset.repeat(),
    steps_per_epoch=n_steps,
    epochs=2
)


# In[ ]:


checkpointPredictions3 = model.predict(test_dataset, verbose=1)
print(checkpointPredictions3[:10])


# #### Train on multilingual external data (created using SSL techniques)

# In[ ]:


n_steps = x_trainOA.shape[0]  // BATCH_SIZE

model.fit(
    train_datasetOA,
    steps_per_epoch=n_steps,
    epochs=1
)


# In[ ]:


checkpointPredictions4 = model.predict(test_dataset, verbose=1)
print(checkpointPredictions4[:10])


# # Submission

# In[ ]:


sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
sub['toxic'] = (checkpointPredictions1*0.05)+(checkpointPredictions2*0.10)+(checkpointPredictions3*0.76)+(checkpointPredictions4*0.09)
sub.to_csv('submission.csv', index=False)


# In[ ]:




