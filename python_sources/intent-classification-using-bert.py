#!/usr/bin/env python
# coding: utf-8

# <font face = "Verdana" size ="5">intent classification is  very important process  in developing dialog system in nlp it is  core process  of all voice assistant developed using nlp.earlier spacy ,svm ,naive bayes ,lstm ,cnnseq2seq models  were used for creating intent classification model .transformer models  have created benchmark in nlp process ,lets  see how it works  in intent classification
#  
#    <font face = "Verdana" size ="4">
#    <br>Feel free to provide me with feedbacks. 
#    
#     
# 

# In[ ]:


get_ipython().system('pip install transformers')


# In[ ]:


import os
import math
import datetime

from tqdm import tqdm

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras


import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc

from sklearn.metrics import confusion_matrix, classification_report

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
from transformers import BertTokenizer, TFBertForSequenceClassification


# In[ ]:


def regular_encode(texts, tokenizer, maxlen=256):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return np.array(enc_di['input_ids' ])


# # Intent Recognition with BERT using Keras and TensorFlow 2

# In[ ]:


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

# Data access
#GCS_DS_PATH = KaggleDatasets().get_gcs_path()

# Configuration
EPOCHS = 1
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
MAX_LEN = 256


# In[ ]:





# In[ ]:


train = pd.read_csv("/kaggle/input/data/train.csv")
valid = pd.read_csv("/kaggle/input/data/valid.csv")
test = pd.read_csv("/kaggle/input/data/test.csv")


# In[ ]:


train.head()


# In[ ]:


train=train.iloc[:,1:]


# In[ ]:


test=test.iloc[:,1:]

valid=valid.iloc[:,1:]


# In[ ]:


train_y=train['intent'].astype('category').cat.codes


# In[ ]:


valid_y=valid['intent'].astype('category').cat.codes
test_y=test['intent'].astype('category').cat.codes


# In[ ]:


train.drop(columns=['intent'],inplace=True)


# In[ ]:


valid.drop(columns=['intent'],inplace=True)
test.drop(columns=['intent'],inplace=True)


# In[ ]:


train_y=train_y.astype('int64')


# In[ ]:


test_y=test_y.astype('int64')
valid_y=valid_y.astype('int64')


# # Intent Recognition with BERT

# In[ ]:


def build_model(transformer, max_len=100):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output
    logits = keras.layers.Dense(units=768, activation="tanh")(cls_token)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=7, activation="softmax")(logits)

    model = Model(inputs=input_word_ids, outputs=logits)
    model.compile(optimizer=keras.optimizers.Adam(1e-5),loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)
    

    
    
    return model


# ## Training

# 

# In[ ]:


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


# In[ ]:


x_train = regular_encode(train.text	.values, tokenizer, maxlen=MAX_LEN)
x_valid = regular_encode(valid.text	.values, tokenizer, maxlen=MAX_LEN)
x_test = regular_encode(test.text.values, tokenizer, maxlen=MAX_LEN)


# In[ ]:



 
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, train_y.values))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, valid_y.values))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)
test_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_test,test_y.values))
    .batch(BATCH_SIZE)
)


# In[ ]:


from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


# In[ ]:



get_ipython().run_cell_magic('time', '', "with strategy.scope():\n    model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')\n    model = build_model(model, max_len=256)\nmodel.summary()")


# In[ ]:


EPOCHS = 5
n_steps = x_train.shape[0] // 16
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
    
)


# In[ ]:


y_pred = model.predict(x_test).argmax(axis=-1)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_pred,test_y)


# In[ ]:


y_pred


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(test_y, y_pred))


# In[ ]:


predictions = model.predict(x_test).argmax(axis=-1)


# # References
# 
# - https://mccormickml.com/2019/07/22/BERT-fine-tuning/
# - https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines
# - https://jalammar.github.io/illustrated-bert/
# - https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03
# - https://www.reddit.com/r/MachineLearning/comments/ao23cp/p_how_to_use_bert_in_kaggle_competitions_a/
