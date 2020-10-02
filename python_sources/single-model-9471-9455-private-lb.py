#!/usr/bin/env python
# coding: utf-8

# Single model

# No K-flod

# No external data except the translated data

# training step from this notebook https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta

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
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
import time
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input,GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors


# first , we should align the translated data so that we can reduce duplication when randomly drawing.It is easy to do.We just need to align the 'id'. It takes 20s on my pc,but 2.5h in this kernal.

# In[ ]:


def align_data():
    en=pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
    f1=pd.read_csv('../input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-es-cleaned.csv')#translated data that we want to align

    result=[]
    last_cell=[]
    for i in tqdm(range(len(en['id']))):
        index=np.where(f1['id']==en['id'][i])[0]
        l=len(index)
        if l==1:
            cell=[f1['comment_text'][index[0]],'es',f1['toxic'][index[0]]]
            result.append(cell)
            last_cell=cell
        else:
            cell=last_cell
            result.append(cell)   
    print(len(en['toxic']))
    print(len(result))
    column=['comment_text','lang','toxic']
    toxic=pd.DataFrame(columns=column,data=result)
    toxic.to_csv('/kaggle/working/train1_es_align.csv'%(LANG,LANG),index=False)


# After that , we make all translated data have the same amount : 223549 (named train1) . And unintended_bias amount : 1902194 (named train2_max).

# Then we can mix them in a certain ratio . For every row , we choose one from 6 files randomly.

# In[ ]:


def mix_data():
    #ratio for every language 
    ES_FRAC=0.132232182034727
    TR_FRAC=0.21939447125932426
    RU_FRAC=0.17156647652479157
    IT_FRAC=0.13310975991976431
    FR_FRAC=0.17112768758227292
    PT_FRAC=0.17256942267911993

    train_tr=pd.read_csv('../input/jigsaw_toxic_comment_train_align/train1_tr.csv.csv')
    train_es=pd.read_csv('../input/jigsaw_toxic_comment_train_align/train1_es.csv.csv')
    train_it=pd.read_csv('../input/jigsaw_toxic_comment_train_align/train1_it.csv.csv')
    train_fr=pd.read_csv('../input/jigsaw_toxic_comment_train_align/train1_fr.csv.csv')
    train_ru=pd.read_csv('../input/jigsaw_toxic_comment_train_align/train1_ru.csv.csv')
    train_pt=pd.read_csv('../input/jigsaw_toxic_comment_train_align/train1_pt.csv.csv')

    result=[]
    label=[]
    for i in tqdm(range(0, len(train_es['comment_text']))):
    rf = random.random()
    if rf < ES_FRAC:
        cell = [train_es['comment_text'][i], train_es['lang'][i], train_es['toxic'][i]]
        label.append(cell[2])
        result.append(cell)
    elif rf < (ES_FRAC + FR_FRAC):
        cell = [train_fr['comment_text'][i], train_fr['lang'][i], train_fr['toxic'][i]]
        label.append(cell[2])
        result.append(cell)
    elif rf < (ES_FRAC + FR_FRAC + TR_FRAC):
        cell = [train_tr['comment_text'][i], train_tr['lang'][i], train_tr['toxic'][i]]
        label.append(cell[2])
        result.append(cell)
    elif rf < (ES_FRAC + FR_FRAC + TR_FRAC + IT_FRAC):
        cell = [train_it['comment_text'][i], train_it['lang'][i], train_it['toxic'][i]]
        label.append(cell[2])
        result.append(cell)
    elif rf < (ES_FRAC + FR_FRAC + TR_FRAC + IT_FRAC + PT_FRAC):
        cell = [train_pt['comment_text'][i], train_pt['lang'][i], train_pt['toxic'][i]]
        label.append(cell[2])
        result.append(cell)
    else:
        cell = [train_ru['comment_text'][i], train_ru['lang'][i], train_ru['toxic'][i]]
        label.append(cell[2])
        result.append(cell)
    print(len(result))
    print(len(label))

    label=np.array(label)
    np.save('/kaggle/working/train1_label.npy')
    column=['comment_text','lang','toxic']
    toxic=pd.DataFrame(columns=column,data=result)
    toxic.to_csv('/kaggle/working/train1_mix.csv')
    x_train=regular_encode(toxic['comment_text'], tokenizer, maxlen=MAX_LEN)
    np.save('/kaggle/working/train1_mix.npy')


# After encode we get a data with 6 laguages.   Because it is generated randomly.   we can do it again and again to get a different data.

# omit the encoding step

# Then we use the encoded data for training

# In[ ]:


def build_model(transformer, max_len=512,learnig_rate=1e-5):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    #x = tf.keras.layers.Dropout(0.2)(cls_token)
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=learnig_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# In[ ]:


AUTO = tf.data.experimental.AUTOTUNE

# Data access
GCS_DS_PATH = KaggleDatasets().get_gcs_path("jigsaw-multilingual-toxic-comment-classification")

# Configuration
EPOCHS = 3
BATCH_SIZE = 24 * strategy.num_replicas_in_sync
MAX_LEN = 192
MODEL = 'jplu/tf-xlm-roberta-large'


# In[ ]:


valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
#valid = pd.read_csv('/kaggle/input/trans-tr/validation_tr.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')


# In[ ]:


x_valid=np.load('/kaggle/input/toxic-npy/x_valid_encode192.npy')
print(x_valid)
print(x_valid.shape)

y_valid = valid.toxic.values
print(y_valid)
print(y_valid.shape)


# In[ ]:


x_test=np.load('/kaggle/input/toxic-npy/x_test_encode192.npy')
#x_test=np.load('/kaggle/input/trans-tr/test_tr_data.npy')
print(x_test)
print(x_test.shape)
print("finish x_test_encode")


# In[ ]:


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


# train2max ( shape(1902194,192) ) first .

# In[ ]:


def train_train2max():
    print('train train2max')

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


    x_train=np.load('/kaggle/input/mix-data2/train2max21.npy')


    y_train=np.load('/kaggle/input/mix-data2/train2max_label21.npy')
    y_train=y_train.round().astype(int)


    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_train, y_train))
        .repeat()
        .batch(BATCH_SIZE)
        .shuffle(2048)
        .prefetch(AUTO)
    )


    with strategy.scope():
        transformer_layer = TFAutoModel.from_pretrained(MODEL)
        model = build_model(transformer_layer, max_len=MAX_LEN,learnig_rate=5e-6)
    model.summary()



    n_steps = x_train.shape[0] // BATCH_SIZE
    #n_steps = 680
    train_history = model.fit(
        train_dataset,
        steps_per_epoch=n_steps,
        validation_data=valid_dataset,
        epochs=1
    )

    model.save_weights('/kaggle/working/model_mix_1.h5')


# Then train1(shape(223549,192))

# In[ ]:


def train_train1():
    print('train train1')
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

    x_train=np.load('/kaggle/input/mix-data2/train1_mix_data21.npy')


    y_train=np.load('/kaggle/input/mix-data2/train1_label21.npy')
    print(y_train)
    print(y_train.shape)

    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_train, y_train))
        .repeat()
        .batch(BATCH_SIZE)
        .shuffle(2048)
        .prefetch(AUTO)
    )

    with strategy.scope():
        transformer_layer = TFAutoModel.from_pretrained(MODEL)
        model = build_model(transformer_layer, max_len=MAX_LEN,learnig_rate=5e-6)


    model.load_weights('/kaggle/working/model_mix_1.h5')
    print("learning rate:5e-6")
    n_steps = x_train.shape[0] // BATCH_SIZE
    #n_steps = 680
    train_history = model.fit(
        train_dataset,
        steps_per_epoch=n_steps,
        validation_data=valid_dataset,
        epochs=1
    )



    model.save_weights('/kaggle/working/model_mix_1.h5')


# Finally train on valid dataset

# In[ ]:


def train_valid():
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

    with strategy.scope():
        transformer_layer = TFAutoModel.from_pretrained(MODEL)
        model = build_model(transformer_layer, max_len=MAX_LEN,learnig_rate=4e-6)


    model.load_weights('/kaggle/working/model_mix_1.h5')

    n_steps = x_valid.shape[0] // BATCH_SIZE
    train_history_2 = model.fit(
        valid_dataset.repeat(),
        steps_per_epoch=n_steps,
        epochs=2
    )


# In[ ]:


def predict():
    score = model.predict(test_dataset,verbose=1)
    sub['toxic']=score[:sub['toxic'].shape[0]]
    sub.to_csv('submission.csv', index=False)


# train_train2max() -> train_train1() -> train_valid() -> predict()  is a complete process. Every time we can get a submissiom. After repeating it 20 times. We can get 20 submissions

# In[ ]:


#train_train2max()
#train_train1()
#train_valid()
#predict()


# Then we can sum them up , and get the mean

# In[ ]:


submission_train2mix1 = pd.read_csv('/kaggle/input/haveatry/submission9450.csv')
submission_train2mix2 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_2.csv')
submission_train2mix3 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_3.csv')
submission_train2mix4 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_4.csv')
submission_train2mix5 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_5.csv')
submission_train2mix6 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_6.csv')
submission_train2mix7 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_7.csv')
submission_train2mix8 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_8.csv')
submission_train2mix9 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_9.csv')
submission_train2mix10 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_10.csv')
submission_train2mix11 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_11.csv')
submission_train2mix12 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_12.csv')
submission_train2mix13 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_13.csv')
submission_train2mix14 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_14.csv')
submission_train2mix15 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_15.csv')
submission_train2mix16 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_16.csv')
submission_train2mix17 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_17.csv')
submission_train2mix18 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_18.csv')
submission_train2mix19 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_19.csv')
submission_train2mix20 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_20.csv')
submission_train2mix21 = pd.read_csv('/kaggle/input/haveatry/submission_train2mix_21.csv')


# In[ ]:


for i in range(len(submission_train2mix1['toxic'])):
    sub['toxic'][i]=(submission_train2mix1['toxic'][i]+submission_train2mix2['toxic'][i]
                     +submission_train2mix3['toxic'][i]+submission_train2mix4['toxic'][i]
                     +submission_train2mix5['toxic'][i]+submission_train2mix6['toxic'][i]
                     +submission_train2mix7['toxic'][i]+submission_train2mix8['toxic'][i]
                     +submission_train2mix9['toxic'][i]+submission_train2mix10['toxic'][i]
                     +submission_train2mix11['toxic'][i]+submission_train2mix12['toxic'][i]
                     +submission_train2mix13['toxic'][i]+submission_train2mix14['toxic'][i]
                    +submission_train2mix15['toxic'][i]+submission_train2mix16['toxic'][i]
                    +submission_train2mix17['toxic'][i]+submission_train2mix18['toxic'][i]
                    +submission_train2mix19['toxic'][i]+submission_train2mix20['toxic'][i]
                    +submission_train2mix21['toxic'][i])/21


# In[ ]:


sub.to_csv('submission.csv', index=False)#~.9471(Public LB)


# 5times:~.9450(Public LB)

# 12times:~.9464(Public LB)

# 21times:~.9471(Public LB)

# With easy ensemble,it can easily get a sliver medal.For me it's ~.9472(Private) ~.9488(Public)
