#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os, time
import tensorflow as tf
import math
from transformers import TFXLMRobertaModel
from tensorflow.keras.optimizers import Adam
import os
from kaggle_datasets import KaggleDatasets
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# There I will build model upon xlm-roberta using techniques to fight unbalanced classes, as I've shown there:
# https://www.kaggle.com/vgodie/class-balancing
# 
# Using preprocessed data from this my notebook
# 
# https://www.kaggle.com/vgodie/data-encoding

# In[ ]:


#set TPU coniguration
AUTO = tf.data.experimental.AUTOTUNE

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
print(tpu_strategy.num_replicas_in_sync)


# In[ ]:


MY_GCS_PATH=KaggleDatasets().get_gcs_path('dataencoding')


# In[ ]:


data_path = "../input/dataencoding/"


train_ids = np.load(os.path.join(data_path, "ids.npy"))
train_labels = np.load(os.path.join(data_path, "labels.npy")).astype(int)
val_ids = np.load(os.path.join(data_path, "val_ids.npy"))
val_labels = np.load(os.path.join(data_path, "val_labels.npy"))


# In[ ]:


MODEL = "jplu/tf-xlm-roberta-large"
SEQUENCE_LENGTH = 192
BATCH_SIZE =  16 * tpu_strategy.num_replicas_in_sync


# In[ ]:


pos = train_ids[np.where(train_labels == 1)[0]]
neg = train_ids[np.where(train_labels == 0)[0]]

pos_labels = train_labels[np.where(train_labels==1)[0]]
neg_labels = train_labels[np.where(train_labels==0)[0]]


# In[ ]:


def make_ds(features, labels):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.shuffle(1500000).repeat()
    return ds


# In[ ]:





# In[ ]:


del train_ids
del train_labels


# Make resampled dataset to ensure that classes are balanced in training batches

# In[ ]:


pos_ds = make_ds(pos, pos_labels)
neg_ds = make_ds(neg, neg_labels)

resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(AUTO)


# In[ ]:


del pos
del neg


# In[ ]:


val_dataset = (tf.data.Dataset.from_tensor_slices((val_ids, val_labels))
               .shuffle(len(val_ids))
               .repeat()
               .batch(BATCH_SIZE)
               .prefetch(AUTO)
              )


# Making model taking sentence emdedding as concatenation of max and average pooling

# In[ ]:


def make_model(embed_model):
    
    
    input_ids = tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,), name='input_token', dtype='int32')

    embed_layer = embed_model([input_ids])[0]
    avg_pool = tf.reduce_mean(embed_layer, axis=1)
    max_pool = tf.reduce_max(embed_layer, axis=1)
    X = tf.concat([avg_pool, max_pool], axis=1)
    X = tf.keras.layers.Dropout(0.3)(X)
    X = tf.keras.layers.Dense(1, activation="sigmoid")(X)
    model = tf.keras.Model(inputs=input_ids, outputs = X)
    return model


# In[ ]:


with tpu_strategy.scope():
    xlm_roberta = TFXLMRobertaModel.from_pretrained(MODEL)
    xr_model = make_model(xlm_roberta)
    xr_model.summary()
    xr_model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-5), loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1), metrics=[tf.keras.metrics.AUC()])


# In[ ]:


N_STEPS = 1200000//BATCH_SIZE
N_STEPS

VAL_STEPS = val_ids.shape[0]//BATCH_SIZE


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

def build_lrfn(lr_start=0.000001, lr_max=0.000002, 
               lr_min=0.0000001, lr_rampup_epochs=7, 
               lr_sustain_epochs=0, lr_exp_decay=.87):
    lr_max = lr_max * tpu_strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    
    return lrfn

lrfn = build_lrfn()


es = EarlyStopping(monitor='val_accuracy', mode='max', patience=2, 
                   restore_best_weights=True, verbose=1)
lr_callback = LearningRateScheduler(lrfn, verbose=1)
callbacks = [es, lr_callback]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = xr_model.fit(resampled_ds,\n                       validation_data=val_dataset,\n                       epochs=2,\n                       steps_per_epoch=N_STEPS,\n                       validation_steps = VAL_STEPS,\n                       callbacks=callbacks\n                      )')


# In[ ]:


val_history = xr_model.fit(val_dataset,
                       epochs=2,
                       steps_per_epoch=VAL_STEPS,
                       callbacks=callbacks
                      )


# In[ ]:


xr_model.save_weights("weights.h5")


# In[ ]:


sub = pd.read_csv(os.path.join('../input/jigsaw-multilingual-toxic-comment-classification/','sample_submission.csv'))
test_ids = np.load("../input/dataencoding/test_ids.npy", allow_pickle=True)
sub['toxic'] = xr_model.predict(test_ids, verbose=1)
sub.to_csv('submission.csv', index=False)

