#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install transformers --quiet')
get_ipython().run_line_magic('tensorflow_version', '2.x')


# In[ ]:


import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
from transformers import TFBertModel
import pandas as pd
from transformers.tokenization_bert import BertTokenizer
from sklearn.model_selection import train_test_split

print('TensorFlow:', tf.__version__)


# In[ ]:


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.MirroredStrategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[ ]:


batch_size = 8 * strategy.num_replicas_in_sync
epochs = 1

autotune = tf.data.experimental.AUTOTUNE
pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)


# In[ ]:


with strategy.scope():
   tweets = tf.keras.Input(shape=(512,), dtype=tf.int32)
   bert = TFBertModel.from_pretrained(pretrained_weights)
   tweets_hidden_mean = tf.reduce_mean(bert(tweets)[0], axis=1)
   x = Dense(units=64)(tweets_hidden_mean)
   logits = Dense(units=1, name='logits', activation='sigmoid')(x)
   model = tf.keras.Model(inputs=[tweets], outputs=[logits])


# In[ ]:


model.summary()


# In[ ]:


df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv') # Change this when running in GCP or Colab

encoded_text = [tokenizer.encode(text, max_length=512, pad_to_max_length=True) for text in df['text']]
labels = df['target']
df.head()


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(encoded_text, labels, test_size=0.15)
len(x_train), len(x_test)


# In[ ]:


def make_tfdataset(x, y):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.shuffle(512)
    dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(autotune)
    return dataset


# In[ ]:


with strategy.scope():
    train_dataset = make_tfdataset(x_train, y_train)
    test_dataset = make_tfdataset(x_test, y_test)


# In[ ]:


with strategy.scope():
    metrics_list = [tf.metrics.BinaryAccuracy(), tf.metrics.Precision(), tf.metrics.Recall()]
    model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=False),
                  metrics=metrics_list,
                  optimizer=tf.optimizers.Adam(1e-4))
model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, steps_per_epoch=2)  # change steps_per_epoch when training

