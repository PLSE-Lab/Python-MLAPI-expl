#!/usr/bin/env python
# coding: utf-8

# # Introducion
# 
# This simple script shows how to train a model using Keras framework and TF records format as source. 
# 
# TODO:
# - write inference part of the script
# 
# Note:
# - This is a demo with limited data, you will have to dowload data and run scripts locally
# - You will need need to export the TensorfFow MetaGraph from Keras models to be eligible for ranking.
# 

# In[218]:


from glob import glob
from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf


# In[219]:


def parser(record, training=True):
    """
    In training mode labels will be returned, otherwise they won't be
    """
    keys_to_features = {
        "mean_rgb": tf.FixedLenFeature([1024], tf.float32),
        "mean_audio": tf.FixedLenFeature([128], tf.float32)
    }
    
    if training:
        keys_to_features["labels"] =  tf.VarLenFeature(tf.int64)
    
    parsed = tf.parse_single_example(record, keys_to_features)
    x = tf.concat([parsed["mean_rgb"], parsed["mean_audio"]], axis=0)
    if training:
        y = tf.sparse_to_dense(parsed["labels"].values, [3862], 1)
        return x, y
    else:
        x = tf.concat([parsed["mean_rgb"], parsed["mean_audio"]], axis=0)
        return x


# In[230]:


def make_datasetprovider(tf_records, repeats=1000, num_parallel_calls=12, 
                         batch_size=32): 
    """
    tf_records: list of strings - tf records you are going to use.
    repeats: how many times you want to iterate over the data.
    """
    dataset = tf.data.TFRecordDataset(tf_records)
    dataset = dataset.map(map_func=parser, num_parallel_calls=num_parallel_calls)
    dataset = dataset.repeat(repeats)

    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)

    d_iter = dataset.make_one_shot_iterator()
    return d_iter

def data_generator(tf_records, batch_size=1, repeats=1000, num_parallel_calls=12, ):
    tf_provider = make_datasetprovider(tf_records, repeats=repeats, num_parallel_calls=num_parallel_calls,
                                       batch_size=batch_size)
    sess = tf.Session()
    next_el = tf_provider.get_next()
    while True:
        try:
          yield sess.run(next_el)
        except tf.errors.OutOfRangeError:
            print("Iterations exhausted")
            break
            
def fetch_model():
    model = Sequential()
    model.add(Dense(2048, activation="relu", input_shape=(1024 + 128,)))
    model.add(Dense(3862, activation="sigmoid"))
    model.compile("adam", loss="binary_crossentropy")
    return model


# In[231]:


# TODO: Locally create lists with all TF records
train_data = glob("../input/video/train00.tfrecord")
eval_data = glob("../input/video/train01.tfrecord")

my_train_iter = data_generator(train_data)
my_eval_iter = data_generator(eval_data)
model = fetch_model()


# In[233]:


model.fit_generator(my_train_iter, 
                    steps_per_epoch=30, 
                    epochs=10, 
                    validation_data=my_eval_iter, 
                    validation_steps=20)

