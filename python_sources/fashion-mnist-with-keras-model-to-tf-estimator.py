#!/usr/bin/env python
# coding: utf-8

# Copyright 2016 Google Inc. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# --------------------------------------

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
print(tf.__version__)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().system('python -V')


# In[3]:


data_train_file = "../input/fashion-mnist_train.csv"
data_test_file = "../input/fashion-mnist_test.csv"

df_train = pd.read_csv(data_train_file)
df_test = pd.read_csv(data_test_file)


# In[4]:


df_train.head()


# ## Preprocessing the dataset

# In[5]:


# Note this returns numpy arrays
def get_features_labels(df):
    # Select all columns but the first
    features = df.values[:, 1:]/255
    # The first column is the label. Conveniently called 'label'
    labels = df['label'].values
    return features, labels


# In[6]:


train_features, train_labels = get_features_labels(df_train)
test_features, test_labels = get_features_labels(df_test)


# ## One-hot encoding

# In[25]:


print(train_features.shape)
print(train_labels.shape)


# In[8]:


train_labels_1hot = tf.keras.utils.to_categorical(train_labels)
test_labels_1hot = tf.keras.utils.to_categorical(test_labels)


# In[9]:


print(train_labels_1hot.shape)
print(test_labels_1hot.shape)


# ## Training Parameters

# In[14]:


BATCH_SIZE=128
EPOCHS=2


# ## Create a Keras Model

# In[39]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu, input_shape=(784,)))
model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


# Create a TensorFlow optimizer, rather than using the Keras version
# This is currently necessary when working in eager mode
# optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

# We will now compile and print out a summary of our model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()


# In[40]:


model.fit(train_features, train_labels_1hot, epochs=EPOCHS, batch_size=BATCH_SIZE)


# In[41]:


test_loss, test_acc = model.evaluate(test_features, test_labels_1hot)
print('test_acc:', test_acc)


# ## Convert Keras model to TensorFlow estimator
# Use `model_to_estimator`

# In[42]:


tf_classifier = tf.keras.estimator.model_to_estimator(keras_model=model)


# ## Train TensorFlow model
# This is essentially the same code as original

# In[43]:


model.input_names
# Use this name as the dictionary key in the TF input function


# In[44]:


input_name = model.input_names[0]
input_name


# In[45]:


train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={input_name: train_features},
        y=train_labels_1hot,
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        shuffle=True)


# In[46]:


# We again use the same name from the keras model.input_names
feature_columns = [tf.feature_column.numeric_column(input_name, shape=784)]


# In[47]:


classifier = tf_classifier
# .estimator.LinearClassifier(
#                 feature_columns=feature_columns, 
#                 n_classes=10,
#                 model_dir="./"
#                 )


# In[48]:


classifier.train(input_fn=train_input_fn)


# In[49]:


evaluate_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={input_name: test_features},
        y=test_labels_1hot,
        num_epochs=1,
        shuffle=False)


# In[50]:


classifier.evaluate(input_fn=evaluate_input_fn)["accuracy"]


# ## Model Export
# Exporting our Keras and TF models

# In[51]:


model.save('keras_model.h5')


# In[35]:


feature_spec = {
    input_name: tf.FixedLenFeature(shape=[784], dtype=np.float32)
}
serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
export_dir = classifier.export_savedmodel(export_dir_base="models/export", 
                            serving_input_receiver_fn=serving_fn)
export_dir = export_dir.decode("utf8")


# In[36]:


export_dir


# In[37]:


get_ipython().system('ls {export_dir}')


# In[38]:


get_ipython().system('tar -zcvf exported_model.tgz {export_dir}')


# Model is zipped and ready for download. To unzip, run
# 
# `tar -zxvf exported_model.tgz`

# In[ ]:




