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

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import tensorflow as tf

from tensorflow import keras

import numpy as np

import pandas as pd

import tensorflow_hub as hub

from sklearn.model_selection import train_test_split


# In[ ]:


train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")


# In[ ]:


submission.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# In[ ]:


train.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train.text, train.target, test_size=0.2,random_state =42)


# In[ ]:


X_test_lst = test.text.to_list()


# In[ ]:


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))


# In[ ]:


validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))


# In[ ]:


train_examples_batch, train_labels_batch = next(iter(train_dataset.batch(10)))
train_examples_batch


# Make sure "Internet" is on in Kaggle Settings

# In[ ]:


embedding = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=False)
hub_layer(tf.convert_to_tensor(train_examples_batch[:3]))


# In[ ]:


model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()


# In[ ]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[ ]:


history = model.fit(train_dataset.shuffle(10000).batch(512),
                    epochs=40,
                    validation_data=validation_dataset.batch(512),
                    verbose=1)


# In[ ]:


y_pred_list = []
for text in X_test_lst:
  y_pred = model.predict(tf.expand_dims(text, 0))
  y_pred_list.append(y_pred.tolist()[0])


# In[ ]:


y_pred_boolean = tf.greater(y_pred_list, .7)


# In[ ]:


final=[]
for data in y_pred_boolean:
  if data == True:
    final.append(1)
  else:
    final.append(0)


# In[ ]:


output = pd.DataFrame({'id':test['id'],'target':final})


# In[ ]:


output


# In[ ]:


output.to_csv("output.csv",index=False)

