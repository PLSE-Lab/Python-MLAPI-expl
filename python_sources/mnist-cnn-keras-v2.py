#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Tensorflow 2.0
import tensorflow as tf 

import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

# Check that we are using a GPU, if not switch runtimes
#   using Runtime > Change Runtime Type > GPU
assert len(tf.config.list_physical_devices('GPU')) > 0


# In[ ]:


import os
os.listdir('../input/digit-recognizer')


# In[ ]:


import pandas as pd

base = '../input/digit-recognizer/'
train_df = pd.read_csv(base+'train.csv')
test_df = pd.read_csv(base+'test.csv')

# df.head(1)


# In[ ]:


df = pd.concat([train_df, test_df])
cols = list(df.columns)
cols.remove('label')
df[cols] = df[cols] / 255.0
df.shape


# In[ ]:


train = df.iloc[:len(train_df)]
test = df.iloc[len(train_df):]
test.drop(columns=['label'], inplace=True)

X_train = train.drop(columns=['label'])
Y_train = train['label']
X_train = X_train.values.reshape(-1,28,28,1).astype(np.float32)
X_test = test.values.reshape(-1,28,28,1).astype(np.float32)


# In[ ]:


from sklearn.model_selection import train_test_split

X_sample, X_val, Y_sample, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state = 22)


# ## CNN MODELING
# #### Using Maxpooling

# In[ ]:


def build_cnn_model():
    cnn_model = tf.keras.Sequential([
        
#         tf.keras.layers.Flatten(),
        tf.keras.layers.Conv2D(filters=24, kernel_size=(3,3), activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(filters=36, kernel_size=(3,3), activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        
    ])
    
    return cnn_model

cnn_model = build_cnn_model()
cnn_model.predict(X_sample[:3])
print(cnn_model.summary())


# In[ ]:


cnn_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

BATCH_SIZE = 64

cnn_model.fit(X_sample, Y_sample, batch_size = BATCH_SIZE, epochs=20)


# In[ ]:


test_loss, test_acc = cnn_model.evaluate(X_val, Y_val, batch_size=BATCH_SIZE)
print("Test Accuracy: {:.4f}".format(test_acc))


# In[ ]:


predictions = cnn_model.predict(X_test)
predictions[1]


# In[ ]:


predictions = [np.argmax(predictions[i]) for i in range(len(predictions))]
predictions[:3]


# In[ ]:


sample = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
sample.head(3)


# In[ ]:


sample['Label'] = predictions
sample.head(2)


# In[ ]:


sample.to_csv('CNN_2layers_Adagrad2.csv', index=False)


# In[ ]:




