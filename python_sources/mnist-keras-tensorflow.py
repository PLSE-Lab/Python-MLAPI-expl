#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import tensorflow as tf


# In[ ]:


df_train = pd.read_csv("../input/digit-recognizer/train.csv")
df_test = pd.read_csv("../input/digit-recognizer/test.csv")


# In[ ]:


def initial_observation(df):
    if isinstance(df, pd.DataFrame):
        total_na = df.isna().sum().sum()
        print("Dimensions : %d rows, %d columns" % (df.shape[0], df.shape[1]))
        print("Total NA Values : %d " % (total_na))
        print("%38s %10s     %10s %10s" % ("Column Name", "Data Type", "#Distinct", "NA Values"))
        col_name = df.columns
        dtyp = df.dtypes
        uniq = df.nunique()
        na_val = df.isna().sum()
        for i in range(len(df.columns)):
            print("%38s %10s   %10s %10s" % (col_name[i], dtyp[i], uniq[i], na_val[i]))
        
    else:
        print("Expect a DataFrame but got a %15s" % (type(df)))


# In[ ]:


initial_observation(df_train)


# In[ ]:


initial_observation(df_test)


# In[ ]:


df_train['label'].value_counts().plot(kind='bar')


# In[ ]:


x = df_train.drop(["label"], axis = 1)
y = df_train["label"]


# In[ ]:


X_test = df_test.copy()


# ### Preprocessing

# In[ ]:


from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


batch_size = 128
no_classes = 10
epochs = 5
image_height, image_width = 28, 28


# In[ ]:


x = x/ 255.0
X_test = X_test/255.0


# In[ ]:


x.shape


# In[ ]:


X_test.shape


# In[ ]:


x = x.values.reshape(-1,image_height,image_width,1)
input_shape = (image_height, image_width, 1)


# In[ ]:


X_test = X_test.values.reshape(-1,28,28,1)


# In[ ]:


y = tf.keras.utils.to_categorical(y, no_classes)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(x,y)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.imshow(X_train[8][:,:,0])


# In[ ]:


def cnn(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters = 64,
                                    kernel_size = (3,3),
                                    activation = "relu",
                                    input_shape = input_shape))
    model.add(tf.keras.layers.Conv2D(filters = 128,
                                    kernel_size = (3,3),
                                    activation = "relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size= (2,2)))
    model.add(tf.keras.layers.Dropout(rate = 0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units = 1024, activation = "relu" ))
    model.add(tf.keras.layers.Dropout(rate = 0.3))
    model.add(tf.keras.layers.Dense(units = no_classes, activation = "softmax"))
    model.compile(loss = tf.keras.losses.categorical_crossentropy,
                 optimizer = tf.keras.optimizers.Adam(),
                 metrics = ["accuracy"])
    return model

simple_cnn_model = cnn(input_shape)
    


# In[ ]:


history = simple_cnn_model.fit(X_train, Y_train, batch_size, epochs, (X_val, Y_val))
train_loss, train_accuracy = simple_cnn_model.evaluate(X_train, Y_train, verbose = 0)

print("Train Loss:", train_loss)
print("Train Accuracy:", train_accuracy)


# In[ ]:


val_scores = val_loss, val_accuracy = simple_cnn_model.evaluate(X_val, Y_val, verbose = 0)

print("Val Loss:", val_loss)
print("Val Accuracy:", val_accuracy)


# In[ ]:


results = simple_cnn_model.predict(X_test)

