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


df = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")
df


# In[ ]:


y_train = np.array(df.label)
y_train


# In[ ]:


df.columns


# In[ ]:


print(28*28)


# In[ ]:


df.iloc[0][1:]


# In[ ]:


df_X = df.drop('label', axis=1)


# In[ ]:


df_X.head()


# In[ ]:


df_X.info()


# In[ ]:


test_array = np.array(df_X)
test_array


# In[ ]:


test_array.shape


# In[ ]:


X_train = test_array.reshape(60000, 28, 28)
X_train


# In[ ]:


import matplotlib.pyplot as plt

plt.imshow(X_train[1])
plt.show()


# In[ ]:


y_train[1]


# In[ ]:


# Zusammenfassung

def get_data(path):
    df = pd.read_csv(path)
    y = np.array(df.label)
    df = df.drop("label", axis=1)
    X = np.array(df)
    X = X.reshape(X.shape[0], 28, 28)
    return X, y


# In[ ]:


X_train, y_train = get_data("/kaggle/input/mnist-in-csv/mnist_train.csv")
X_test, y_test = get_data("/kaggle/input/mnist-in-csv/mnist_test.csv")


# In[ ]:


plt.imshow(X_test[0])
plt.show()


# In[ ]:


y_test[0]


# In[ ]:


import tensorflow as tf


# In[ ]:


X_train = tf.keras.utils.normalize(X_train)
X_test = tf.keras.utils.normalize(X_test)


# In[ ]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer="adam",
             loss="sparse_categorical_crossentropy",
             metrics = ["accuracy"])

model.fit(X_train, y_train, epochs=3)


# In[ ]:


val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_loss, val_acc)


# In[ ]:


np.argmax(model.predict(np.array([X_test[3]])))


# In[ ]:


plt.imshow(X_test[3])
plt.show()


# In[ ]:




