#!/usr/bin/env python
# coding: utf-8

# # TensorFlow 2.x - Multiclass Classification
# 
# 
# ## Dataset
# 
# IRIS dataset is used in this kernel to demonstrate the multiclass classification using `TensorFlow 2.x`. This dataset has 5 features, out of which 4 features are numeric features and 1 is a categorical feature. 

# ## 1. Import dependent libraries

# In[ ]:


import pandas as pd
import numpy as np
import os

# Plotting libraries
import matplotlib.pyplot as plt


# SKLearn libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential


# In[ ]:


# Data file path
FILE_PATH = '/kaggle/input/iris-flower-dataset/IRIS.csv'

# Dataframe from csv file
iris_data = pd.read_csv(FILE_PATH, header=0)


# In[ ]:


iris_data.info()
print("=="*40)
iris_data.head(10)


# ## 2. Preparing dataset

# In[ ]:


X = iris_data.loc[:, iris_data.columns != 'species']
y = iris_data.loc[:, ['species']]


# In[ ]:


y_enc = LabelEncoder().fit_transform(y)
# Converting the label into a matrix form
y_label = tf.keras.utils.to_categorical(y_enc)


# Dataset will be prepared by the tensorflow `from_tensor_slice()` method.

# In[ ]:


# X_train_full, X_test, y_train_full, y_test = train_test_split(X, y_label, test_size=0.15)

# Validation set
# X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size=0.3)


# In[ ]:


print(f"Train shape : {X_train.shape}, Y Train : {y_train.shape}")
print(X_train.shape[1:])


# In[ ]:


def get_model():
    model = Sequential([
        keras.layers.Input(shape=X_train.shape[1:]),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(500, activation='relu',),
        keras.layers.Dense(300, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(3, activation='softmax')
    ])
    
    return model


# In[ ]:


model.summary()


# In[ ]:


model = get_model()

# Compile the model
model.compile(optimizer='adam', 
              loss=keras.losses.CategoricalCrossentropy(),
             metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), verbose=1)


# In[ ]:


model.evaluate(X_test, y_test)


# ## Performance Monitor

# In[ ]:


pd.DataFrame(history.history).plot(figsize=(10,6))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


# In[ ]:


new_data, y_actual = X_test[:3], y_test[:3]

y_proba = model.predict(new_data)

print(f"Actual data : {y_actual}")

for pred in y_proba:
    print(np.argmax(pred))


# In[ ]:




