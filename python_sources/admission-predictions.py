#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


def load_dataset(filepath):
    dataset = pd.read_csv(filepath)
     
    return dataset


# In[ ]:


filepath = '../input/Admission_Predict.csv'
dataset = load_dataset(filepath)


# In[ ]:


dataset.describe()


# In[ ]:


dataset.info()


# In[ ]:


from sklearn.model_selection import train_test_split

def generate_train_test_sets(X, y, test_size = 0.2, shuffle = False, random_state = 0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, shuffle = shuffle, random_state = random_state)
    
    return (X_train, y_train), (X_test, y_test)


# In[ ]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values.reshape(-1, 1)
(X_train, y_train), (X_test, y_test) = generate_train_test_sets(X, y, shuffle = True)

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)


# In[ ]:


from sklearn.preprocessing import StandardScaler

def normalize_features(X):
    std_scaler = StandardScaler()
    X = std_scaler.fit_transform(X)
    
    return X


# In[ ]:


X_train = normalize_features(X_train)
X_test = normalize_features(X_test)


# In[ ]:


def adjust_label_values(Y):
    labels = np.zeros((Y.shape[0], Y.shape[1]))
    labels[Y > 0.5] = 1
    
    return labels


# In[ ]:


y_train = adjust_label_values(y_train)
y_test = adjust_label_values(y_test)


# In[ ]:


import keras

def build_model():   
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(32, activation = 'relu', input_shape = (X_train.shape[1],)))
    model.add(keras.layers.Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
    
    return model


# In[ ]:


model = build_model()
model.summary()


# In[ ]:


history = model.fit(X_train, y_train, 
                    epochs = 40,
                    validation_split = 0.2)


# In[ ]:


import matplotlib.pyplot as plt

def plot_metrics(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize = (10, 7))
    
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, label = 'Training Accuracy')
    plt.plot(epochs, val_acc, label = 'Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label = 'Training Loss')
    plt.plot(epochs, val_loss, label = 'Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    
    plt.show()


# In[ ]:


plot_metrics(history)


# In[ ]:


predictions = model.predict(X_test)
predictions = np.where(predictions > 0.5, 1, 0)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print('Evaluated accuracy: ', (accuracy * 100))


# In[ ]:




