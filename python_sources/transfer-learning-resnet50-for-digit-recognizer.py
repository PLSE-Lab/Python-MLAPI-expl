#!/usr/bin/env python
# coding: utf-8

# # ResNet50 transfer learning for Digit Recognizer

# ## Setup

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Data

# In[ ]:


train = pd.read_csv("../input/digit-recognizer/train.csv")
print(train.shape)


# ## Preprocessing

# In[ ]:


y = train['label'].values

# normalize it by dividing by 255
X = train.drop('label', axis = 1).values.reshape((-1, 28, 28, 1)) / 255

# since ResNet is for 3 channel color image but digit is grayscale 1 channle, change it to 3 channel below
X = np.concatenate((X, X, X), axis = 3)

print(X.shape)


# In[ ]:


# We need below to work categorical_crossentropy loss in compiling
y_binary = to_categorical(y)
print(y_binary.shape)
print(y_binary[:6, :])


# ## Model

# In[ ]:


num_classes = 10

model = Sequential()
model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))
model.add(Dense(num_classes, activation = 'softmax'))

# specified not to change pre-trained weights
model.layers[0].trainable = False


# In[ ]:


model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])


# In[ ]:


history = model.fit(X,
                    y_binary,
                    batch_size = 100,
                    epochs = 10,
                    verbose = 1,
                    validation_split = 0.2)


# In[ ]:


# function to monitor model performance in each epoch
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['acc'], label = 'Training accuracy')
    plt.plot(hist['epoch'], hist['val_acc'], label = 'Validation accuracy')
    plt.legend()
    plt.show()


# In[ ]:


plot_history(history)


# ## Test data prediction

# In[ ]:


test = pd.read_csv("../input/digit-recognizer/test.csv")
print(test.shape)
print(test.head())


# In[ ]:


X_test = test.values.reshape((-1, 28, 28, 1)) / 255
X_test = np.concatenate((X_test, X_test, X_test), axis = 3)
print(X_test.shape)


# In[ ]:


pred_test = model.predict(X_test)
print(pred_test.shape)
print(pred_test[0])
print(np.argmax(pred_test[0]))


# ## Submission

# In[ ]:


test_id = np.arange(1, test.shape[0] + 1, 1)
predictions = np.argmax(pred_test, axis = 1)

sub = pd.DataFrame(data = {'ImageId': test_id,
                           'Label': predictions})
sub.head()

