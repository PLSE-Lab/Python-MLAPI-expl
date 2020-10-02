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


from keras.utils.np_utils import to_categorical

# load data
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv').drop(labels = ["id"],axis = 1) 
Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 


# In[ ]:


X_train = X_train / 255.0
test = test / 255.0


# In[ ]:


timesteps = 28
data_dim = 28

X_train = X_train.values.reshape(-1, timesteps, data_dim)
test = test.values.reshape(-1, timesteps, data_dim)


# In[ ]:


Y_train = to_categorical(Y_train, num_classes = 10)


# In[ ]:


from keras.layers import LSTM, Dropout, Dense, Flatten, Embedding
from keras.models import Sequential

# the number of neurons
n_neurons = 150

model = Sequential()
# LSTM layer1
model.add(LSTM(n_neurons, return_sequences=True, input_shape=(timesteps, data_dim)))
model.add(Dropout(0.25))

# LSTM layer2
model.add(LSTM(n_neurons, return_sequences=True))
model.add(Dropout(0.25))

# LSTM layer3
model.add(LSTM(n_neurons))
model.add(Dropout(0.25))

# FC layer
model.add(Dense(784, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=60, batch_size=150, validation_data=(X_train, Y_train))


# In[ ]:


history.history.keys()


# In[ ]:


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
# plt.ylim([0,1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
# plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# In[ ]:


# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

id_ = np.arange(0,results.shape[0])


# In[ ]:


save = pd.DataFrame({'id':id_,'label':results})
save.to_csv('submission.csv',index=False)

