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


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
import keras


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


x_train = np.array(train.drop('label', axis=1))
y_train = np.array(train['label']).reshape(x_train.shape[0], 1)
x_test = np.array(test)
print(x_train.shape, y_train.shape)
print(x_test.shape)


# In[ ]:


enc = OneHotEncoder()
enc.fit(y_train)
y_train_hot = np.array(enc.transform(y_train).toarray())
print(y_train_hot.shape)


# In[ ]:


x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train_hot, test_size=0.3, random_state=420)
print(x_train.shape, y_train.shape)
print(x_cv.shape, y_cv.shape)


# In[ ]:


mean_px = x_train.mean().astype(np.float32)
std_px = x_train.std().astype(np.float32)

def standardize(x): 
    return (x-mean_px)/std_px


# In[ ]:


plt.imshow(x_train[0].reshape(28, 28))


# In[ ]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# In[ ]:


x_cv = x_cv.reshape(x_cv.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)


# In[ ]:


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))


# In[ ]:


model.compile(loss=keras.losses.categorical_crossentropy, 
              optimizer=keras.optimizers.Adadelta(), 
              metrics=['accuracy'])


# In[ ]:


from keras.callbacks import EarlyStopping
my_callbacks = [EarlyStopping(monitor="acc", patience=5, mode=max)]


# In[ ]:


hist = model.fit(x_train, y_train, batch_size=128, 
                 epochs=100, verbose=1, validation_split=0.3, callbacks=my_callbacks)


# In[ ]:


score = model.evaluate(x_cv, y_cv)
print("Testing Loss:", score[0])
print("Testing Accuracy:", score[1])


# In[ ]:


model.summary()


# In[ ]:


score


# In[ ]:


epoch_list = list(range(1, len(hist.history['acc']) + 1))
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend(("Training Accuracy", "Validation Accuracy"))
plt.show()


# In[ ]:


y_pred = model.predict(x_test, batch_size=128)


# In[ ]:


y_pred.shape


# In[ ]:


y_p = y_pred.argmax(axis=1)
y_p.shape


# In[ ]:


y_p[:10]


# In[ ]:


n=3
print(y_p[n])
plt.imshow(x_test[n].reshape(28,28))


# In[ ]:


my_submission = pd.DataFrame({'ImageId': np.array(range(1,y_p.shape[0]+1)), 'Label': y_p})


# In[ ]:


my_submission.head()


# In[ ]:


#my_submission.to_csv('my_submission.csv', index=False)


# In[ ]:


from keras.layers.normalization import BatchNormalization


# Initial model

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Dropout(0.5))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())


# In[ ]:


# Initial fIT & Evaluate initial model

num_epochs = 30
BatchSize = 2048

model.fit(x_train, y_train, epochs=num_epochs, batch_size=BatchSize)
test_loss, test_acc = model.evaluate(x_cv, y_cv)
print("_"*80)
print("Accuracy on test ", test_acc)


# In[ ]:


model.save_weights('mnist_model_weights.h5')


# In[ ]:


y_pred = model.predict(x_test)


# In[ ]:


y_pred[1]


# In[ ]:


y_p = y_pred.argmax(axis=1)
y_p.shape


# In[ ]:


n=3
print(y_p[n])
plt.imshow(x_test[n].reshape(28,28))


# In[ ]:


my_submission2 = pd.DataFrame({'ImageId': np.array(range(1,y_p.shape[0]+1)), 'Label': y_p})
my_submission2.head()


# In[ ]:


#my_submission2.to_csv('my_submission2.csv', index=False)


# In[ ]:




