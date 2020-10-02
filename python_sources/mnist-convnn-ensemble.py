#!/usr/bin/env python
# coding: utf-8

# **In this notebook I'm exploring ConvNN model**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt # to plot charts

from sklearn.model_selection import train_test_split

seed = 42
np.random.RandomState(seed)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Kers modules
from keras.optimizers import SGD
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from keras.callbacks import EarlyStopping, History, LearningRateScheduler


# ## Load & Prepare data

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
train_df.head(5)


# In[ ]:


train_df.shape


# In[ ]:


y = train_df.values[:, 0]
X = train_df.values[:, 1:]
X = X.reshape(-1, 28, 28, 1)


# In[ ]:


X.shape


# In[ ]:


y = to_categorical(y)
y.shape


# In[ ]:


validation_split = .3

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, stratify=y, random_state=seed)


# In[ ]:


print(X_train.shape, y_train.shape)


# In[ ]:


print(X_val.shape, y_val.shape)


# ## Build Network

# In[ ]:


model = Sequential()

model.add(Conv2D(64, kernel_size=3, padding='same', strides=1, activation='relu', input_shape=(28, 28, 1)))

model.add(Conv2D(64, kernel_size=3, padding='same', strides=1, activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2))

model.add(Conv2D(128, kernel_size=3, padding='same', strides=1, activation='relu'))

model.add(Conv2D(128, kernel_size=3, padding='same', strides=1, activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2))

model.add(Conv2D(256, kernel_size=3, padding='same', strides=1, activation='relu'))

model.add(Conv2D(256, kernel_size=3, padding='same', strides=1, activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=2))

model.add(Flatten())

model.add(Dense(2048, activation='relu'))

model.add(Dropout(rate=0.15))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(rate=0.15))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


early_stopping_monitor = EarlyStopping(patience=5) # (min_delta=1e-3)
annealing = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)


# In[ ]:


training = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=32, 
                     callbacks=[early_stopping_monitor, annealing])


# In[ ]:


history = training.history

sns.lineplot(x=range(len(history["acc"])),y=history["acc"], label="acc")
sns.lineplot(x=range(len(history["val_acc"])),y=history["val_acc"], label="val_acc")


# ### Make Predictions

# In[ ]:


predict_df = pd.read_csv("../input/test.csv")

predict_df.head(2)


# In[ ]:


X_predict = predict_df.values
X_predict = X_predict.reshape(-1, 28, 28, 1)


# In[ ]:


X_predict.shape


# In[ ]:


plt.imshow(X_train[0].reshape(28, 28))


# In[ ]:


predictions = model.predict_classes(X_predict)
predictions


# In[ ]:


predictions.shape


# In[ ]:


## create output
output = pd.DataFrame()

output["ImageId"] = [i for i in range(1, predictions.shape[0]+1)]
output["Label"] = predictions

output.to_csv("predictions_cnn_3.csv", index=False)

