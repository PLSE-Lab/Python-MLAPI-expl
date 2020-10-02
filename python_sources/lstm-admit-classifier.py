#!/usr/bin/env python
# coding: utf-8

# This is an attempt to use a basic LSTM implementation to find out whether a student will be admittable. We define a classifier which tries to figure out whether a students chance of admittability will be above a certain treshold, in this case 80%. No advanced feature engineering is done, the purpose of this is to solely explore LSTM usage for this task.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.callbacks import TensorBoard
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Embedding
from keras.optimizers import Nadam
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# First, we load the data as part of a Pandas DataFrame.

# In[ ]:


# Load pandas frame and output examples
df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
df.head(5)


# We then initialize the TensorBoard which will track our learning process with regards to the loss-function and accuracy.

# In[ ]:


# Setup TensorBoard
tBoard = TensorBoard(log_dir='graphs/', histogram_freq=0,
                     write_graph=True, write_images=True)


# Afterwards, the data gets represented in a machine readable format, dropping the index column and splitting the data into training features and predictable categories. Should a specified treshold (in this case 80%) of admission chance be reached, the feature vector is regarded as "admittable", marked as a 1 in the category list. Otherwise, it will be marked as a 0.

# In[ ]:


# define data
chance_admit = df[df.columns[8]].values
features = df.drop(df.columns[[0, 8]], axis=1).values
categorical = []
print("Raw chances:\t",chance_admit[:10])
print("Raw features:\t",features[:10])

treshold_admittable = 0.8

for row in chance_admit:
    if row >= treshold_admittable:
        categorical.append(1)
    else:
        categorical.append(0)
categorical = to_categorical(categorical, num_classes=2)
print("Categorical chances:\t",categorical[:10])


# We then split the input lists into training and testing set with 20% being in the training set.

# In[ ]:


# split train/test
X_train, X_test, y_train, y_test = train_test_split(features, categorical, test_size=0.2, random_state=42)


# Then the machine learning model is defined using Keras. We use a two-layer LSTM architecture.

# In[ ]:


# define DNN
model=None

model = Sequential()
model.add(Embedding(df['GRE Score'].max()+1, 80, input_length=7))
model.add(LSTM(80, dropout=0.3, recurrent_dropout=0.4, return_sequences=True))
model.add(LSTM(80))
model.add(Dense(2, activation='softmax', kernel_regularizer='l1'))
model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=0.0001), metrics=['categorical_accuracy'])
print("Model summary:")
model.summary()


# The data gets trained in 20 epochs as batches of 20.

# In[ ]:


# train neural net
history = model.fit(X_train, y_train, batch_size=20, epochs=20,validation_data=(X_test,y_test), callbacks=[tBoard])


# We then measure some metrics for our classifier using scikit-learns built-in evaluation feature.

# In[ ]:


y_pred = model.predict(X_test)
y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred))


# Seems pretty good for such a basic try. We could get even better results by exploring our training features and sorting out useless features while optimizing the training parameter, perhaps using a grid search.
