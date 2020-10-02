#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Great chance to follow along with this notebook:
# https://www.kaggle.com/souravroy/digit-recognition-with-keras-acc-0-98/notebook

# In[ ]:




import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPool2D,Flatten,Dropout
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split


# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_train.head(3)


# In[ ]:


df_test = pd.read_csv("../input/test.csv")
df_test.head(3)


# In[ ]:


features = ["%s%s" %("pixel",pixel_no) for pixel_no in range(0,784)]
df_train_features = df_train[features]
df_train_features.shape


# In[ ]:


df_test.shape


# In[ ]:


df_train_labels = df_train["label"]
df_train_labels_categorical = np_utils.to_categorical(df_train_labels)
df_train_labels_categorical[0:3]


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(df_train_features, df_train_labels_categorical, test_size=0.10,random_state=32)


# # Architecture of keras model

# In[ ]:


model = Sequential()
model.add(Dense(32,activation='relu',input_dim=784))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.03))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.03))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(X_train.values, y_train, batch_size=128, epochs=50, verbose=1)


# In[ ]:


pred_classes = model.predict_classes(df_test.values)


# In[ ]:


submission = pd.DataFrame({
    "ImageId": df_test.index+1,
    "Label": pred_classes
})
print(submission[0:10])

submission.to_csv('./keras_model_1.csv', index=False)


# In[ ]:




