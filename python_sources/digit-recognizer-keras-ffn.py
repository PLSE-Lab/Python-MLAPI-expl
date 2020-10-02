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


import keras 
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import to_categorical


# In[ ]:


train=pd.read_csv("../input/digit-recognizer/train.csv")
test=pd.read_csv("../input/digit-recognizer/test.csv")


# In[ ]:


test


# In[ ]:


y_train=train["label"]
x_train=train.drop(["label"],axis=1)
x_test=test


# In[ ]:


model=Sequential()


# In[ ]:


model.add(Dense(40,input_dim=784,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="softmax"))


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss="categorical_crossentropy" ,
              optimizer="adam",
              metrics=["accuracy"])


# In[ ]:


y_train = to_categorical(y_train)


# In[ ]:





# In[ ]:


model.fit(x_train.values,y_train,epochs=70)


# In[ ]:


yp=model.predict(x_test.values)


# In[ ]:


yp


# In[ ]:


import tensorflow as tf


# In[ ]:


pred=tf.keras.backend.argmax(
    yp, axis=-1)


# In[ ]:


pred=pred.numpy()


# In[ ]:


pred


# In[ ]:


pd.Series(pred).value_counts()


# In[ ]:


pred


# In[ ]:


results = pd.Series(pred,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submit.csv",index=False)


# In[ ]:


submission

