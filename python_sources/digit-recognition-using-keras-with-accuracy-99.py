#!/usr/bin/env python
# coding: utf-8

# In[21]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[22]:


class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(float(logs.get('acc'))>0.9999 or float(logs.get('loss'))<0.005):
            print("\nReached Accuracy above 0.9999 or loss below 0.005 so training is stopped")
            self.model.stop_training=True


# In[23]:


training_data = pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')


# In[24]:


training_data.head()


# In[25]:


x_train=training_data.drop(columns=['label'])
y_train=training_data['label']
x_test=training_data.drop(columns=['label'])
y_test=training_data['label']


# In[26]:


x_train=x_train/255.0
test_data=test_data/255.0


# In[27]:


callbacks=myCallback()


# In[28]:


model=keras.models.Sequential()
model.add(keras.layers.Dense(128,activation='relu', input_shape=(x_train.shape[1],)))
#model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[29]:


model.fit(x_train, y_train, epochs=20, callbacks=[callbacks])


# In[30]:


prediction=model.predict([test_data])


# In[31]:


print(prediction)


# In[32]:


model.evaluate(x_test, y_test)


# In[34]:


result = np.argmax(prediction,axis = 1) 
result = pd.Series(result,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)
submission.to_csv('fourth_submission.csv', index=False)

