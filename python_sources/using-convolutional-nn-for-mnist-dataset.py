#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
from keras import optimizers
from keras import models
from keras import layers
from keras import losses
from keras.utils import to_categorical

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")


# In[ ]:


train_data.head(2)


# In[ ]:


test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


test_data.head(2)


# In[ ]:


train_label = train_data['label']


# In[ ]:


train_data = train_data.drop('label',axis=1)


# In[ ]:


train_data /= 255
test_data /= 255


# In[ ]:


train_label_to_cat= to_categorical(train_label)
train_label_to_cat.shape


# In[ ]:


train_data.shape


# our data is flatten and if we want to use convolutional nn we must give the network as a pixel

# In[ ]:


train_data = train_data.values.reshape(-1,28,28,1)


# In[ ]:


test_data = test_data.values.reshape(-1,28,28,1)


# In[ ]:


train_data.shape


# now let's build our model

# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))


# In[ ]:


model.summary()


# In[ ]:


model.add(layers.Flatten())
model.add(layers.Dense(1024,activation='relu'))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss= losses.categorical_crossentropy,
             metrics=['accuracy'])


# In[ ]:


model.fit(train_data,train_label_to_cat,epochs=60)


# In[ ]:


predict = model.predict_classes(test_data)


# In[ ]:


test_data_DF = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


output = pd.DataFrame({'ImageId': test_data_DF.index+1,
                       'Label': predict})
output.to_csv('submission.csv', index=False)


# In[ ]:




