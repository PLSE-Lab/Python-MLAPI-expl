#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# **importing the necessary components**

# In[2]:


from tensorflow.keras import  utils
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split


# **Data preprocessing**

# In[3]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# splitting on true labels and features

# In[4]:


x_train = train.drop(["label"], axis=1)


# In[5]:


y_train = train["label"]


# In[6]:


x_test=test


# Reshape data in 2D format

# In[7]:


x_train = np.array(x_train)
x_test = np.array(x_test)
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)


# convert true labels to categorical format

# In[8]:


y_train = utils.to_categorical(y_train,10)


# **Data normalization**

# In[9]:


x_train = x_train / 255
x_test = x_test / 255


# splitting train set to two sets: train and crossvalidation

# In[10]:


x_train, x_val, y_train , y_val=train_test_split(x_train,y_train,test_size=0.1)


# **Data augmentation**

# creating a datagenerator

# In[11]:


datagen= ImageDataGenerator(
            zoom_range=0.3,
            rotation_range=0.2,
            width_shift_range=0.5,
            height_shift_range=0.5
            )


# **Creating a CNN model**

# In[12]:


model = Sequential()


# In[13]:


model.add(Conv2D(filters=32,kernel_size=(5,5),input_shape=(x_train.shape[1:]),padding="Same",activation="relu"))
model.add(Conv2D(filters=32,kernel_size=(5,5),padding="Same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64,kernel_size=(5,5),padding="Same",activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(5,5),padding="Same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),))
model.add(Dropout(0.25))
model.add(Conv2D(filters=128,kernel_size=(5,5),padding="Same",activation="relu"))
model.add(Conv2D(filters=128,kernel_size=(5,5),padding="Same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10,activation="softmax"))


# compiling our model!

# In[14]:


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())


# **Creating some callbacks**

# In[15]:


checkpoint=ModelCheckpoint("mnistCNN",
                           monitor="val_acc",
                          save_best_only=True,
                          verbose=1)


# In[16]:


lr_reduce=ReduceLROnPlateau(monitor="val_acc",
                            patience=4,
                            verbose=1,
                            factor=0.5,
                            min_lr=0.00000001
    )


# **training our model**

# In[18]:


model.fit(datagen.flow(x_train,y_train,batch_size=100),
         batch_size=1,
         epochs=40,
         validation_data=(x_val,y_val),
         verbose=1,
         callbacks=[checkpoint,lr_reduce])


# **predicting on test set**

# In[19]:


model.load_weights("mnistCNN")


# In[20]:


predictions = model.predict(x_test)


# In[21]:


print(predictions.shape)


# In[22]:


predictions=np.argmax(predictions, axis=1)


# In[23]:


print(predictions.shape)


# **creating submission file**

# In[24]:


submission=pd.DataFrame({"ImageId":range(1,x_test.shape[0]+1),"Label": predictions})


# In[25]:


submission = submission.to_csv("submission.csv",index=False)

