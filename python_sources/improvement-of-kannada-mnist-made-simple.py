#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Conv2D,MaxPooling2D,Input,Dense,Activation, LeakyReLU, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler,ModelCheckpoint,EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import math
from sklearn.model_selection import train_test_split
import random


# In[ ]:


train_datas=pd.read_csv("../input/Kannada-MNIST/train.csv")
val_datas = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")


# In[ ]:


datas = pd.concat([train_datas,val_datas],axis=0)
print(datas.shape)
datas_X = np.array(datas.drop("label",axis=1),dtype=np.float32)
datas_Y = np.array(datas[["label"]],dtype=np.int32)
train_X,val_X,train_Y,val_Y = train_test_split(datas_X,datas_Y,test_size=0.2,shuffle=True)


# In[ ]:


train_X = train_X  / 255.0
val_X = val_X  / 255.0

train_X = np.reshape(train_X,(-1,28,28,1))
val_X = np.reshape(val_X,(-1,28,28,1))


# In[ ]:


from tensorflow.keras.layers import Activation,GlobalAveragePooling2D


# In[ ]:


#nets = 3

model = Sequential()
# block 1
model.add(Conv2D(64, (3,3), padding='same', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# block 2
model.add(Conv2D(128,  (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.1))
model.add(Conv2D(128,  (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2))

# Block 3
model.add(Conv2D(256,  (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

#Block 4
model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2))

#Block 5
model.add(Conv2D(1024, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(1024, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))



#attention layer
'''model.add(Conv2D(256, (1,1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, kernel_size = (1,1), padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))'''


model.add(GlobalAveragePooling2D())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy",metrics=["accuracy"])
print(model.summary())


# In[ ]:


datagen = ImageDataGenerator(
    rotation_range = 15,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = False)


# In[ ]:


def scheduler(epoch):
    if epoch % 1 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.8)
        print("lr changed to {}".format(lr * 0.8))
    return K.get_value(model.optimizer.lr)


# In[ ]:


reduce_lr = LearningRateScheduler(scheduler)
epochs = 20
batch_size = 128


# In[ ]:


model.fit_generator(datagen.flow(train_X,train_Y, batch_size=batch_size),
        epochs = epochs, steps_per_epoch = math.ceil(train_X.shape[0]*1.0/batch_size),  
        validation_data = (val_X,val_Y), callbacks=[reduce_lr],verbose=1)


# In[ ]:



test_csv = pd.read_csv("../input/Kannada-MNIST/test.csv")


# In[ ]:


data_test = np.array(test_csv.drop("id",axis=1),dtype=np.float32)
data_test = np.reshape(data_test,(-1,28,28,1))
X_test = data_test / 255.0
#X_test = data_test
print(X_test.shape,X_test.dtype)


# In[ ]:


results = np.zeros( (X_test.shape[0],10))
results = model.predict(X_test)
results = np.argmax(results,axis = 1)


# In[ ]:


submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")
submission['label'] = results
submission.to_csv("submission.csv",index=False)

