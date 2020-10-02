#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
import math
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


X_train=np.load("../input/he-challenge-data-32x32/X_train_64.npy")
X_test=np.load("../input/he-challenge-data-32x32/X_test_64.npy")
Y_train=np.load("../input/he-challenge-data-32x32/Y_train_np.npy")
plt.imshow(X_test[0])


# In[ ]:


img=cv2.imread('../input/flower-recognition-he/he_challenge_data/data/test/18540.jpg')
plt.imshow(img)


# In[ ]:


X_train_csv=pd.read_csv("../input/flower-recognition-he/he_challenge_data/data/train.csv")
X_test_csv=pd.read_csv("../input/flower-recognition-he/he_challenge_data/data/test.csv")


# In[ ]:


X_train_csv["category"].value_counts().plot(kind="bar",figsize=(25,5))


# In[ ]:


from keras import optimizers,layers,models,regularizers
from keras.layers import GlobalMaxPool1D,GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from keras.applications import DenseNet121,DenseNet169
from keras.models import Sequential
from keras.layers import Dense, Flatten,Activation,Conv2D,MaxPooling2D,Dense,Dropout,BatchNormalization


# In[ ]:


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


# In[ ]:


# from keras.preprocessing import image
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input
# import numpy as np

# model_vgg = VGG16(weights='imagenet', include_top=False,input_shape=(64,64,3))
# model = models.Sequential()
# model.add(model_vgg)
# model.add(layers.BatchNormalization())
# # model.add(layers.GlobalMaxPool1D())
# model.add(layers.Flatten())
# model.add(layers.Dropout(rate=0.2))
# model.add(layers.Dense(units=256,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
# model.add(layers.Dense(units=102,activation='softmax',kernel_regularizer=regularizers.l2(0.01)))


# In[ ]:


# from keras.applications.inception_v3 import InceptionV3
# base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(64,64,3))
# model.add(base_model.output)
# model.add(GlobalAveragePooling2D())
# # model.add(layers.BatchNormalization())
# # model.add(layers.GlobalMaxPool1D())
# # model.add(layers.Flatten())
# # model.add(layers.Dropout(rate=0.5))
# model.add(layers.Dense(units=256,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
# model.add(layers.Dropout(rate=0.5))
# model.add(layers.Dense(units=102,activation='softmax',kernel_regularizer=regularizers.l2(0.01)))


# In[ ]:


from keras.applications.resnet50 import ResNet50
resnet_weights_path = '../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
model_resnet=ResNet50(include_top=False, weights=resnet_weights_path,input_shape=(64,64,3))
model = models.Sequential()
model.add(model_resnet)
model.add(layers.BatchNormalization())
# model.add(layers.GlobalMaxPool1D())
model.add(layers.Flatten())
model.add(layers.Dropout(rate=0.2))
model.add(layers.Dense(units=256,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(units=102,activation='softmax',kernel_regularizer=regularizers.l2(0.01)))


# In[ ]:


model.summary()


# In[ ]:


adam=optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,epsilon=1e-8)
# sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=adam,loss="categorical_crossentropy",metrics=['accuracy'])


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


# In[ ]:


from numpy import clip
X_train=X_train/255
for img in range(0,len(X_train)):
    mean,std=X_train[img].mean(),X_train[img].std()
    X_train[img] = (X_train[img] - mean) / std
    X_train[img] = clip(X_train[img], -1.0, 1.0)
    X_train[img] = (X_train[img] + 1.0) / 2.0


# In[ ]:


train_set,val_set,y_set,y_val=train_test_split(X_train,Y_train,test_size=0.2,random_state=42)
train_gen = ImageDataGenerator(
                                zoom_range=[0.8,1.0],
                                width_shift_range=0.2, 
                                height_shift_range=0.2,
                                rotation_range=15,
#                                 brightness_range=[0.8,1.0],
                                fill_mode='nearest',
                                horizontal_flip=True,
#                                 vertical_flip=True
#                                 ,shear_range=1.
#                                zca_whitening=True,
#                                featurewise_std_normalization=True,
#                                samplewise_std_normalization=False
                              )
train_gen.fit(train_set)
train_gen_flow  =  train_gen.flow(train_set,y_set,batch_size=32)


# In[ ]:


val_gen = ImageDataGenerator(rescale=1./1)
val_gen.fit(val_set)
val_gen_flow = val_gen.flow(val_set,y_val,batch_size=32)


# In[ ]:


for x_batch, y_batch in train_gen.flow(train_set, y_set, batch_size=1):
    for i in range(0, 1):
        fig = plt.figure()
        ax = fig.add_subplot(110 + 1 + i)
        ax.imshow(x_batch[i], interpolation='nearest')
        ax.set_aspect(0.5)
    plt.show()
    break


# In[ ]:


lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

model.fit_generator(
    train_gen_flow,
    steps_per_epoch=len(train_set)/32, 
    epochs=20,
    validation_data = val_gen_flow,
    validation_steps = 32
    ,callbacks=callbacks_list
                       )


# In[ ]:


# %reset_selective -f X_train     
# import gc
# gc.collect()


# In[ ]:


# del X_test ,train_gen_flow ,val_gen_flow,train_set,val_set,y_set,y_val,Y_train,train_gen,val_gen,x_batch,y_batch,X_test_csv
# gc.collect()


# In[ ]:


from numpy import clip
X_train2=np.load("../input/he-challenge-data-32x32/X_train_64.npy")
X_train2=X_train2/255
for img in range(0,18540):
    mean,std=X_train2[img].mean(),X_train2[img].std()
    X_train2[img] = (X_train2[img] - mean) / std
    X_train2[img] = clip(X_train2[img], -1.0, 1.0)
    X_train2[img] = (X_train2[img] + 1.0) / 2.0
Y_pred_train=model.predict(X_train2)


# In[ ]:


count=0
sets=[]
for i in range(0,18540):
    k=0
    for j in range(0,102):
        if(Y_pred_train[i][j]>=0.55 and Y_pred_train[i][j]<0.999) :
            count=count+1
            k=1
            break
    if(k):continue
    else: sets.append(i)
print(count)


# In[ ]:


X_train = np.delete(X_train, sets, 0)
Y_train = np.delete(Y_train, sets, 0)
print(len(X_train),len(Y_train))


# In[ ]:


Y_pred_train=(Y_pred_train>0.5)*1
Y_pred_train=Y_pred_train.argmax(axis=1)
Y_pred_train=Y_pred_train+1


# In[ ]:


Y_pred_train=pd.DataFrame(data=Y_pred_train,copy=True)
Y_pred_train[0].value_counts().plot(kind='bar',figsize=(25,5))


# In[ ]:


# New Model

from keras.applications.resnet50 import ResNet50
resnet_weights_path = '../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
model_resnet=ResNet50(include_top=False, weights=resnet_weights_path,input_shape=(64,64,3))
model = models.Sequential()
model.add(model_resnet)
model.add(layers.BatchNormalization())
# model.add(layers.GlobalMaxPool1D())
model.add(layers.Flatten())
model.add(layers.Dropout(rate=0.2))
model.add(layers.Dense(units=256,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(units=102,activation='softmax',kernel_regularizer=regularizers.l2(0.01)))

# Learning rate Scheduler

def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

# Optimizers

adam=optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,epsilon=1e-8)
# sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=adam,loss="categorical_crossentropy",metrics=['accuracy'])


# In[ ]:


train_set,val_set,y_set,y_val=train_test_split(X_train,Y_train,test_size=0.2,random_state=42)
train_gen = ImageDataGenerator(
                                zoom_range=[0.8,1.0],
                                width_shift_range=0.2, 
                                height_shift_range=0.2,
                                rotation_range=15,
#                                 brightness_range=[0.8,1.0],
                                fill_mode='nearest',
                                horizontal_flip=True,
#                                 vertical_flip=True
#                                 ,shear_range=1.
#                                zca_whitening=True,
#                                featurewise_std_normalization=True,
#                                samplewise_std_normalization=False
                              )
train_gen.fit(train_set)
train_gen_flow  =  train_gen.flow(train_set,y_set,batch_size=32)


# In[ ]:


val_gen = ImageDataGenerator(rescale=1./1)
val_gen.fit(val_set)
val_gen_flow = val_gen.flow(val_set,y_val,batch_size=32)


# In[ ]:


model.fit_generator(
    train_gen_flow,
    steps_per_epoch=len(train_set)/32, 
    epochs=20,
    validation_data = val_gen_flow,
    validation_steps = 32
    ,callbacks=callbacks_list
                       )


# In[ ]:


import gc
del X_train2,Y_pred_train
gc.collect()


# In[ ]:


from numpy import clip
X_test2=np.load("../input/he-challenge-data-32x32/X_test_64.npy")
X_test2=X_test2/255
for img in range(0,2009):
    mean,std=X_test2[img].mean(),X_test2[img].std()
    X_test2[img] = (X_test2[img] - mean) / std
    X_test2[img] = clip(X_test2[img], -1.0, 1.0)
    X_test2[img] = (X_test2[img] + 1.0) / 2.0
Y_pred=model.predict(X_test2)
Y_pred=(Y_pred>0.5)*1
Y_pred=Y_pred.argmax(axis=1)+1


# In[ ]:


sub=pd.read_csv("../input/flower-recognition-he/he_challenge_data/data/sample_submission.csv")
sub["category"]=Y_pred


# In[ ]:


sub.to_csv('submission.csv',index=False)


# In[ ]:


# import numpy as np
# N = 18540
# NN=102
# b = np.zeros((N,NN+1))
# b[:,1:] = Y_train
# count=0
# for i in range(0,18540):
#     if(b[i][0]==0):count=count+1
# count


# In[ ]:


Y_train.argmax(axis=1)+1

