#!/usr/bin/env python
# coding: utf-8

# ## Code Modules

# In[ ]:


import numpy as np,pandas as pd
import h5py,pylab as pl
import tensorflow_hub as th,tensorflow as tf
from tensorflow import image as timage


# ## Data

# In[ ]:


fpath='../input/tf-cats-vs-dogs/'
f='CatDogImages.h5'
f=h5py.File(fpath+f,'r')
keys=list(f.keys()); print(keys)
x_test=np.array(f[keys[0]])
y_test=np.array(f[keys[1]],dtype='int8')
x_train=np.array(f[keys[2]])
y_train=np.array(f[keys[3]],dtype='int8')
N=len(y_train); shuffle_ids=np.arange(N)
np.random.RandomState(12).shuffle(shuffle_ids)
x_train,y_train=x_train[shuffle_ids],y_train[shuffle_ids]
N=len(y_test); shuffle_ids=np.arange(N)
np.random.RandomState(23).shuffle(shuffle_ids)
x_test,y_test=x_test[shuffle_ids],y_test[shuffle_ids]
n=int(len(x_test)/2)
x_valid,y_valid=x_test[:n],y_test[:n]
x_test,y_test=x_test[n:],y_test[n:]
del f
pd.DataFrame([[x_train.shape,x_valid.shape,x_test.shape],
              [x_train.dtype,x_valid.dtype,x_test.dtype],
              [y_train.shape,y_valid.shape,y_test.shape],
              [y_train.dtype,y_valid.dtype,y_test.dtype]],
             columns=['train','valid','test'],
             index=['image shape','image type',
                    'label shape','label type'])


# ## TF Hub Models

# In[ ]:


fw='weights.best.hdf5'
def premodel(pix,den,mh,lbl,activ,loss):
    model=tf.keras.Sequential([
        tf.keras.layers.Input((pix,pix,3),
                              name='input'),
        th.KerasLayer(mh,trainable=True),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(den,activation='relu'),
        tf.keras.layers.Dropout(rate=.5),
        tf.keras.layers.Dense(lbl,activation=activ)])
    model.compile(optimizer='adam',
                  metrics=['accuracy'],loss=loss)
    display(model.summary())
    return model
def cb(fw):
    early_stopping=tf.keras.callbacks    .EarlyStopping(monitor='val_loss',patience=20,verbose=2)
    checkpointer=tf.keras.callbacks    .ModelCheckpoint(filepath=fw,save_best_only=True,verbose=2)
    lr_reduction=tf.keras.callbacks    .ReduceLROnPlateau(monitor='val_loss',verbose=2,
                       patience=5,factor=.8)
    return [checkpointer,early_stopping,lr_reduction]


# In[ ]:


[handle_base,pixels]=["inception_v3",128]
mhandle="https://tfhub.dev/google/imagenet/{}/classification/4".format(handle_base)


# In[ ]:


model=premodel(pixels,1024,mhandle,1,
               'sigmoid','binary_crossentropy')
history=model.fit(x=x_train,y=y_train,batch_size=128,
                  epochs=5,callbacks=cb(fw),
                  validation_data=(x_valid,y_valid))


# In[ ]:


model.load_weights(fw)
model.evaluate(x_test,y_test)

