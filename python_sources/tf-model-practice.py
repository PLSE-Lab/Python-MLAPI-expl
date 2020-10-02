#!/usr/bin/env python
# coding: utf-8

# ## Code Modules & Functions

# In[ ]:


import numpy as np,pandas as pd,pylab as pl
import h5py,tensorflow as tf
from tensorflow import image as timg
import tensorflow.keras.layers as tfkl


# In[ ]:


def resize_info(x_train,y_train,
                x_test,y_test,pixels):
#    x_train=np.array(timg.resize(x_train,[pixels,pixels]))
#    x_test=np.array(timg.resize(x_test,[pixels,pixels]))
    N=len(y_train); shuffle_ids=np.arange(N)
    np.random.RandomState(12).shuffle(shuffle_ids)
    x_train,y_train=x_train[shuffle_ids],y_train[shuffle_ids]
    N=len(y_test); shuffle_ids=np.arange(N)
    np.random.RandomState(23).shuffle(shuffle_ids)
    x_test,y_test=x_test[shuffle_ids],y_test[shuffle_ids]
    n=int(len(x_test)/2)
    x_valid,y_valid=x_test[:n],y_test[:n]
    x_test,y_test=x_test[n:],y_test[n:]
    df=pd.DataFrame([[x_train.shape,x_valid.shape,x_test.shape],
                     [x_train.dtype,x_valid.dtype,x_test.dtype],
                     [y_train.shape,y_valid.shape,y_test.shape],
                     [y_train.dtype,y_valid.dtype,y_test.dtype]],
                    columns=['train','valid','test'],
                    index=['image shape','image type',
                           'label shape','label type'])
    display(df)    
    return [[x_train,x_valid,x_test],
            [y_train,y_valid,y_test]]
def display_examples(x,y):
    n=np.random.randint(1,50)
    fig=pl.figure(figsize=(11,4))
    for i in range(n,n+5):
        ax=fig.add_subplot(1,5,i-n+1,        xticks=[],yticks=[],title=y[i])
        ax.imshow(x[i]); 
    pl.show()


# ## Data

# In[ ]:


pixels=128
fpath='../input/tf-cats-vs-dogs/'
f='CatDogImages.h5'
f=h5py.File(fpath+f,'r')
keys=list(f.keys()); print(keys)
x_test=np.array(f[keys[0]])
y_test=np.array(f[keys[1]]).reshape(-1)
x_train=np.array(f[keys[2]])
y_train=np.array(f[keys[3]]).reshape(-1)
[[x_train,x_valid,x_test],
 [y_train,y_valid,y_test]]=\
resize_info(x_train,y_train,x_test,y_test,pixels)


# In[ ]:


display_examples(x_valid,y_valid)


# ## One Example of CNN Models

# In[ ]:


def CNN(pixels):
    model=tf.keras.Sequential([
        tfkl.Conv2D(32,(5,5),padding='same',
                    input_shape=(pixels,pixels,3)),
        tfkl.LeakyReLU(alpha=.02),
        tfkl.MaxPooling2D(pool_size=(2,2)),
        tfkl.Dropout(.2),    
        tfkl.Conv2D(196,(5,5)),
        tfkl.LeakyReLU(alpha=.02),
        tfkl.MaxPooling2D(pool_size=(2,2)),
        tfkl.Dropout(.2),
        tfkl.GlobalMaxPooling2D(),
        tfkl.Dense(1024),
        tfkl.LeakyReLU(alpha=.02),
        tfkl.Dropout(.5),
        tfkl.Dense(1,activation=tf.nn.sigmoid)])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# In[ ]:


fw='weights.best.hdf5'
def cb(fw):
    early_stopping=tf.keras.callbacks    .EarlyStopping(monitor='val_loss',patience=20,verbose=2)
    checkpointer=tf.keras.callbacks    .ModelCheckpoint(filepath=fw,save_best_only=True,verbose=2)
    lr_reduction=tf.keras.callbacks    .ReduceLROnPlateau(monitor='val_loss',verbose=2,
                       patience=5,factor=.8)
    return [checkpointer,early_stopping,lr_reduction]
def history_plot(fit_history):
    pl.figure(figsize=(12,10)); pl.subplot(211)
    keys=list(fit_history.history.keys())[0:4]
    pl.plot(fit_history.history[keys[0]],
            color='slategray',label='train')
    pl.plot(fit_history.history[keys[2]],
            color='#4876ff',label='valid')
    pl.xlabel("Epochs"); pl.ylabel("Loss")
    pl.legend(); pl.grid()
    pl.title('Loss Function')     
    pl.subplot(212)
    pl.plot(fit_history.history[keys[1]],
            color='slategray',label='train')
    pl.plot(fit_history.history[keys[3]],
            color='#4876ff',label='valid')
    pl.xlabel("Epochs"); pl.ylabel("Accuracy")    
    pl.legend(); pl.grid()
    pl.title('Accuracy'); pl.show()


# In[ ]:


model=CNN(pixels)
history=model.fit(x=x_train,y=y_train,
                  batch_size=128,epochs=15,
                  callbacks=cb(fw),
                  validation_data=(x_valid,y_valid))


# In[ ]:


history_plot(history)


# In[ ]:


model.load_weights(fw)
model.evaluate(x_test,y_test)

