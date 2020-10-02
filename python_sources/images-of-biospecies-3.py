#!/usr/bin/env python
# coding: utf-8

# ## Code Modules & Functions

# In[ ]:


get_ipython().system('pip install tensorflow_datasets')


# In[ ]:


import warnings; warnings.filterwarnings('ignore')
import numpy as np,pandas as pd,pylab as pl
import h5py,os,tensorflow as tf
from tensorflow import image as timage
import tensorflow_datasets as tfds
import tensorflow_hub as th


# In[ ]:


def resize_display(x_train,y_train,
                   x_test,y_test,pixels):
    x_train=np.array(timage.resize(x_train,[pixels,pixels]))
    x_test=np.array(timage.resize(x_test,[pixels,pixels]))
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


# ## Data

# In[ ]:


splits=['train[:80%]','train[80%:]']
tfds.disable_progress_bar()
(raw_train,raw_test),metadata=tfds.load('cats_vs_dogs:4.0.0',split=splits,
          with_info=True,as_supervised=True)


# In[ ]:


s=128; ntest=4652
ntrain=23262-ntest
x_test=np.zeros((ntest,s,s,3),dtype='float32')
y_test=np.zeros((ntest,1),dtype='int32')
x_train=np.zeros((ntrain,s,s,3),dtype='float32')
y_train=np.zeros((ntrain,1),dtype='int32')
i=0
for f,t in raw_test.take(ntest):
    f=timage.resize(f,[s,s])
    x_test[i,:]=f.numpy()/255
    y_test[i,:]=t; i+=1
i=0
for f,t in raw_train.take(ntrain):
    f=timage.resize(f,[s,s])
    x_train[i,:]=f.numpy()/255
    y_train[i,:]=t; i+=1
fig=pl.figure(figsize=(10,4))   
for i in range(5):    
    ax=fig.add_subplot(1,5,i+1,    xticks=[],yticks=[],title=y_train[i])
    ax.imshow((x_train[i]));


# In[ ]:


pd.DataFrame([[x_train.shape,x_test.shape],
              [x_train.dtype,x_test.dtype],
              [y_train.shape,y_test.shape],
              [y_train.dtype,y_test.dtype]],               
             columns=['train','test'])


# In[ ]:


with h5py.File('CatDogImages.h5','w') as f:
    f.create_dataset('train_images',data=x_train)
    f.create_dataset('train_labels',data=y_train)
    f.create_dataset('test_images',data=x_test)
    f.create_dataset('test_labels',data=y_test)
os.stat('CatDogImages.h5')


# In[ ]:


[[x_train,x_valid,x_test],
 [y_train,y_valid,y_test]]=\
resize_display(x_train,y_train,x_test,y_test,96)


# In[ ]:


del raw_train,raw_test


# In[ ]:


fpath='../input/image-classification-for-biospecies/'
f=h5py.File(fpath+'DogBreedImages.h5','r')
keys=list(f.keys()); print(keys)
x_test2=np.array(f[keys[0]]); y_test2=np.array(f[keys[1]])
x_train2=np.array(f[keys[2]]); y_train2=np.array(f[keys[3]])
fig=pl.figure(figsize=(10,4))   
for i in range(5):    
    ax=fig.add_subplot(1,5,i+1,    xticks=[],yticks=[],title=y_train2[i])
    ax.imshow((x_train2[i]));


# In[ ]:


[[x_train2,x_valid2,x_test2],
 [y_train2,y_valid2,y_test2]]=\
resize_display(x_train2,y_train2,x_test2,y_test2,96)


# ## NN Examples

# In[ ]:


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


[handle_base,pixels]=["mobilenet_v2_075_96",96]
mhandle="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
fw='weights.best.hdf5'


# In[ ]:


model=premodel(pixels,2048,mhandle,1,
               'sigmoid','binary_crossentropy')
history=model.fit(x=x_train,y=y_train,batch_size=128,
                  epochs=5,callbacks=cb(fw),
                  validation_data=(x_valid,y_valid))


# In[ ]:


model.load_weights(fw)
model.evaluate(x_test,y_test)


# In[ ]:


model=premodel(pixels,4096,mhandle,120,
               'softmax','sparse_categorical_crossentropy')
history=model.fit(x=x_train2,y=y_train2,batch_size=128,
                  epochs=5,callbacks=cb(fw),
                  validation_data=(x_valid2,y_valid2))


# In[ ]:


model.load_weights(fw)
model.evaluate(x_test2,y_test2)

