#!/usr/bin/env python
# coding: utf-8

# Here is the training notebook for [this](https://www.kaggle.com/unkownhihi/dfdc-lrcn-inference) inference notebook. The reason why I'm doing it, is because I saw myself in other competition, struggling to get to the medal zone, but don't have a clue how people got there. I removed the dataset because I want to avoid people just clone, run, submit. 
# 
# This training notebook is nothing special, just load, train.

# In[ ]:


import os
import pandas as pd
import numpy as np
import cv2
import glob
import random
from tqdm.notebook import tqdm
folders=os.listdir('../input/example1')+os.listdir('../input/example2')
metadatas=[]
for x in tqdm(folders):
    metadatas.append(pd.read_json('../input/metadatas/metadata'+x.replace('example','')+'.json'))


# In[ ]:


paths=[]
Y=[]
for x in tqdm(folders):
    for y in glob.glob('../input/dfdc-images-p*/'+x+'/*.jpg'):
        if '_' in y:
            continue
        if not os.path.exists(y):
            continue
        Y.append(['REAL','FAKE'].index(metadatas[folders.index(x)][y.replace('../input/dfdc-images-p1/','').replace('../input/dfdc-images-p2/','').replace(x+'/','').replace('.jpg','.mp4')]['label']))
        paths.append(y)


# In[ ]:


real=[]
fake=[]
for m,n in zip(paths,Y):
    if n==0:
        real.append(m)
    else:
        fake.append(m)
fake=random.sample(fake,len(real))
paths,Y=[],[]
for x in real:
    paths.append(x)
    Y.append(0)
for x in fake:
    paths.append(x)
    Y.append(1)


# In[ ]:


def shuffle(X,y):
    new_train=[]
    for m,n in zip(X,y):
        new_train.append([m,n])
    random.shuffle(new_train)
    X,y=[],[]
    for x in new_train:
        X.append(x[0])
        y.append(x[1])
    return X,y


# In[ ]:


paths,y=shuffle(paths,Y)


# In[ ]:


def get_birghtness(img):
    return img/img.max()


# In[ ]:


def process_img(img):
    imgs=[]
    for x in range(10):
        imgs.append(get_birghtness(img[:,x*240:(x+1)*240,:]))
    return np.array(imgs)


# In[ ]:


def gets(paths):
    al=[]
    for x in paths:
        al.append(process_img(cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2RGB)))
    return al
def generator(paths,y,batch_size=16):
    while True:
        for x in range(len(paths)//batch_size):
            if x*batch_size+batch_size>len(paths):
                yield (np.array(gets(paths[x*batch_size:])),y[x*batch_size:])
            yield (np.array(gets(paths[x*batch_size:x*batch_size+batch_size])),
                   y[x*batch_size:x*batch_size+batch_size])


# In[ ]:


get_ipython().system('pip install efficientnet')
import efficientnet.keras as efn
bottleneck = efn.EfficientNetB1(weights='imagenet',include_top=False,pooling='avg')
from keras.layers import *
inp=Input((10,240,240,3))
x=TimeDistributed(bottleneck)(inp)
x = LSTM(128)(x)
x = Dense(64, activation='elu')(x)
x = Dense(1,activation='sigmoid')(x)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(paths,y,test_size=0.15)
from keras import Model
model=Model(inp,x)
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def schedule(epoch):
    return [6e-4,1e-4][epoch]
callback=LearningRateScheduler(schedule)
model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-4))
#model.fit(X,y,batch_size=16)
model.fit_generator(generator(X_train,y_train,4),steps_per_epoch=len(X_train)//4+1,validation_data=generator(X_test,y_test,4),validation_steps=len(X_test)//4+1,epochs=2)


# In[ ]:


model.save('model.h5')


# Please upvote if you found this helpful.
