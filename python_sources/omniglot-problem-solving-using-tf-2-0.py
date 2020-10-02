#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install --upgrade tensorflow tensorflow-gpu
get_ipython().run_line_magic('reload_ext', 'tensorboard')
#!pip uninstall tensorflow
#!pip install tensorflow==2.0r0
#!pip install extra-keras-metrics


# In[ ]:


import os 
import cv2 
import matplotlib.pyplot as pl
import numpy as np
import random

#Neural Network Material
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,BatchNormalization,Lambda,Input,Layer,concatenate
from tensorflow.keras.regularizers import l2,l1,l1_l2
from tensorflow.keras.optimizers import Adam,RMSprop
from keras.utils import np_utils
from sklearn.metrics import average_precision_score as acc
import keras.backend as K
import tensorflow as tf


# In[ ]:


img = cv2.imread('/kaggle/input/omniglot/images_background/Gujarati/character02/0419_05.png')
pl.imshow(img)
pl.show()


# In[ ]:


def loadImages(path):
    X = []
    alpha = 0
    chars = 0
    for a in os.listdir(path):
        alpha += 1
        for b in os.listdir(os.path.join(path,a)):
            chars += 1
            X.append([alpha,chars,cv2.imread(os.path.join(path,a,b),0) // 255])
    
    X = np.array(X)
    return X
path = '/kaggle/input/omniglot/images_background/Gujarati'
X = loadImages(path)
path = '/kaggle/input/omniglot/images_background/Greek'
XNew=loadImages(path)


# In[ ]:


print(X.shape)
print(X[959][0])
print(X[1][2].shape)
X[:,2:3] = X[:,2:3]


# In[ ]:


# XNew[1][2].shape


# In[ ]:


def randomizer(batch,s='train'):
    if s == 'train':
        x = X
    else:
        x = XNew
    Pair = [[],[],[]]
    for a in range(batch):
        alpha,ind = random.randint(0,x[x.shape[0]-1][0]),random.randint(0,19)
        Pair[0].append((x[alpha*ind][2]))
        Pair[1].append((x[alpha*ind+1][2]))
        alpha = random.randint(0,x[x.shape[0]-1][0])
        Pair[2].append((x[alpha*random.randint(0,19)][2]))
    Pair = np.array(Pair)
    return Pair
def generator(batch,s='train'):
    while True:
        pairs = randomizer(batch,s)
        pairs.shape = pairs.shape[0],pairs.shape[1],pairs.shPair = randomizer(10)
for a in range(2):
    f,ax = pl.subplots(1,3)
    ax[0].imshow(Pair[0][a][:])
    ax[1].imshow(Pair[1][a][:])
    ax[2].imshow(Pair[2][a][:])
    f.show()

Pair = Pair.reshape(Pair.shape[0], Pair.shape[1],Pair.shape[2],Pair.shape[3],1)
print(Pair.shape)ape[2],pairs.shape[3],1
        #print(pairs.shape)
        #print(targets.shape)
        yield (pairs[0][:],pairs[1][:],pairs[2][:]),np.array([0]*batch)


# In[ ]:


Pair = randomizer(10)
for a in range(2):
    f,ax = pl.subplots(1,3)
    ax[0].imshow(Pair[0][a][:])
    ax[1].imshow(Pair[1][a][:])
    ax[2].imshow(Pair[2][a][:])
    f.show()

Pair = Pair.reshape(Pair.shape[0], Pair.shape[1],Pair.shape[2],Pair.shape[3],1)
print(Pair.shape)


# In[ ]:


class TripletLossLayer(Layer):
        def __init__(self, alpha, **kwargs):
            self.alpha = alpha
            super(TripletLossLayer, self).__init__(**kwargs)
        def triplet_loss(self, inputs):
            try:
                print("CLASS (TripletLossLayer) : ",inputs)
                #inputs = tf.strided_slice(inputs,[0,1,2],[0,1,2])
                a, p, n = inputs
                p_dist = K.sum(tf.square(a-p), axis=1)
                n_dist = K.sum(K.square(a-n), axis=1)
                return K.abs(K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0))
            except Exception as A:
                print("Error Is In Base Model Triplet Loss Layer")
                print("Error Is : ",A)
                
        def call(self, inputs):
            loss = self.triplet_loss(inputs)
            self.add_loss(loss)
            return loss
def triplet_loss_dist(inputs,alpha=0.4,**kwargs):
        try:
            print("CLASS (TripletLossLayer) : ",inputs)
            #inputs = tf.strided_slice(inputs,[0,1,2],[0,1,2])
            a, p, n = inputs
            p_dist = K.sum(tf.square(a-p), axis=1)
            n_dist = K.sum(K.square(a-n), axis=1)
            return K.abs(K.sum(K.maximum(p_dist - n_dist + alpha, 0), axis=0))
        except Exception as A:
            print("Error Is In Base Model Triplet Loss Layer")
            print("Error Is : ",A)
                
def kernel(shape,name=None,dtype=None):
    return np.random.normal(loc=0,scale=1e-2,size=shape)

def bias(shape,name=None,dtype=None):
    return np.random.normal(loc=0.5,scale=1e-2,size=shape)

def get_base(inputs):
    model = Sequential([
    Conv2D(64,(10,10),activation='relu',kernel_regularizer=l2(2e-4),padding='same',bias_initializer=bias
    ,kernel_initializer=kernel),
    BatchNormalization(),
    MaxPool2D(),
    Dropout(0.3),
    Conv2D(128,(7,7),activation='relu',kernel_regularizer=l2(2e-4),padding='same',bias_initializer=bias
    ,kernel_initializer=kernel),
    BatchNormalization(),
    MaxPool2D(),
    Dropout(0.3),
    Conv2D(256,(4,4),activation='relu',kernel_regularizer=l2(2e-4),padding='same',bias_initializer=bias
    ,kernel_initializer=kernel),
    BatchNormalization(),
    MaxPool2D(),
    Flatten(),
    #Dense(512,activation='relu',kernel_regularizer=l2(2e-4),bias_initializer=bias,kernel_initializer=kernel),
    Dense(4096,activation='sigmoid',kernel_regularizer=l2(1e-3),bias_initializer=bias,kernel_initializer=kernel)
    ])
#     model.summary()
    return model(inputs)
    
def getModel():
    #inp = Input(shape=INPUT)
    ashape = Input(Pair[0][0][:].shape)
    pshape = Input(Pair[1][0][:].shape)
    nshape = Input(Pair[2][0][:].shape)
    
    
    A = get_base(ashape)
    P = get_base(pshape)
    N = get_base(nshape)    
#     A.summary()
# Layer that computes the triplet loss from anchor, positive and negative embedding vectors
    #triplet_loss_layer = TripletLossLayer(alpha=0.5, name='triplet_loss_layer')([A,P,N])
    #lam = Lambda(triplet_loss_dist)([A,P,N])
    merged_vector = concatenate([A, P, N], axis=-1, name='merged_layer')
    mids = Dense(10,activation='relu',bias_initializer=bias,kernel_initializer=kernel)(merged_vector)
    Norm = Dense(1,activation='sigmoid',bias_initializer=bias,kernel_initializer=kernel)(mids)
    root = Model(inputs=[ashape, pshape, nshape], outputs=Norm)
    return root

print('Modelling Started')
root = getModel()
root.summary()
print('Modelling Finished')
cb = tf.keras.callbacks.TensorBoard(histogram_freq=1)
#print('Gujarati Learning')
#root.evaluate([Pair[0][:],Pair[1][:]],Sim)
#print('Greek Learning')
#evaluate()


# In[ ]:


def triplet_loss(y_true, y_pred, alpha = 0.3):    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]    
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)), axis=-1)    
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)), axis=-1)    
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)    
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))       
    return loss
def acc(y_true,y_pred):
    return (1-y_pred) * 100
print('Compilation Started')
#def accuracy():
#    correct=0
#    pred = root.predict([Pair[0][:],Pair[1][:]]).astype(float)
#    for a in range(Sim.shape[0]):
#        if np.round(Sim[a]) == np.round(pred[a][0]):

            #correct+=1
    #print('Real Accuracy Is ',(100.0 * correct) / Sim.shape[0])
try:
    tl = TripletLossLayer(alpha=0.5)
    ad = tf.optimizers.Adam(lr=0.00001)
    root.compile(loss=triplet_loss,optimizer=ad)
except Exception as A:
    print("Compilation Crash",A)
print('Compilation Finished')


# In[ ]:


bat = 30
print('Training Started')
History = root.fit_generator(generator(bat,s="train"),steps_per_epoch=20,epochs=100,callbacks=[cb]
                             ,validation_data=(generator(bat//2,s='validation')),validation_steps=5)
print('Training Finished')
#print('Gujarati Learning')
#root.evaluate([Pair[0][:],Pair[1][:]],Sim)
#print('Greek Learning')
#evaluate()


# In[ ]:


def show_analytics():
    #pl.plot(History.history['accuracy'],c='g')
    pl.plot(History.history['loss'],c='r')
    #pl.title('Training')
    #pl.show()
    #pl.plot(History.history['val_accuracy'],c='g')
    pl.plot(History.history['val_loss'],c='g')
    pl.title('Runtime Validation')
    pl.show()
    #pl.plot(History.history['accuracy'],c='g')
    #pl.title('Accuracy')
    #pl.show()
    #pl.plot(History.history['val_accuracy'],c='g')
    #pl.title('Validation Accuracy')
    ##pl.show()
show_analytics()


# In[ ]:


P = randomizer(1,s='X')
print(P.shape)


# In[ ]:


P = randomizer(32)
P = P.reshape(P.shape[0],P.shape[1],P.shape[2],P.shape[3],1)
print(P.shape)
print("Distance From Others")
for a in range(2):
    for b in root.predict([[P[0][a]],[P[1][a]],[P[2][a]]]):
            print(b)
print("Distance From Self")
for a in range(2):
    for b in root.predict([[P[0][a]],[P[0][a]],[P[0][a]]]):
            print(b)

#pl.imshow(P[2][2][:])

