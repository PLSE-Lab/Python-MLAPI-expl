#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Dense,Flatten,BatchNormalization
from keras.optimizers import Adamax
from keras.layers.core import Dense,Dropout,Activation
from keras.regularizers import l2
import keras.backend as K
from keras.layers import Input,Concatenate
from keras.layers.pooling import AveragePooling2D,GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,LearningRateScheduler,EarlyStopping
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Preparing data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


Y_train = train["label"]
X_train = train.drop(labels="label",axis=1)

#g = sns.countplot(Y_train)

Y_train.value_counts()


# In[ ]:


X_train = X_train/255.0
test = test/255.0


# In[ ]:


X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# In[ ]:


Y_train = to_categorical(Y_train,num_classes=10)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.05)


# In[ ]:


def conv_layer(x,concat_axis,nb_filter,dropout_rate=None,weight_decay=1E-4):
    x = BatchNormalization(axis=concat_axis,
                          gamma_regularizer=l2(weight_decay),
                          beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter,(3,3),padding='same',kernel_regularizer=l2(weight_decay),use_bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x
def transition_layer(x,concat_axis,nb_filter,dropout_rate=None,weight_decay=1E-4):
    x = BatchNormalization(axis=concat_axis,
                          gamma_regularizer=l2(weight_decay),
                          beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter,(1,1),padding='same',kernel_regularizer=l2(weight_decay),use_bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2,2),strides=(2,2))(x)
    return x

def denseblock(x,concat_axis,nb_filter,nb_layers,growth_rate,dropout_rate=None,weight_decay=1E-4):
    list_features = [x]
    for i in range(nb_layers):
        x = conv_layer(x,concat_axis,growth_rate,dropout_rate=None,weight_decay=1E-4)
        list_features.append(x)
        x = Concatenate(axis=concat_axis)(list_features)
        nb_filter += growth_rate
    return x,nb_filter

def Densenet(nb_classes,img_dim,depth,nb_dense_block,nb_filter,growth_rate,
             dropout_rate=None,weight_decay=1E-4):
    if K.image_dim_ordering() == "th":
        concat_axis = 1
    elif K.image_dim_ordering() == "tf":
        concat_axis = -1
        
    model_input = Input(shape=img_dim)
    
    assert (depth-4)%3  == 0 , "Depth must be 4*N +3"
    
    nb_layers = int((depth-4 )/ 3) 
    
    x = Conv2D(nb_filter,(3,3),padding='same',use_bias=False,
               kernel_regularizer=l2(weight_decay))(model_input)
    
    for block_id in range(nb_dense_block-1):
        
        x,nb_filter = denseblock(x,concat_axis,nb_filter,nb_layers,growth_rate,
                                 dropout_rate=None,weight_decay=1E-4)
        x = transition_layer(x,concat_axis,nb_filter,dropout_rate=None,weight_decay=1E-4)
        
    x = BatchNormalization(axis=concat_axis,
                          gamma_regularizer=l2(weight_decay),
                          beta_regularizer=l2(weight_decay))(x)
    
    x = Activation('relu')(x)
    
    x = GlobalAveragePooling2D(data_format=K.image_data_format())(x)
    
    x = Dense(nb_classes,activation='softmax',kernel_regularizer=l2(weight_decay),
                            bias_regularizer=l2(weight_decay))(x)
    densenet = Model(inputs=[model_input], outputs=[x], name="DenseNet")
    
    return densenet


# In[ ]:


model =         Densenet(nb_classes=10,
                         img_dim=(28,28,1),
                         depth = 34,
                         nb_dense_block = 5,
                         growth_rate=12,
                         nb_filter=32,
                         dropout_rate=0.2,
                         weight_decay=1E-4)
model.summary()


# In[ ]:


model_filepath = 'model.h5'
batch_size=64
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=1e-5, patience=2, verbose=1)
msave = ModelCheckpoint(model_filepath, save_best_only=True)


# In[ ]:


model.compile(loss='categorical_crossentropy',
             optimizer = Adamax(),
             metrics=['accuracy'])

model.fit(X_train ,Y_train,
               batch_size = 64,
               validation_data = (X_test,Y_test),
               epochs = 20,
               callbacks=[lr_reduce,msave,annealer],
               verbose = 1)


# In[ ]:


results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)


# In[ ]:





# In[ ]:




