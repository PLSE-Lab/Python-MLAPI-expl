#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#checking where the files are
import os
os.listdir("/kaggle/input/training")


# In[ ]:


#importing the rquired libraries
import pandas as pd
import numpy as np
import keras
from tqdm import tqdm
from keras import backend as K


# In[ ]:


lookid_data = pd.read_csv("/kaggle/input/IdLookupTable.csv")
lookid_data.head()


# In[ ]:


samplesubmission = pd.read_csv("/kaggle/input/SampleSubmission.csv")
samplesubmission.head()


# In[ ]:


train = pd.read_csv("/kaggle/input/training/training.csv")
train.head().T


# In[ ]:


train.isnull().sum()


# In[ ]:


#filling the nan values
train.fillna(method = 'ffill',inplace = True)


# #### Preparing the training data

# In[ ]:


X = train.Image.values
del train['Image']
Y = train.values


# In[ ]:


x = []
for i in tqdm(X):
    q = [int(j) for j in i.split()]
    x.append(q)
len(x)


# In[ ]:


x = np.array(x)
x = x.reshape(7049, 96,96,1)
x  = x/255.0
x.shape


# #### Splitting the data into 90-10 train test split

# In[ ]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,Y,random_state = 69,test_size = 0.1)


# In[ ]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# #### Model

# In[ ]:


from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D


# In[ ]:


model = Sequential()

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
# model.add(BatchNormalization())
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30))
model.summary()


# In[ ]:


# def root_mean_squared_error(y_true, y_pred):
#         return K.sqrt(K.mean(K.square(y_pred - y_true)))


# In[ ]:


model.compile(optimizer = 'adam',loss = 'mean_squared_error', metrics = ['mae','acc'])
model.fit(x_train,y_train,batch_size=256, epochs=50,validation_data=(x_test,y_test))


# #### Training on the complete Dataset now

# In[ ]:


model.compile(optimizer = 'adam',loss = 'mean_squared_error', metrics = ['mae'])
model.fit(x,Y,batch_size=64, epochs=100)
model.fit(x,Y,batch_size=128, epochs=50)
model.fit(x,Y,batch_size=256, epochs=50)


# #### Predicting for test data

# In[ ]:


test = pd.read_csv("/kaggle/input/test/test.csv")
test.head()


# In[ ]:


test.isnull().sum()


# In[ ]:


test = test.Image.values
x_t = []
for i in tqdm(test):
    q = [int(j) for j in i.split()]
    x_t.append(q)
x_t = np.array(x_t)
x_t = x_t.reshape(-1, 96,96,1)
x_t = x_t/255.0
x_t.shape


# In[ ]:


pred = model.predict(x_t)
pred.shape


# In[ ]:


lookid_list = list(lookid_data['FeatureName'])
imageID = list(lookid_data['ImageId']-1)
pre_list = list(pred)


# In[ ]:


rowid = lookid_data['RowId']
rowid=list(rowid)


# In[ ]:


feature = []
for f in list(lookid_data['FeatureName']):
    feature.append(lookid_list.index(f))


# In[ ]:


preded = []
for x,y in zip(imageID,feature):
    preded.append(pre_list[x][y])


# In[ ]:


rowid = pd.Series(rowid,name = 'RowId')


# In[ ]:


loc = pd.Series(preded,name = 'Location')


# In[ ]:


submission = pd.concat([rowid,loc],axis = 1)


# In[ ]:


submission.to_csv('Utkarsh.csv',index = False)


# In[ ]:




