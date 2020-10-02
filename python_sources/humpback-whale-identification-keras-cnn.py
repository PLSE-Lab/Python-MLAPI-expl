#!/usr/bin/env python
# coding: utf-8

# **Humpback Whale Identification - CNN with Keras**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
from IPython.display import  Image

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input,Dense,Activation,BatchNormalization,Conv2D,Flatten
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout
from keras.models import Model

import keras.backend as k
from keras.models import Sequential

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)


# In[ ]:


os.listdir('../input/')


# In[ ]:


train_df=pd.read_csv('../input/train.csv')

train_df.head()


# In[ ]:


train_df.columns


# In[ ]:


# Random image

from IPython.display import  Image
import random

Image(filename='../input/train/'+random.choice(train_df.Image))


# **Preprocessing the images in training then converting into Array**

# In[ ]:


def
X_train=np.zeros((train_df.shape[0],100,100,3))
count=0

for fig in train_df['Image']:
    img=image.load_img('../input/train/'+fig,target_size=(100,100,3))
    x=image.img_to_array(img)
#     x=preprocess_input(img)
    
    X_train[count]=x
    
return X_train


# In[ ]:


def prepareImage(data,n,datset):
    print('Printing Images')
    X_train=np.zeros((n,100,100,3))
    count=0
    
    for fig in data['Image']:
        img=image.load_img('../input/'+datset+'/'+ fig,target_size=(100,100,3))
        x=image.img_to_array(img)
        x=preprocess_input(x)
        
        X_train[count]=x
        
        if(count%500==0):
            print("ProcessingImage : " , count+1,", ",fig)
        count +=1
        
    return X_train     
   


# In[ ]:


X = prepareImage(train_df, train_df.shape[0], "train")
X /= 255


# In[ ]:





# In[ ]:


def prepare_labels(y):
    values=np.array(y)
    label_encoder=LabelEncoder()
    integer_encoded=label_encoder.fit_transform(values)
    print(integer_encoded)
    
    onehot_encoder=OneHotEncoder(sparse=False)
    integer_encoded=integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded=onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)
    y=onehot_encoded
    return y,label_encoder


# In[ ]:


y,label_encoder=prepare_labels(train_df['Id'])


# In[ ]:


y.shape


# In[ ]:


# CNN architecture 

model=Sequential()

# convolution
model.add(Conv2D(32,(7,7),strides=(1,1),name = 'conv0', input_shape=(100,100,3)))

#Batch Normalization

model.add(BatchNormalization(axis=3,name='bn0'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2),name='max_pool'))

model.add(Conv2D(64,(3,3),strides=(1,1),name='conv1'))
model.add(Activation('relu'))
model.add(AveragePooling2D((3,3),name='avg_pool'))

model.add(Flatten())
model.add(Dense(500,activation='relu',name='r1'))
model.add(Dropout(0.8))
model.add(Dense(y.shape[1],activation='softmax',name='sm'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[ ]:


history=model.fit(X,y,epochs=15,batch_size=100,verbose=1)
gc.collect()


# In[ ]:


import gc
gc.collect()


# In[ ]:


plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()


# In[ ]:


test = os.listdir("../input/test/")
print(len(test))


# In[ ]:


col = ['Image']
test_df = pd.DataFrame(test, columns=col)
test_df['Id'] = ''


# In[ ]:


X = prepareImage(test_df, test_df.shape[0], "test")
X /= 255


# In[ ]:


predictions = model.predict(np.array(X), verbose=1)


# In[ ]:


for i, pred in enumerate(predictions):
    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))


# In[ ]:


test_df.head(10)
test_df.to_csv('submission.csv', index=False)

