#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import cv2
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


x1=(os.listdir("../input/train"))

image=[]
catg=[]

for x in range(0,5000):
   image.append(os.path.join("../input/train/"+x1[x]))
   catg.append(x1[x][:3])
data=pd.DataFrame()
data['image']=image
data['catg']=catg
data


# In[ ]:


#eda
data.catg.value_counts()
for x in data:
    print(x.dtype)
    img=cv2.imread(x,1)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2YUV_YV12)
    plt.imshow(img)
    break


# In[ ]:


#preparing data x and y
y=pd.get_dummies(data.catg)
X_train=[]
for x in data.image:
        t=cv2.resize(cv2.imread(x,1),(224,224))
        X_train.append(t)


# In[ ]:


X_train=np.array(X_train)


# In[ ]:


X_train.shape


# In[ ]:


x=np.ones((25000,250,250,3))
for y in range(0,25000):
    x[y]=X_train[y]
    
    


# In[ ]:


from keras.applications.imagenet_utils import preprocess_input
plt.imshow(X_train[0])
image = np.expand_dims(X_train[0], axis=0)
image = preprocess_input(image)
image.shape


# In[ ]:


#training of data
import keras
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D

model=Sequential()
model.add(VGG16(include_top=False,input_shape=(224,224,3)))
#model.add(Conv2D(32,(3,3),padding='valid',input_shape=(250,250,3)))
model.add(Flatten())
model.add(Dense(2,activation="softmax"))
model.layers[0].trainable=True

model.compile(optimizer="rmsprop",loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])

model.fit(X_train,y,batch_size=30,epochs=2,verbose=1)


# In[ ]:


#plt.imshow(pre[0])
from keras.applications.vgg16 import decode_predictions
P = decode_predictions(pre)


# In[ ]:


model.summary()

