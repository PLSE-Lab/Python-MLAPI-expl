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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from PIL import Image
import cv2


# In[ ]:


img=[]
labels=[]


# In[ ]:


Parasitized=os.listdir("../input/cell_images/cell_images/Parasitized/")
for a in Parasitized:
    try:
        image=cv2.imread("../input/cell_images/cell_images/Parasitized/"+a)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((50, 50))
        img.append(np.array(size_image))
        labels.append(0)
    except AttributeError:
        print("")


# In[ ]:


Uninfected=os.listdir("../input/cell_images/cell_images/Uninfected/")
for b in Uninfected:
    try:
        image=cv2.imread("../input/cell_images/cell_images/Uninfected/"+b)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((50, 50))
        img.append(np.array(size_image))
        labels.append(1)
    except AttributeError:
        print("")


# In[ ]:


len(img)


# In[ ]:


len(labels)


# In[ ]:


img=np.array(img)
labels=np.array(labels)


# In[ ]:


img.shape


# In[ ]:


len_data=len(img)


# In[ ]:


s=np.arange(img.shape[0])
np.random.shuffle(s)
img=img[s]
labels=labels[s]


# In[ ]:


(x_train,x_test)=img[(int)(0.1*len_data):],img[:(int)(0.1*len_data)]
x_train = x_train.astype('float32')/255 
x_test = x_test.astype('float32')/255
(y_train,y_test)=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]


# In[ ]:


y_train=np_utils.to_categorical(y_train)


# In[ ]:


y_train


# In[ ]:


y_test=np_utils.to_categorical(y_test)


# In[ ]:


model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(250,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2,activation="softmax"))#2 represent output layer neurons 
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.fit(x_train,y_train,batch_size=50,epochs=30,verbose=1,validation_split=0.1)


# In[ ]:


pred=model.predict(x_test)


# In[ ]:


pred


# In[ ]:


pred=np.argmax(pred,axis=1)


# In[ ]:


pred=np_utils.to_categorical(pred)


# In[ ]:


pred


# In[ ]:


y_test


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)

