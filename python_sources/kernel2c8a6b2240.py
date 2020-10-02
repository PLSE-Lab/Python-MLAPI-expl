#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import cv2
import numpy as np
import os
import zipfile
from os import listdir
listdir('../input')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from keras.utils import to_categorical
import sys
import os
import keras
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import callbacks
from keras import backend as K
from keras.optimizers import Adam,SGD
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from keras.applications import MobileNetV2
from keras.models import Model


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))


# In[ ]:


trainarr=[]
trainclass=[]
valarr=[]
valclass=[]
for count,i in enumerate(listdir('../input/imagess01/1/')):
    image=cv2.imread('../input/imagess01/1/'+i)
    image=cv2.resize(image,(224,224))
    if count>0 and count<140:
        trainarr.append(image)
        trainclass.append(0)
    elif count>=160 : 
        valarr.append(image)
        valclass.append(0)
for count,i in enumerate(listdir('../input/imagess02/2/')):
    image=cv2.imread('../input/imagess02/2/'+i)
    image=cv2.resize(image,(224,224))
    if count>0 and count<140:
        trainarr.append(image)
        trainclass.append(1)
    elif count>=160 : 
        valarr.append(image)
        valclass.append(1)  
        
for count,i in enumerate(listdir('../input/imagess03/3/')):
    image=cv2.imread('../input/imagess03/3/'+i)
    image=cv2.resize(image,(224,224))
    if count>0 and count<140:
        trainarr.append(image)
        trainclass.append(2)
    elif count>=160 : 
        valarr.append(image)
        valclass.append(2)    
for count,i in enumerate(listdir('../input/imagesss04/4/')):
    image=cv2.imread('../input/imagesss04/4/'+i)
    image=cv2.resize(image,(224,224))
    if count>0 and count<140:
        trainarr.append(image)
        trainclass.append(3)
    elif count>=160 : 
        valarr.append(image)
        valclass.append(3)    

for count,i in enumerate(listdir('../input/imagesss05/5/')):
    image=cv2.imread('../input/imagesss05/5/'+i)
    image=cv2.resize(image,(224,224))
    if count>0 and count<140:
        trainarr.append(image)
        trainclass.append(4)
    elif count>=160 : 
        valarr.append(image)
        valclass.append(4)    

for count,i in enumerate(listdir('../input/imagesss06/6/')):
    image=cv2.imread('../input/imagesss06/6/'+i)
    image=cv2.resize(image,(224,224))
    if count>0 and count<140:
        trainarr.append(image)
        trainclass.append(5)
    elif count>=160 : 
        valarr.append(image)
        valclass.append(5)    

for count,i in enumerate(listdir('../input/imagesss07/7/')):
    image=cv2.imread('../input/imagesss07/7/'+i)
    image=cv2.resize(image,(224,224))
    if count>0 and count<140:
        trainarr.append(image)
        trainclass.append(6)
    elif count>=160 : 
        valarr.append(image)
        valclass.append(6)    

for count,i in enumerate(listdir('../input/imagesss08/8/')):
    image=cv2.imread('../input/imagesss08/8/'+i)
    image=cv2.resize(image,(224,224))
    if count>0 and count<140:
        trainarr.append(image)
        trainclass.append(7)
    elif count>=160 : 
        valarr.append(image)
        valclass.append(7)    

for count,i in enumerate(listdir('../input/imagesss09/9/')):
    image=cv2.imread('../input/imagesss09/9/'+i)
    image=cv2.resize(image,(224,224))
    if count>0 and count<140:
        trainarr.append(image)
        trainclass.append(8)
    elif count>=160 : 
        valarr.append(image)
        valclass.append(8)    

for count,i in enumerate(listdir('../input/imagesss010/10/')):
    image=cv2.imread('../input/imagesss010/10/'+i)
    image=cv2.resize(image,(224,224))
    if count>0 and count<140:
        trainarr.append(image)
        trainclass.append(9)
    elif count>=160 : 
        valarr.append(image)
        valclass.append(9)    

for count,i in enumerate(listdir('../input/imagesss011/11/')):
    image=cv2.imread('../input/imagesss011/11/'+i)
    image=cv2.resize(image,(224,224))
    if count>0 and count<140:
        trainarr.append(image)
        trainclass.append(10)
    elif count>=160 : 
        valarr.append(image)
        valclass.append(10)    

for count,i in enumerate(listdir('../input/imagesss012/12/')):
    image=cv2.imread('../input/imagesss012/12/'+i)
    image=cv2.resize(image,(224,224))
    if count>0 and count<140:
        trainarr.append(image)
        trainclass.append(11)
    elif count>=160 : 
        valarr.append(image)
        valclass.append(11)    

for count,i in enumerate(listdir('../input/imagesss013/13/')):
    image=cv2.imread('../input/imagesss013/13/'+i)
    image=cv2.resize(image,(224,224))
    if count>0 and count<140:
        trainarr.append(image)
        trainclass.append(12)
    elif count>=160 : 
        valarr.append(image)
        valclass.append(12)    

for count,i in enumerate(listdir('../input/imagesss0014/14/')):
    image=cv2.imread('../input/imagesss0014/14/'+i)
    image=cv2.resize(image,(224,224))
    if count>0 and count<140:
        trainarr.append(image)
        trainclass.append(13)
    elif count>=160 : 
        valarr.append(image)
        valclass.append(13)    

for count,i in enumerate(listdir('../input/imagesss015/15/')):
    image=cv2.imread('../input/imagesss015/15/'+i)
    image=cv2.resize(image,(224,224))
    if count>0 and count<140:
        trainarr.append(image)
        trainclass.append(14)
    elif count>=160 : 
        valarr.append(image)
        valclass.append(14)    





# In[ ]:


len(valarr)


# In[ ]:





# In[ ]:


valarr=np.array(valarr)


# In[ ]:


trainarr=np.array(trainarr)


# In[ ]:


for count,i in enumerate(valarr):
    valarr[count]=i/255


# In[ ]:


for count,i in enumerate(trainarr):
    trainarr[count]=i/255


# In[ ]:





# In[ ]:


model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11),                  strides=(4, 4), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())
model.summary()


# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='valid',data_format = 'channels_last'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(5, 5), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())



# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(224 * 224 * 3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.5))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.5))
# Batch Normalisation
model.add(BatchNormalization())
# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.5))
model.add(Dropout(0.5))
# Batch Normalisation
model.add(BatchNormalization())



# Output Layer
model.add(Dense(15))
model.add(Activation('softmax'))


# In[ ]:


y = to_categorical(trainclass, num_classes=15)


# In[ ]:


z = to_categorical(valclass, num_classes=15)


# In[ ]:


model.summary()


# In[ ]:


model.compile(
    optimizer=Adam(lr=0.0001,decay=0.0002),
    loss='categorical_crossentropy',
    metrics=[categorical_crossentropy,
             categorical_accuracy])
model.summary()


# In[ ]:


model.fit(
trainarr,
y,
validation_data=(valarr,z),
batch_size=256,

epochs=50
)


# In[ ]:


best_save_model_file = '../working/mymodel2.h5'


# z = to_categorical(valclass, num_classes=3)

#  model.predict(valarr[1:70])

# In[ ]:


model.predict(valarr[0:60])


# In[ ]:


from matplotlib.pyplot import imshow
imshow(trainarr[0])


# In[ ]:





# In[ ]:


from os import listdir
listdir('../input/trainimage1')

