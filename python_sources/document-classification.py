#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform


# In[ ]:


Image_Files=[]
src1='../input/first-half-training/training_half/Training_half'
src2='../input/rvlcdip-iii/training_second_half/Training_second_half'
src3='../input/rvlcdip-ii/training/Training'


# In[ ]:


C={'letter': 0,
   'form': 1,
   'email': 2,
   'handwritten': 3,
   'advertisement': 4,
   'scientific report': 5,
   'scientific publication': 6,
   'specification': 7,
   'file folder': 8,
   'news article': 9,
   'budget': 10,
   'invoice': 11,
   'presentation': 12,
   'questionnaire': 13,
   'resume': 14,
   'memo': 15}


# In[ ]:


C={'form': 0,
   'budget': 1,
   'invoice': 2,
   }


# In[ ]:



for file in sorted(os.listdir(src1)):
    for k,v in C.items():
       if k in file:
           pair=dict()
           pair[src1+'/'+file]=v
           Image_Files.append(pair)
           break 
           


# In[ ]:


os.listdir('../input/document-classification')


# In[ ]:



for file in os.listdir(src2):
     for k,v in C.items():
        if k in file:
            pair=dict()
            pair[src2+'/'+file]=v
            Image_Files.append(pair)
            


# In[ ]:



for file in os.listdir(src3):
    for k,v in C.items():
        if k in file:
            pair=dict()
            pair[src3+'/'+file]=v
            Image_Files.append(pair)  


# **Check the size of Image Files**

# In[ ]:


len(Image_Files)


# In[ ]:


from sklearn.utils import shuffle
Image_Files=shuffle(Image_Files)


# In[ ]:


Image_Files


# **Plotting some Images**

# In[ ]:


temp_dict=Image_Files[400]
for k,v in temp_dict.items():
   image=plt.imread(k)
   print(image.shape)
   plt.imshow(image)


# In[ ]:


import cv2
image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
plt.imshow(image)


# **Generator Definiton**

# In[ ]:


def generator(batchsize):
    i = 0 
    while True: 
        Xtrain = np.ndarray(shape=(batchsize,224,224,3),dtype='float32') 
        Ytrain = np.ndarray(shape=(batchsize,3),dtype='float32') 
        count = 0 
        while count < batchsize: 
            temp_dict = Image_Files[i]
            for k,v in temp_dict.items():
                image = plt.imread(k)
                label = v
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)    
            newImage = skimage.transform.resize(image, (224, 224), mode='constant') 
            Xtrain[count] = newImage 
            new_label=np.zeros(3)
            new_label[label]=1;
            Ytrain[count] = new_label
            count = count + 1 
            i = (i+1)%len(Image_Files) 
        yield (Xtrain, Ytrain)  


# In[ ]:


import keras
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
 


# In[ ]:


from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v2 import MobileNetV2


# In[ ]:





# 

# **Model Training**

# In[ ]:


from keras.models import load_model
path='../input/document-classification/Epoch6_to_10_ResNet50.h5'
model=load_model(path)   
batch=64
epoch=14
history = model.fit_generator( generator(batch), steps_per_epoch = (len(Image_Files)//batch),initial_epoch=10, epochs = epoch, verbose = 1)


# In[ ]:



os.system('echo saving_model')
model.save('Epoch11_to_15_ResNet50.h5')

