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
from PIL import Image
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense,Flatten
from keras.optimizers import Adam
from keras.preprocessing.image  import ImageDataGenerator
from keras.metrics import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.applications.vgg16 import decode_predictions
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def print_accuracy(c,d):
    print("accuracy: ",end=" "),
    print(((c/d)*100),end=""),
    print("%")
    


# In[ ]:





# In[ ]:


dirs=os.listdir( "../input/repository/spMohanty-PlantVillage-Dataset-442d23a/raw/color" )


# In[ ]:


classes_color=[]
for item in dirs:
    classes_color.append(item)


# In[ ]:


classes_color[1:5]


# In[ ]:


pil_im=Image.open("../input/repository/spMohanty-PlantVillage-Dataset-442d23a/raw/color/Tomato___Early_blight/55761153-0f6a-4dad-981e-16fde2ac683f___RS_Erly.B 6387.JPG")
imshow(np.asarray(pil_im))


# In[ ]:


from keras.models import load_model
loaded_model=load_model("../input/alexnet_20.h5")


# In[ ]:


img=np.array(pil_im)


# In[ ]:


image = np.expand_dims(img, axis=0)


# In[ ]:


classes_color[np.argmax(loaded_model.predict(image))]


# In[ ]:


IMAGE_WIDTH=256
IMAGE_HEIGHT=256


# In[ ]:


train_path= "../input/repository/spMohanty-PlantVillage-Dataset-442d23a/raw/color"
train_batch=ImageDataGenerator().flow_from_directory(train_path,  target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),batch_size=32,classes=classes_color)


# In[ ]:


test_imgs,test_labels=next(train_batch)


# In[ ]:


predictions=loaded_model.predict(test_imgs,steps=1,verbose=0)


# In[ ]:


y_pred=[]
for i in range(len(predictions)):
    y_pred.append(np.argmax(predictions[i]))


# In[ ]:


y_label=[]
for i in range(32):
    y_label.append(np.argmax(test_labels[i]))


# In[ ]:


c=0
for i in range(32):
    if(y_label[i]==y_pred[i]):
        c=c+1


# In[ ]:


print_accuracy(c,32)


# In[ ]:


print("actual_class",end ="                          "),print("predicted_class")
print()
for i in range(10):
    print(classes_color[y_label[i]],end ="                          ")
    print(classes_color[y_pred[i]])
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




