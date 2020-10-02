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


os.listdir("../input/train")


# In[ ]:


os.mkdir("md")
os.listdir("md")


# In[ ]:


d=pd.read_csv("../input/labels.csv")


# In[ ]:


print(d)


# In[ ]:


for i in d.iloc[:,0]:
    print(i)


# In[ ]:


labels=d.columns[1]
print(labels)


# In[ ]:


from pandas import get_dummies
indices=d.index.tolist()
print(indices)


# In[ ]:


indices=np.array(indices)
print(indices)


# In[ ]:


y = d.reindex(indices)[labels]
print(y)


# In[ ]:


y = get_dummies(y)
print(y)


# In[ ]:


x=y[:0]
print(x)


# In[ ]:


for i in x:
    print(i)


# In[ ]:


for i in x:
    os.mkdir("md/"+i)


# In[ ]:


os.listdir("md")


# In[ ]:


from shutil import copyfile
for file in os.listdir("../input/train"):
    name=file.split(".")[0]
    for i in range(10222):
        if name==d.iloc[i,0]:
            for j in x:
                if j==d.iloc[i,1]:
                    filename='../input/train/'+file
                    copyfile(filename,'md/'+j+"/"+file)


# In[ ]:


os.listdir("md/pug")


# In[ ]:


import matplotlib.pyplot as plt
from PIL import Image
image=Image.open("md/pug/f6e3a909254785d410b2418647034a5a.jpg")
plt.imshow(image)
plt.show()


# In[ ]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255)


# In[ ]:


train_generator=train_datagen.flow_from_directory("md/",batch_size=20,target_size=(150,150),
                                                  class_mode='categorical')


# In[ ]:


model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(120,activation='sigmoid')
])


# In[ ]:


model.summary()


# In[ ]:


from tensorflow.keras.optimizers import RMSprop

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['accuracy'])


# In[ ]:


history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=15)


# In[ ]:


import os

from tensorflow.keras import layers
from tensorflow.keras import Model
get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5     -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
  
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (256, 256, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
  layer.trainable = False
  
# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

