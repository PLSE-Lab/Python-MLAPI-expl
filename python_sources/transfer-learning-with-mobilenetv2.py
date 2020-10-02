#!/usr/bin/env python
# coding: utf-8

# ## Demonstrartion of Transefer Learning With MobileNetV2

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/garbage classification/Garbage classification"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install "tensorflow_hub==0.4.0"')


# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub
import keras
import glob,os, random
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


# In[ ]:


base_dir = '../input/garbage classification/Garbage classification'
img_list = glob.glob(os.path.join(base_dir,'*/*.jpg'))
print(len(img_list))


# In[ ]:


for i,img_path in enumerate(random.sample(img_list,6)):
    img = image.load_img(img_path)
    img = image.img_to_array(img,dtype=np.uint8)
    
    plt.subplot(2,3,i+1)
    plt.imshow(img.squeeze())


# In[ ]:


trainImage_data_gen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.1,
    zoom_range = 0.1,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip=True,
    vertical_flip = True,
    validation_split = 0.1

)

testImage_data_gen = ImageDataGenerator(
    rescale = 1./255,
    validation_split =0.1
)

train_data_gen =trainImage_data_gen.flow_from_directory(
    base_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical',
    subset='training',
    seed=0
)
val_data_gen = testImage_data_gen.flow_from_directory(
    base_dir,
    target_size=(224,224),
    batch_size = 16,
    class_mode='categorical',
    subset='validation',
    seed=0
)


# In[ ]:


labels = (train_data_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())

print(labels)


# In[ ]:


URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor = hub.KerasLayer(URL,input_shape=(224,224,3))


# In[ ]:


feature_extractor.trainable = False


# In[ ]:


model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(6, activation='softmax')
])


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


# In[ ]:


model.summary()


# Tune the Epochs parameters to increase the accuracy

# In[ ]:


batch_size = 16
history = model.fit_generator(train_data_gen, 
                              epochs=10, 
                              validation_data=val_data_gen,
                             )


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[ ]:


test_x, test_y = val_data_gen.__getitem__(1)

prediction = model.predict(test_x)

plt.figure(figsize=(16, 16))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.title('pred:%s / truth:%s' % (labels[np.argmax(prediction[i])], labels[np.argmax(test_y[i])]))
    plt.imshow(test_x[i])


# In[ ]:




