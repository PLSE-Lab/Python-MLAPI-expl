#!/usr/bin/env python
# coding: utf-8

# **Author - Rahul Mishra**
# 
# Here I will augment the images and use inception v3 model from tensorflow and demonstrate a transfer learning in practice

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
"""
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


os.listdir('/kaggle/working')


# 1. move all image to working dir
# 2. use image generators scale and augment the data (training and validation)
# 3. build an inception model

# In[ ]:


#shutil.rmtree('/kaggle/working/intel-image-classification/')


# In[ ]:



## run only once
### move all images to working dir (change this code according to your needs)

src_dir = '/kaggle/input/'
output_dir = '/kaggle/working/intel-image-classification'

try :
    shutil.copytree(src_dir, output_dir)
except : 
    pass


# In[ ]:



print(os.listdir('/kaggle/working/intel-image-classification/intel-image-classification/seg_test/seg_test'))


# **use image generators for scaling and augmentation**

# In[ ]:


train_dir = '/kaggle/working/intel-image-classification/intel-image-classification/seg_train/seg_train'
validation_dir = '/kaggle/working/intel-image-classification/intel-image-classification/seg_test/seg_test'
train_dir


# In[ ]:


# get count of distinct class

num = []
name = []

for d in os.listdir(train_dir):
    name.append(d)
    num.append(len(os.listdir(os.path.join(train_dir, d))))


sns.barplot(name, num)
plt.title('Class distribution')
plt.xlabel('classes')
plt.ylabel('number os samples')
plt.show()


# here are the parameters Ill pass for image augmentation in my (ImageDataGenerator) functions below
# 
# 1. rescale -> to normalize images since larger values will dominate smaller values and cause loss of information
# 2. rotation_range -> rotate the images randomly to 30 degrees
# 3. width_shift_range -> randomly shift the images left <-> right uptp 20%
# 4. heigth_shift_range -> randomly shift the image up <-> down by 20%
# 5. shear_range -> tilt the image back <-> front bt 20%
# 6. horizontal_flip -> flip the image hrizontally
# 7. vertical_flip -> flip the image vertically

# In[ ]:



train_gen = 
tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                           rotation_range=30,
                                                           width_shift_range=0.2,
                                                           height_shift_range=0.2,
                                                           shear_range=0.2,
                                                           zoom_range=0.3,
                                                           horizontal_flip=True,
                                                           vertical_flip=True)

train_data = 
train_gen.flow_from_directory(directory=train_dir,
                                    target_size=(150,150),
                                    batch_size=32,
                                    class_mode='categorical')

# remember - do not augment the validation set
validation_gen = 
tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

validation_data = 
validation_gen.flow_from_directory(directory=validation_dir,
                                                    target_size=(150,150),
                                                    batch_size=32,
                                                    class_mode='categorical')


# **Build model**

# In[ ]:


# we will use tranfer learning from an inceptionV3 network


pretrained_model = 
tf.keras.applications.InceptionV3(include_top=False,
                                                    input_shape=(150,150,3),
                                                    weights='imagenet')

# set the layers to non-trainable since they already carry weights from ImageNet

for layer in pretrained_model.layers:
    layer.trainable = False

#print the original model structure -> very long
#pretrained_model.summary()


# Ok! we dont really need such a huge network and hence we shall use only a 50% of its layers

# In[ ]:


last_layer = pretrained_model.get_layer('mixed5')
last_layer_output = last_layer.output
print(last_layer_output)


# In[ ]:


# add you own layers to the end

# flatten the last layer to match the output 
x = tf.keras.layers.Flatten()(last_layer_output)
# add your fully connected layer
x = tf.keras.layers.Dense(1280, activation=tf.keras.activations.relu)(x)
# add a dropout layer
x = tf.keras.layers.Dropout(0.2)(x)
# add a prediction layer
x = tf.keras.layers.Dense(6, activation=tf.keras.activations.softmax)(x)

model = tf.keras.Model(pretrained_model.input, x)

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

# print summary of the model -> very long
# model.summary()


# In[ ]:


threshold = .8
class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > threshold :
            self.model.stop_training = True
            print('\n Stopping training as the model reached ', str(threshold), '% accuracy ')
            
mycallback_func = mycallback()


# In[ ]:


# train model
history = model.fit(train_data, validation_data=validation_data, epochs=10, callbacks=[mycallback_func], batch_size=32, verbose=1)


# In[ ]:


# plot accuracy

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training accuracy', 'Validation accuracy'])
plt.show()


# In[ ]:


# plot loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training accuracy', 'Validation accuracy'])
plt.show()


# In[ ]:





# In[ ]:




