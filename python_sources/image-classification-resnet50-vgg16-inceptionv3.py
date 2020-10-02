#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import glob
import zipfile
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array


# In[ ]:


# Download ResNet50 weights
get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5     -O /kaggle/working/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Download Inception_v3 weights
get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5     -O /kaggle/working/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Download VGG16 weights
get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5     -O /kaggle/working/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


# In[ ]:


# Define train and validation directories
import os

train_dir = os.path.join('../input/intel-image-classification/seg_train/seg_train')
validation_dir = os.path.join('../input/intel-image-classification/seg_test/seg_test')


# In[ ]:





# Directory with our training mountain pictures
train_mountain_dir = os.path.join('../input/intel-image-classification/seg_train/seg_train/mountain')

# Directory with our training forest pictures
train_forest_dir = os.path.join('../input/intel-image-classification/seg_train/seg_train/forest')

# Directory with our validation mountain pictures
validation_mountain_dir = os.path.join('../input/intel-image-classification/seg_test/seg_test/mountain')

# Directory with our validation forest pictures
validation_forest_dir = os.path.join('../input/intel-image-classification/seg_test/seg_test/forest') 

train_mountain_fnames = glob.glob(train_mountain_dir+"/*")
train_forest_fnames = glob.glob(train_forest_dir+"/*")


# In[ ]:


sample_size = 5

sample_mountain_fnames = np.random.choice(train_mountain_fnames, size=sample_size, replace=False)
sample_forest_fnames = np.random.choice(train_forest_fnames, size=sample_size, replace=False)

sample_mountain_images = [img_to_array(load_img(fname, target_size=(150, 150), interpolation='bilinear')) for fname in sample_mountain_fnames]
sample_forest_images = [img_to_array(load_img(fname, target_size=(150, 150), interpolation='bilinear')) for fname in sample_forest_fnames]

sample_mountain_images = np.array(sample_mountain_images).astype('float32')/255.
sample_forest_images = np.array(sample_forest_images).astype('float32')/255.


# In[ ]:


fig, ax = plt.subplots(1, sample_size, figsize=(sample_size * 5, 15))
for i in range(sample_size):
  ax[i].imshow(sample_mountain_images[i])
 

fig, ax = plt.subplots(1, sample_size, figsize=(sample_size * 5, 15))
for i in range(sample_size):
  ax[i].imshow(sample_forest_images[i])
  


# In[ ]:


# Image augmentation techniques and data generators
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Note that the validation data should not be augmented!
validation_datagen = ImageDataGenerator(rescale = 1.0/255.)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 32,
                                                    class_mode = 'categorical', 
                                                    target_size = (150, 150))    

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              batch_size=32,
                                                              class_mode='categorical',
                                                              target_size=(150, 150))


# In[ ]:


# Build models
vgg16_weights_path = '/kaggle/working/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
inception_v3_weights_path = '/kaggle/working/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
resnet50_weights_path = '/kaggle/working/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

vgg16_model = VGG16(input_shape = (150, 150, 3),
                    include_top = False,
                    weights = None)

inception_v3_model = InceptionV3(input_shape = (150, 150, 3),
                                 include_top = False,
                                 weights = None)

resnet50_model = ResNet50(input_shape = (150, 150, 3),
                          include_top = False,
                          weights = None)

vgg16_model.load_weights(vgg16_weights_path)
inception_v3_model.load_weights(inception_v3_weights_path)
resnet50_model.load_weights(resnet50_weights_path)


# In[ ]:


# Plotting the model architecture: vgg16
plot_model(vgg16_model, to_file='/tmp/vgg16_model.png')


# In[ ]:


# Plotting the model architecture: inception_v3
plot_model(inception_v3_model, to_file='/tmp/inception_v3_model.png')


# In[ ]:


# Plotting the model architecture: resnet50
plot_model(resnet50_model, to_file='/tmp/resnet50_model.png')


# In[ ]:


# Describe vgg16
vgg16_model.summary()


# In[ ]:


# Describe inception_v3
inception_v3_model.summary()


# In[ ]:


# Describe resnet50
resnet50_model.summary()


# In[ ]:


# Freeze layers
for layer in vgg16_model.layers:
  layer.trainable = False

for layer in inception_v3_model.layers:
  layer.trainable = False

for layer in resnet50_model.layers:
  layer.trainable = False


# In[ ]:


vgg16_cut_layer = 'block4_pool'
inception_v3_cut_layer = 'mixed7'
resnet50_cut_layer = 'conv5_block3_add'

def compile_model(pre_trained_model, cut_layer):
  last_layer = pre_trained_model.get_layer(cut_layer)
  last_output = last_layer.output

  x = layers.Flatten()(last_output)
  x = layers.Dense(1024, activation='relu')(x)
  x = layers.Dropout(0.2)(x)                  
  x = layers.Dense(6, activation='softmax')(x)           

  model = Model(pre_trained_model.input, x) 
  model.compile(optimizer = RMSprop(lr=0.0001), 
                loss = 'categorical_crossentropy', 
                metrics = ['accuracy'])
  return model

transfered_vgg16_model = compile_model(vgg16_model, vgg16_cut_layer)
transfered_inception_v3_model = compile_model(inception_v3_model, inception_v3_cut_layer)
transfered_resnet50_model= compile_model(resnet50_model, resnet50_cut_layer)


# In[ ]:


# Describe transfered_vgg16_model
transfered_vgg16_model.summary()


# In[ ]:


transfered_inception_v3_model.summary()


# In[ ]:


# Describe transfered_resnet50_model
transfered_resnet50_model.summary()


# In[ ]:



     history_vgg16 = transfered_vgg16_model.fit(
                            train_generator,
                            validation_data = validation_generator,
                            epochs = 20,
                            validation_steps = 50,
                            verbose = 2)
    
    


# In[ ]:



     # history_resnet50 = transfered_resnet50_model.fit(
                           # train_generator,
                            #validation_data = validation_generator,
                           # epochs = 20,
                            #validation_steps = 50,
                            #verbose = 2)
        


# In[ ]:



      history_inception_v3 = transfered_inception_v3_model.fit(
                            train_generator,
                            validation_data = validation_generator,
                            epochs = 20,
                            validation_steps = 50,
                             verbose = 2)
        


# In[ ]:


def plot_acc(history, model_name):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(acc))

  plt.plot(epochs, acc, 'r', label='Training accuracy')
  plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
  plt.title(f'Training and validation accuracy [{model_name}]')
  plt.legend(loc=0)
  plt.figure()
  plt.show()

plot_acc(history_vgg16, 'VGG16')
plot_acc(history_inception_v3, 'Inception_v3')
plot_acc(history_resnet50, 'ResNet50')


# In[ ]:


sample_size = 5
model = transfered_vgg16_model 

sample_mountain_fnames = np.random.choice(train_mountain_fnames, size=sample_size, replace=False)
sample_forest_fnames = np.random.choice(train_forest_fnames, size=sample_size, replace=False)

sample_mountain_images = [img_to_array(load_img(fname, target_size=(150, 150), interpolation='bilinear')) for fname in sample_mountain_fnames]
sample_forest_images = [img_to_array(load_img(fname, target_size=(150, 150), interpolation='bilinear')) for fname in sample_forest_fnames]

sample_mountain_images = np.array(sample_mountain_images).astype('float32')/255.
sample_forest_images = np.array(sample_forest_images).astype('float32')/255.



mountain_pred = model.predict(sample_mountain_images).flatten()
forest_pred = model.predict(sample_forest_images).flatten()

fig, ax = plt.subplots(1, sample_size, figsize=(sample_size * 5, 15))
for i in range(sample_size):
  ax[i].imshow(sample_mountain_images[i])
  ax[i].set_title(f"Mountain: {mountain_pred[i]:.2f}")

fig, ax = plt.subplots(1, sample_size, figsize=(sample_size * 5, 15))
for i in range(sample_size):
  ax[i].imshow(sample_forest_images[i])
  ax[i].set_title(f"Forest: {forest_pred[i]:.2f}")

