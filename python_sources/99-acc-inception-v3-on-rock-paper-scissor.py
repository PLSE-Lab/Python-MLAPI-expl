#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# > I will use PreTrained Model Inception Netowrk to train my model. Off Course because we need to go deeper :)
# 
# ![](https://miro.medium.com/max/1656/1*uW81y16b-ptBDV8SIT1beQ.png)
# 
# Inceptionv3 is a convolutional neural network for assisting in image analysis and object detection, and got its start as a module for Googlenet. It is the third edition of Google's Inception Convolutional Neural Network, originally introduced during the ImageNet Recognition Challenge. Just as ImageNet can be thought of as a database of classified visual objects, Inception helps classification of objects in the world of computer vision. One such use is in life sciences, where it aids in the research of Leukemia. It was "codenamed 'Inception' after the film of the same name"

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os

from tensorflow.keras import layers
from tensorflow.keras import Model


# In[ ]:


rock_dir = os.path.join('../input/rock-paper-scissor/rps/rps/rock')
paper_dir = os.path.join('../input/rock-paper-scissor/rps/rps/paper')
scissors_dir = os.path.join('../input/rock-paper-scissor/rps/rps/scissors')

print('Total training rock images:', len(os.listdir(rock_dir)))
print('\nTotal training paper images:', len(os.listdir(paper_dir)))
print('\nTotal training scissors images:', len(os.listdir(scissors_dir)))


# In[ ]:


rock_files = os.listdir(rock_dir)
print("Some Rock File Names\n", rock_files[:5])

paper_files = os.listdir(paper_dir)
print("\nSome Paper File Names\n",paper_files[:5])

scissors_files = os.listdir(scissors_dir)
print("\nSome Scissor File Names\n",scissors_files[:5])


# > Let us see few images :)

# In[ ]:


import matplotlib.image as mpimg

pic_index = 3

next_rock = [os.path.join(rock_dir, fname) 
                for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) 
                for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) 
                for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()


# # Image Preprocessing
# 
# I will use ImageDataGenerator to make it easier to work with images on the go as well as i will also use Data Augmentation.

# In[ ]:


import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator


# In[ ]:


TRAINING_DIR = "../input/rock-paper-scissor/rps/rps/"
training_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

VALIDATION_DIR = "../input/rock-paper-scissor/rps-test-set/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    class_mode='categorical',
  batch_size=100
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150,150),
    class_mode='categorical',
    batch_size=50
)


# # Inception V3
# 
# Let us load Inception V3 Pre-trained model.

# In[ ]:


get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5     -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False


# > I am using 'mixed7' as last layer from Inception network and i am freezing layers to not retrain them.
# 
# **Use summery() to see layers.**

# In[ ]:


#pre_trained_model_tra.summary()


# In[ ]:


last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


# # DNN Model
# 
# Let us add our own model at last layer of Inception.

# In[ ]:


from tensorflow.keras.optimizers import RMSprop
adam = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense  (3, activation='softmax')(x)        

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = adam, 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])


# In[ ]:


history = model.fit(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 25,
            epochs = 10,
            validation_steps = 10,
            verbose = 1)


# In[ ]:


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.style.use('seaborn-whitegrid')
plt.figure(dpi = 100)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.show();


# > Do not worry why curve is fluctuating so much, this is because our data is different than what is is in Inception Netowrk. We can try much deeper network to our data to increase accuracy on Validation set. Can you see how good it is doing with adding just one layer after inception. That's the beauty of Transfer Learning.
# 
# **Thank you for reading my Notebook. I hope you learned something new. You can try tweaking parameters and play around them.**
