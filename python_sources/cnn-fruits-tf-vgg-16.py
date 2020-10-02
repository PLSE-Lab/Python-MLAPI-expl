#!/usr/bin/env python
# coding: utf-8

# # CNN Fruits VGG-16

# In[ ]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(np.array(Image.open("../input/fruits360/fruits.jpg")))


# ## Goal
# 
# Goal of this notebook is to show the VGG-16 layers implementation with Tensorflow Keras.<br>
# Use VGG-16 layers to train and detect Fruits.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD


# In[ ]:


import glob
import cv2
import os


# In[ ]:


training_dir = '../input/fruits/fruits-360/Training/'
validation_dir = '../input/fruits/fruits-360/Test/'
test_dir = '../input/fruits/fruits-360/test-multiple_fruits/'


# # Architecture

# In[ ]:


plt.figure(figsize=(20,10))
plt.axis('off')
plt.imshow(np.array(Image.open("../input/fruits360/vgg-16.png")))


# # Implementation

# In[ ]:


def create_model():
    # Instantiate an empty sequential model
    model = Sequential()

    # block
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', input_shape=(64,64, 3)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    # block
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    # block
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    # block
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    # block
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(MaxPool2D((2,2), strides=(2,2)))

    # block #6 (classifier)
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(131, activation='softmax'))

    # reduce learning rate by 0.1 when the validation error plateaus
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1))

    # set the SGD optimizer with lr of 0.01 and momentum of 0.9
    optimizer = SGD(lr = 0.01, momentum = 0.9)

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

training_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2, 
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    preprocessing_function=preprocess_input)

validation_datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function=preprocess_input)


# In[ ]:


IMAGE_SIZE = [64, 64]


# In[ ]:


training_generator = training_datagen.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = 200, class_mode = 'categorical')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size = IMAGE_SIZE, batch_size = 200, class_mode = 'categorical')


# In[ ]:


training_images = 37836
validation_images = 12709


# In[ ]:


checkpoint_path = '../input/fruits360/fruits.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=1)


# In[ ]:


def train(model):
    model.fit(training_generator,
               steps_per_epoch = training_images,
               epochs = 2,
               validation_data = validation_generator,
               validation_steps = validation_images,
               callbacks=[cp_callback])
    return mdeol


# In[ ]:


def load(model):
    model.load_weights(checkpoint_path)
    return model


# In[ ]:


# train or load model
model = create_model()

# model = train(model)
model = load(model)

loss,acc = model.evaluate(validation_generator, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# # Predict

# In[ ]:


idx_to_name = {x:i for (x,i) in enumerate(training_generator.class_indices)}

def predict(img):
    to_predict = np.zeros(shape=training_generator[0][0].shape)
    to_predict[0] = img
    
    return idx_to_name[np.argmax(model(to_predict)[0])]


# In[ ]:


img = cv2.imread('../input/fruits/fruits-360/Training/Banana/0_100.jpg')
resized = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA) 


# In[ ]:


predict(resized)


# In[ ]:


plt.imshow(resized)

