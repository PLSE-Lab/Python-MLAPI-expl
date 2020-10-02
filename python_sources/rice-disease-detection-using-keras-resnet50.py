#!/usr/bin/env python
# coding: utf-8

# Import neccessary packages

# In[ ]:


import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Input, Convolution2D, GlobalAveragePooling2D, Dense , DepthwiseConv2D
from keras.models import Model
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[ ]:


EPOCHS = 20
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
directory_root = '../input/rice-diseases-image-dataset/labelledrice/'
width=256
height=256
depth=3


# Function to convert images to array

# In[ ]:


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json


# In[ ]:


# path for Kaggle kernels
input_path = "../input/rice-diseases-image-dataset/labelledrice/Labelled"


# In[ ]:


# DATA AUGMENTATION
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    shear_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    input_path,
    target_size=(256, 256),
    batch_size=BS,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    input_path, # same directory as training data
    target_size=(256, 256),
    batch_size=BS,
    class_mode='categorical',
    subset='validation') # set as validation data


# In[ ]:


conv_base = ResNet50(include_top=False, weights='imagenet')

for layer in conv_base.layers:
    layer.trainable = False


# In[ ]:


x = conv_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
predictions = layers.Dense(4, activation='softmax')(x)
model = Model(conv_base.input, predictions)


# Get Size of Processed Image

# In[ ]:


model.summary()


# In[ ]:


optimizer = keras.optimizers.Adam()
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


# In[ ]:


from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('best_model.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto')  


# In[ ]:


history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=10,
                              epochs=20,
                              validation_data=validation_generator,
                              validation_steps=10,
                              callbacks=[checkpoint]
                             )


# Plot the train and val curve

# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()


# Model Accuracy

# In[ ]:


# print("[INFO] Calculating model accuracy")
# scores = model.evaluate_generator(validation_generator, steps=10)
# print(f"Test Accuracy: {scores[1]*100}")


# In[ ]:


from keras.models import load_model
new_model = load_model('best_model.h5')
print("[INFO] Calculating model accuracy")
scores = new_model.evaluate_generator(validation_generator, steps=10)
print(f"Test Accuracy: {scores[1]*100}")


# Save model using Pickle

# In[ ]:


# save the model to disk
print("[INFO] Saving model...")
pickle.dump(new_model,open('cnn_model_73_50.pkl', 'wb'))

