from __future__ import print_function, division
from builtins import range, input

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

import os
data_path = os.path.abspath('/kaggle/input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images')

vgg = VGG16(input_shape=[224,224,3], weights='imagenet', include_top=False)

# For not training the VGG weights
for layer in vgg.layers:
  layer.trainable = False

x = Flatten()(vgg.output)
prediction = Dense(5, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)

model.summary()

model.compile(
  loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy'])


datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function = preprocess_input)


train_generator = datagen.flow_from_directory(
    data_path,
    subset='training',
    target_size=[224,224],
    classes = ['Mild','No_DR','Moderate','Proliferate','Severe'],
    class_mode = 'categorical')

valid_generator = datagen.flow_from_directory(
    data_path, 
    subset='validation',
    target_size=[224,224],
    classes = ['Mild','No_DR','Moderate','Proliferate','Severe'],
    class_mode = 'categorical')


r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=8,
  steps_per_epoch= 1,
  verbose = 1)


plt.plot(r.history['accuracy'])
plt.plot(r.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of Epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()