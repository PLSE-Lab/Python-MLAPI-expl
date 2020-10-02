#!/usr/bin/env python
# coding: utf-8

# We start by loading the libraries and the Data :
# We are going to use CNN VGG19, which explain the use of the library keras.  

# In[ ]:


import numpy as np # linear algebra
from matplotlib import pyplot as plt
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import shutil
import multiprocessing as mp

import cv2
import numpy as np
import sklearn.metrics as metrics
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import load_model
from keras.models import Sequential
from keras import optimizers
from keras.utils import to_categorical
from sklearn.utils import class_weight
from keras import layers
from keras.layers import Dense, Dropout, Activation, GlobalAveragePooling2D, MaxPooling2D, Conv2D, Input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model

from keras.optimizers import SGD



from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50

get_ipython().run_line_magic('matplotlib', 'inline')

train_dir = '/kaggle/input/food11/training'
validation_dir = '/kaggle/input/food11/validation'

train_files = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
validation_files = [f for f in os.listdir(validation_dir) if os.path.isfile(os.path.join(validation_dir, f))]


# Here we are going to extract the labels and the names of the file names from training and validation set.

# In[ ]:


from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# label extraction
train = []
y_train = []
valid = []
y_valid = []

for file in train_files:
    train.append(file)
    label= file.find("_")
    y_train.append(int(file[0:label]))
for file in validation_files:
    valid.append(file)
    label= file.find("_")
    y_valid.append(int(file[0:label]))


# * We create containers where we store the arrays of the images in the shape (190,190,3)for both training and validation data.
# * We loaded images with PIL library.
# * We chose 190 because of lack of RAM and kernel crashing

# In[ ]:


cnnInput = np.ndarray(shape=(len(train), 190,190, 3), dtype=np.float32)
print('[INFO] Loading training images')
i=0
for file in train:
    image = cv2.imread(train_dir + "/" + file)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # do not normalize for this model, keep 0-255
    image = image.astype("float")
    image = cv2.resize(image, dsize=(190, 190), interpolation=cv2.INTER_CUBIC)
    # no normalization for this model, keep 0-255
    x = img_to_array(image)
    x = x.reshape((1, x.shape[0], x.shape[1],
                                   x.shape[2]))

    cnnInput[i]=x
    i+=1
print('[INFO] Done')


# In[ ]:


cnnValidation = np.ndarray(shape=(len(valid), 190,190, 3), dtype=np.float32)
print('[INFO] Loading validation images')
i=0
for file in valid:
    image = cv2.imread(validation_dir + "/" + file)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # do not normalize for this model, keep 0-255
    image = image.astype("float")
    image = cv2.resize(image, dsize=(190, 190), interpolation=cv2.INTER_CUBIC)
    # no normalization for this model, keep 0-255
    x = img_to_array(image)
    x = x.reshape((1, x.shape[0], x.shape[1],
                                   x.shape[2]))

    cnnValidation[i]=x
    i+=1
print('[INFO] Done')


# We used one hot encoding instead of number of the class for labels. To get an ouput of probabilities for belonging to each class.

# In[ ]:


y_train_2 = to_categorical(y_train)
y_valid_2 = to_categorical(y_valid)


# In[ ]:


vgg_model = VGG19(weights='imagenet', include_top=False)


# In[ ]:


# make explained variable hot-encoded
y_train_hot_encoded = to_categorical(y_train)
y_test_hot_encoded = to_categorical(y_valid)


# In[ ]:


class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)


# In[ ]:


# get layers and add average pooling layer
x = vgg_model.output
x = GlobalAveragePooling2D()(x)

# add fully-connected layer
x = Dense(2048, activation='relu')(x)
x = Dropout(0.3)(x)

# add output layer
predictions = Dense(11, activation='softmax')(x)

model = Model(inputs=vgg_model.input, outputs=predictions)

# freeze pre-trained model area's layer
#for layer in vgg_model.layers:
#    layer.trainable = False

# update the weight that are added
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
#model.fit(cnnInput, y_train_hot_encoded)

# choose the layers which are updated by training
#layer_num = len(model.layers)
#for layer in model.layers[:21]:
#    layer.trainable = False

#for layer in model.layers[21:]:
#    layer.trainable = True

# training
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
history= model.fit(cnnInput,y_train_hot_encoded, batch_size=64, shuffle=True,
                    validation_data=(cnnValidation, y_test_hot_encoded),
                  class_weight=class_weights, epochs=100)
#history = model.fit(cnnInput, y_train_hot_encoded, batch_size=256, epochs=50, shuffle=True,  validation_split=0.1)


# In[ ]:


# training
#history = model.fit(cnnInput, y_train_hot_encoded, batch_size=256, epochs=50, shuffle=True,  validation_split=0.1)


# In[ ]:


model.summary()


# In this part, we did data augmentation. Data augmentation is a way of creating new data with modifications:
# 
# * different orientations (horizontal_flip and vertical_flip)
# * With shift (randomly shift images horizontally or randomly shift images vertically)
# * Zoom We used this image generator for training and validation, but for validation image generator, we use the original images from validation dataset.

# In[ ]:


# Data augmentation
from keras.preprocessing.image import ImageDataGenerator
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=[.6, 1],
    vertical_flip=True,
    horizontal_flip=True)
train_generator = train_datagen.flow(cnnInput, y_train, batch_size=64, seed=11)
valid_datagen = ImageDataGenerator()
valid_generator = valid_datagen.flow(cnnValidation, y_valid, batch_size=64, seed=11)


# In[ ]:


train_datagen.fit(cnnInput)
valid_datagen.fit(cnnValidation)


# In[ ]:


model.fit_generator(train_datagen.flow(cnnInput, y_train_hot_encoded, batch_size=64), shuffle=True,
                    validation_data=valid_datagen.flow(cnnValidation, y_test_hot_encoded, batch_size=64),
                  class_weight=class_weights, epochs=20)

