#!/usr/bin/env python
# coding: utf-8

# So we have flow_from_directory for I/O stage, CNN that gets 0.74 val accuracy for 5 epochs,
# then do image augmentations with ImageDataGenerator, then retrain CNN

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


train_cats_dir = '../input/cat-and-dog/training_set/training_set/cats/'  # directory with our training cat pictures
train_dogs_dir = '../input/cat-and-dog/training_set/training_set/dogs/'  # directory with our training dog pictures
validation_cats_dir = '../input/cat-and-dog/test_set/test_set/cats/' # directory with our validation cat pictures
validation_dogs_dir = '../input/cat-and-dog/test_set/test_set/dogs/'  # directory with our validation dog pictures
train_dir = '../input/cat-and-dog/training_set/training_set/'
validation_dir = '../input/cat-and-dog/test_set/test_set/'


# In[ ]:


num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val


# In[ ]:


print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)


# In[ ]:


train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
#remember the rescale is because we want not values between 0-255, but between 0-1 !!
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data


# In[ ]:


batch_size = 128
epochs = 15
IMG_HEIGHT = 64
IMG_WIDTH = 64
#changing from 50x50 to 64x64


# # So now we use flow_from_directory because the cat images are in the cat directory, and the dog images in dog directory. So the directory names provide the class label (cat or dog). Otherwise, if filenames and class labels in a CSV we could use flow_from_dataframe instead.

# In[ ]:





# In[ ]:


train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           seed=42,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')


# In[ ]:


val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              seed=42,
                                                              class_mode='binary')


# #  Look at some of the images now

# In[ ]:


sample_training_images, _ = next(train_data_gen)


# In[ ]:


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# In[ ]:


plotImages(sample_training_images[:50])


# In[ ]:


from keras import layers, models
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, ZeroPadding2D
from keras.models import Model

import keras.backend as K
from keras.models import Sequential


# # Here we make the CNN model!

# In[ ]:


#model = models.Sequential()
#model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
#model.add(layers.MaxPooling2D((2,2)))
#model.add(layers.Conv2D(64, (3,3), activation='relu'))
#model.add(layers.MaxPooling2D((2,2)))
#model.add(layers.Conv2D(64, (3,3), activation='relu'))
#model.add(layers.Flatten())
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(2, activation='softmax'))

#model = Sequential([
#    Conv2D(16, 3, padding='same', activation='relu', input_shape=(150, 150 ,3)),
#    MaxPooling2D(),
#    Conv2D(32, 3, padding='same', activation='relu'),
#    MaxPooling2D(),
#    Conv2D(64, 3, padding='same', activation='relu'),
#    MaxPooling2D(),
#    Flatten(),
#    Dense(512, activation='relu'),
#    Dense(1) #why is it 1 and not 2 since 2 classes?
#])

#model = Sequential([
#    Conv2D(16, 3, padding='same', activation='relu', input_shape=(150, 150 ,3)),
#    MaxPooling2D((2,2)),
#    Conv2D(32, (3,3), padding='same', activation='relu'),
#    MaxPooling2D((2,2)),
#    Conv2D(64, (3,3), padding='same', activation='relu'),
#    MaxPooling2D(2,2),
#    Flatten(),
#    Dense(512, activation='relu'),
#    Dense(1) #why is it 1 and not 2 since 2 classes?
#])

#model.summary()
#model = Sequential()

#model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (50, 50, 3)))

#model.add(BatchNormalization(axis = 3, name = 'bn0'))
#model.add(Activation('relu'))

#model.add(MaxPooling2D((2, 2), name='max_pool'))
#model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
#model.add(Activation('relu'))
#model.add(AveragePooling2D((3, 3), name='avg_pool'))

#model.add(Flatten())
#model.add(Dense(500, activation="relu", name='rl'))
#model.add(Dropout(0.8))
#model.add(Dense(2, activation='softmax', name='sm'))

#model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
#model.summary()

#model=Sequential()
#model.add(Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
#model.summary()


# # Training the CNN. It's just put into history so can do visualizations etc afterwards

# In[ ]:


#history = model.fit_generator(
#    train_data_gen,
#    steps_per_epoch=total_train // batch_size,
 #   epochs=epochs,
#    validation_data=val_data_gen,
#   validation_steps=total_val // batch_size
#)

history = model.fit_generator(generator=train_data_gen,
                    steps_per_epoch=total_train // batch_size,
                    validation_data=val_data_gen,
                    validation_steps=total_val // batch_size,
                    epochs=5)


# # Ok well it was 0.74 val_accuracy after 5 epochs. Now do some augmentations with ImageDataGenerator

# In[ ]:


image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.2,
                    shear_range=0.2
                    )


# In[ ]:


image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    #rotation_range=45,
                    #width_shift_range=.15,
                    #height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.2
                    #change from 0.5 to 0.2
                    )


# In[ ]:


train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           seed=42,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')


# In[ ]:


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# In[ ]:


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)


# # Train the CNN with the augmented images. Actually got kinda worse! Maybe too many epochs? Or was it due to augmentation?

# In[ ]:


history = model.fit_generator(generator=train_data_gen,
                    steps_per_epoch=total_train // batch_size,
                    validation_data=val_data_gen,
                    validation_steps=total_val // batch_size,
                    epochs=10)


# In[ ]:


#Let me alter the model! This is the first one! Total train params at 2.4M ish
#model=Sequential()
#model.add(Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
#model.summary()

#another one which was very poor. Total trainable params at 500k ish
#model=Sequential()
#model.add(Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.5))
#model.add(Conv2D(32, (2, 2), activation = 'relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
#model.summary()

#Another model, with total trainable params at 187k ish. Result was 0.7055 w 10 epochs
#model = Sequential()
#model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
#model.summary()

#repeating the above but with two dropouts instead of one. 187k trainable params. Result was 0.71 w 10 epochs. Attempt with 15 epochs 
#and we get 0.75

model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
model.summary()

#do the model from the https://www.kaggle.com/sangwookchn/convolutional-neural-networks-cnn-keras notebook next! No dropouts 2.3M
#trainable params
#model = Sequential()
#model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
#note is 2,2
#note this one had 64x64 not 50x50
#model.add(MaxPooling2D(pool_size = (2,2)))
#rest is same as mine
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
#model.summary()


# In[ ]:


history = model.fit_generator(generator=train_data_gen,
                    steps_per_epoch=total_train // batch_size,
                    validation_data=val_data_gen,
                    validation_steps=total_val // batch_size,
                    epochs=15)


# In[ ]:





# Things to do:
# 1. Visualisations 
# 2. Alter image sizes 100x100 vs. 50x50 which is what we started with and got our best val_accuracy of 0.74 with 5 epochs
# 3. Different augmentations
# 4. Other drop outs 
# 
# Can check this one https://www.kaggle.com/sangwookchn/convolutional-neural-networks-cnn-keras which seems to have val_accuracy of 1.0 with input shape of 64x64, Convolution2D(32, 3, 3,), maxpooling 2x2, augmentations are ImageDataGenerator( 
# shear_range = 0.2, 
# zoom_range = 0.2, 
# horizontal_flip = True). NB our input shape is 50x50
# 
# Which is vs. mine which are  rotation_range=45,
#                     width_shift_range=.15,
#                     height_shift_range=.15,
#                     horizontal_flip=True,
#                     zoom_range=0.5
#                     )
# 

# In[ ]:


#let me alter augmentations then per the notebook I refer to above. Will alter the model later based on that notebook

image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2
                    #rotation_range=45,
                    #width_shift_range=.15,
                    #height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.2
                    #change from 0.5 to 0.2
                    )


# In[ ]:


train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           seed=42,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')


# In[ ]:


history = model.fit_generator(generator=train_data_gen,
                    steps_per_epoch=total_train // batch_size,
                    validation_data=val_data_gen,
                    validation_steps=total_val // batch_size,
                    epochs=15)


# In[ ]:


#ok let me change to 64x64 and try this again! Ok that's done and nowhere near 1.0 on val_accuracy after couple of epochs!

