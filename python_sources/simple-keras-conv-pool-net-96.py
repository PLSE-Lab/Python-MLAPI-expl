'''
A very simple keras standard net with some conv and pool layers. 
The keras ImageDataGenerator was used to reduce the effort of obtaining the image data.
It also splits the data into training and validation.
'''


import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import History
import numpy as np

batch_size = 50
num_classes = 2
image_size = 64
K.set_image_dim_ordering('th') 

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,  
    zoom_range=0.2,        
    horizontal_flip=True,
    validation_split=0.3)  

train_generator = datagen.flow_from_directory(
    '../input/cell_images/cell_images',  
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = datagen.flow_from_directory(
    '../input/cell_images/cell_images',
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(
    3, image_size, image_size), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
              
history=History()

history=model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=40,
    validation_data=validation_generator,
    validation_steps=800 // batch_size)

print(np.amax(history.history.get('val_acc')))
