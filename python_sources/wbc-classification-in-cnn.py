#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
import os
import time
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras import backend as K
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

if not os.path.isdir('models'):
    os.mkdir('models')
print('Is using GPU?', tf.test.is_gpu_available())


# In[ ]:


img_width, img_height = 120, 160

train_data_dir = '../input/wbc-dataset/data/train'
validation_data_dir = '../input/wbc-dataset/data/validation'
nb_train_samples = 9957
nb_validation_samples = 2487
epochs = 20
batch_size = 32
#regularizer = tf.keras.regularizers.l2(0.01,)

if K.image_data_format() == 'channels_first':
  input_shape = (3, img_width, img_height)
else:
  input_shape = (img_width, img_height, 3)


# In[ ]:


def create_model():
    
    def add_conv_block(model, num_filters):
        
        model.add(Conv2D(num_filters, 3, activation='relu', padding='same'))            #, kernel_regularizer=regularizer
        model.add(BatchNormalization())
        model.add(Conv2D(num_filters, 3, activation='relu', padding='valid'))           #,kernel_regularizer=regularizer
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.3))

        return model
    
    model = tf.keras.models.Sequential()
    model.add(Input(shape=input_shape))
    
    model = add_conv_block(model, 32)
    model = add_conv_block(model, 64)
    model = add_conv_block(model, 128)

    model.add(Flatten())
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model()
model.summary()


# In[ ]:


train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.3,
    zoom_range = 0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[ ]:


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    #color_mode = 'grayscale',
    class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    #color_mode = 'grayscale',
    class_mode = 'categorical')


# In[ ]:


get_ipython().run_cell_magic('time', '', "h = model.fit_generator(\n    train_generator,\n    steps_per_epoch = nb_train_samples // batch_size,\n    epochs = epochs,\n    validation_data = validation_generator,\n    validation_steps = nb_validation_samples // batch_size,\n    callbacks=[\n        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5),\n        tf.keras.callbacks.ModelCheckpoint(filepath = '/kaggle/working/model_{val_accuracy:.3f}.h5', save_best_only=True,\n                                          save_weights_only=False, monitor='val_accuracy')\n    ])\n\n#model.save_weights('first_try_all_ball_test1.h5')")


# In[ ]:


accs = h.history['accuracy']
val_accs = h.history['val_accuracy']

plt.plot(range(len(accs)),accs, label = 'Training')
plt.plot(range(len(accs)),val_accs, label = 'Validation')
plt.legend()
plt.show()


# In[ ]:


accs = h.history['loss']
val_accs = h.history['val_loss']

plt.plot(range(len(accs)),accs, label = 'Training')
plt.plot(range(len(accs)),val_accs, label = 'Validation')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




