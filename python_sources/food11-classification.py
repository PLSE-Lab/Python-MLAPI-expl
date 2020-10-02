#!/usr/bin/env python
# coding: utf-8

# # Food-11 Classification

# ## using EfficientNet B7

# In[ ]:


import os
print(os.listdir('../input/food11-image-dataset'))


# In[ ]:


trainPath = '../input/food11-image-dataset/training'
validPath = '../input/food11-image-dataset/validation'
testPath  = '../input/food11-image-dataset/evaluation'


# In[ ]:


Foods = os.listdir(trainPath)
Foods.sort()
print(Foods)
labels = Foods


# ## Prepare Data

# In[ ]:


import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical


# ### Data Augmentation

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# In[ ]:


target_size=(224,224)
batch_size = 16


# In[ ]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    trainPath,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',    
    shuffle=True,
    seed=42,
    class_mode='categorical')


# In[ ]:


valid_datagen = ImageDataGenerator(rescale=1./255)

valid_generator = valid_datagen.flow_from_directory(
    validPath,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical')


# In[ ]:


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    testPath,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',    
    class_mode='categorical')


# ## Build Model

# In[ ]:


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, BatchNormalization, Activation, LeakyReLU, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


get_ipython().system('pip install -q efficientnet')
import efficientnet.tfkeras as efn


# In[ ]:


num_classes = 11
input_shape = (224,224,3)


# In[ ]:


# Build Model
net = efn.EfficientNetB7(input_shape=input_shape, weights='imagenet', include_top=False)

# add two FC layers (with L2 regularization)
x = net.output
x = GlobalAveragePooling2D()(x)

x = Dense(256)(x)
x = Dense(32)(x)

# Output layer
out = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=net.input, outputs=out)
model.summary()


# In[ ]:


# Compile Model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


## set Checkpoint : save best only, verbose on
#checkpoint = ModelCheckpoint("food11_vgg16.hdf5", monitor='accuracy', verbose=0, save_best_only=True, mode='auto', save_freq=1)


# ## Train Model

# In[ ]:


num_train = 9866
num_valid = 3430
num_epochs= 20


# In[ ]:


# Train Model
history = model.fit_generator(train_generator,steps_per_epoch=num_train // batch_size,epochs=num_epochs, validation_data=valid_generator, validation_steps=num_valid // batch_size) #, callbacks=[checkpoint])


# ## Save Model

# In[ ]:


## Save Model
model.save('food11.h5')


# ## Evaluate Model

# In[ ]:


score = model.evaluate(valid_generator)


# ### Confusion Matrix (validation set)

# In[ ]:


predY=model.predict(valid_generator)
y_pred = np.argmax(predY,axis=1)
#y_label= [labels[k] for k in y_pred]
y_actual = valid_generator.classes
cm = confusion_matrix(y_actual, y_pred)
print(cm)


# In[ ]:


print(classification_report(y_actual, y_pred, target_names=labels))


# ## Test Model

# In[ ]:


score = model.evaluate(test_generator)


# ### Confusion Matrix (test set)

# In[ ]:


predY=model.predict(test_generator)
y_pred = np.argmax(predY,axis=1)
#y_label= [labels[k] for k in y_pred]
y_actual = test_generator.classes
cm = confusion_matrix(y_actual, y_pred)
print(cm)


# In[ ]:


print(classification_report(y_actual, y_pred, target_names=labels))


# In[ ]:




