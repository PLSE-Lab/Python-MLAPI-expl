#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os, shutil
original_dataset_dir = '/kaggle/input/plant-seedlings-classification'


# In[ ]:


import keras.models


# In[ ]:


base_dir = '/kaggle/input/base'
os.mkdir(base_dir)


# In[ ]:


train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)


# In[ ]:


train_Black_grass_dir = os.path.join(train_dir, 'Black-grass')
train_Charlock_dir = os.path.join(train_dir, 'Charlock')
train_Cleavers_dir = os.path.join(train_dir, 'Cleavers')
train_Common_Chickweed_dir = os.path.join(train_dir, 'Common Chickweed')
train_Common_wheat_dir = os.path.join(train_dir, 'Common wheat')
train_Fat_Hen_dir = os.path.join(train_dir, 'Fat Hen')
train_Loose_Silky_bent_dir = os.path.join(train_dir, 'Loose Silky-bent')
train_Maize_dir = os.path.join(train_dir, 'Maize')
train_Scentless_Mayweed_dir = os.path.join(train_dir, 'Scentless Mayweed')
train_Shepherds_Purse_dir = os.path.join(train_dir, 'Shepherds Purse')
train_Small_flowered_Cranesbill_dir = os.path.join(train_dir, 'Small-flowered Cranesbill')
train_Sugar_beet_dir = os.path.join(train_dir, 'Sugar beet')


# In[ ]:


os.mkdir(train_Black_grass_dir)
os.mkdir(train_Charlock_dir)
os.mkdir(train_Cleavers_dir)
os.mkdir(train_Common_Chickweed_dir)
os.mkdir(train_Common_wheat_dir)
os.mkdir(train_Fat_Hen_dir)
os.mkdir(train_Loose_Silky_bent_dir)
os.mkdir(train_Maize_dir)
os.mkdir(train_Scentless_Mayweed_dir)
os.mkdir(train_Shepherds_Purse_dir)
os.mkdir(train_Small_flowered_Cranesbill_dir)
os.mkdir(train_Sugar_beet_dir)


# In[ ]:


validation_Black_grass_dir = os.path.join(validation_dir, 'Black-grass')
validation_Charlock_dir = os.path.join(validation_dir, 'Charlock')
validation_Cleavers_dir = os.path.join(validation_dir, 'Cleavers')
validation_Common_Chickweed_dir = os.path.join(validation_dir, 'Common Chickweed')
validation_Common_wheat_dir = os.path.join(validation_dir, 'Common wheat')
validation_Fat_Hen_dir = os.path.join(validation_dir, 'Fat Hen')
validation_Loose_Silky_bent_dir = os.path.join(validation_dir, 'Loose Silky-bent')
validation_Maize_dir = os.path.join(validation_dir, 'Maize')
validation_Scentless_Mayweed_dir = os.path.join(validation_dir, 'Scentless Mayweed')
validation_Shepherds_Purse_dir = os.path.join(validation_dir, 'Shepherds Purse')
validation_Small_flowered_Cranesbill_dir = os.path.join(validation_dir, 'Small-flowered Cranesbill')
validation_Sugar_beet_dir = os.path.join(validation_dir, 'Sugar beet')


# In[ ]:


os.mkdir(validation_Black_grass_dir)
os.mkdir(validation_Charlock_dir)
os.mkdir(validation_Cleavers_dir)
os.mkdir(validation_Common_Chickweed_dir)
os.mkdir(validation_Common_wheat_dir)
os.mkdir(validation_Fat_Hen_dir)
os.mkdir(validation_Loose_Silky_bent_dir)
os.mkdir(validation_Maize_dir)
os.mkdir(validation_Scentless_Mayweed_dir)
os.mkdir(validation_Shepherds_Purse_dir)
os.mkdir(validation_Small_flowered_Cranesbill_dir)
os.mkdir(validation_Sugar_beet_dir)


# In[ ]:


import shutil 
import os 

for file_dir in os.listdir('/kaggle/input/plant-seedlings-classification/train'):
    l = int(len(os.listdir('/kaggle/input/plant-seedlings-classification/train/'+file_dir))*0.7)
    count = 0
    for img in os.listdir('/kaggle/input/plant-seedlings-classification/train/'+file_dir):
        if img.endswith(".png"):
            if count < l:
                src_dir = "/kaggle/input/plant-seedlings-classification/train/"+file_dir+'/'+img
                dst_dir = train_dir+'/'+(os.path.basename(os.path.normpath(file_dir)))
                count += 1
                shutil.copy(src_dir,dst_dir)


# In[ ]:


for img in os.listdir('/kaggle/input/plant-seedlings-classification/test'):
    if img.endswith(".png"):
        src = '/kaggle/input/plant-seedlings-classification/test/'+ img
        dst = '/kaggle/input/base/test'
        shutil.copy(src,dst)
        


# In[ ]:


os.listdir('/kaggle/input/base/test')


# In[ ]:


for file_dir in os.listdir('/kaggle/input/plant-seedlings-classification/train'):
    l = int(len(os.listdir('/kaggle/input/plant-seedlings-classification/train/'+file_dir))*0.7)
    count = 0
#     print (l)
    for img in os.listdir('/kaggle/input/plant-seedlings-classification/train/'+file_dir):
        if img.endswith(".png"):
            if count >= l:
                src_dir = "/kaggle/input/plant-seedlings-classification/train/"+file_dir+'/'+img
                dst_dir = validation_dir+'/'+(os.path.basename(os.path.normpath(file_dir)))
#                 print (count)
                shutil.copy(src_dir,dst_dir)
            count += 1
            
#                 


# In[ ]:


print('total training cat images:', len(os.listdir(train_Black_grass_dir)))


# In[ ]:


i = 0
for dirs in os.listdir(validation_dir):
    for img in os.listdir(validation_dir+'/'+dirs):
        i += 1
print (i)


# In[ ]:


len(os.listdir('/kaggle/input/plant-seedlings-classification/train/Black-grass'))


# In[ ]:


from PIL import Image

im = Image.open('/kaggle/input/plant-seedlings-classification/train/Black-grass/d3c72d4c3.png')
width, height = im.size
print (width)
print (height)


# In[ ]:


num_class = 12


# In[ ]:


from keras import layers
from keras import models
from keras.layers.core import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(84, 84, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(layers.Dense(12, activation='softmax'))
# model.add(Activation(activation='softmax'))
# model.add(layers.Dense(12, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


from keras import optimizers
model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.RMSprop(lr=1e-4),
                metrics=['acc'])


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(84, 84),
                batch_size=20,
                class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
                validation_dir,
                target_size=(84, 84),
                batch_size=20,
                class_mode='categorical')


# In[ ]:


for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


# simple ANN

# In[ ]:


history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=10,
            validation_data=validation_generator,
            validation_steps=50)


# In[ ]:


import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


model.save('/kaggle/input/simple.h5')


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')


# In[ ]:


from keras import layers
from keras import models
from keras.layers.core import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras import optimizers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(84, 84, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(12, activation='softmax'))
model.compile(loss='categorical_crossentropy',
optimizer=optimizers.RMSprop(lr=1e-4),
metrics=['acc'])


# In[ ]:


train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(84, 84),
batch_size=32,
class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
validation_dir,
target_size=(84, 84),
batch_size=32,
class_mode='categorical')


history = model.fit_generator(
train_generator,
steps_per_epoch=100,
epochs=30,
validation_data=validation_generator,
validation_steps=50)


# In[ ]:


model.save('/kaggle/input/simple_augmentation.h5')


# In[ ]:


from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
include_top=False,
input_shape=(84, 84, 3))


# In[ ]:


conv_base.summary()


# In[ ]:


from keras import models
from keras import layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(12, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


conv_base.trainable = False


# 1. Fine tuning 

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(84, 84),
batch_size=20,
class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
validation_dir,
target_size=(84, 84),
batch_size=20,
class_mode='categorical')
model.compile(loss='categorical_crossentropy',
optimizer=optimizers.RMSprop(lr=2e-5),
metrics=['acc'])

history = model.fit_generator(
train_generator,
steps_per_epoch=100,
epochs=50,
validation_data=validation_generator,
validation_steps=50)


# In[ ]:


conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


# In[ ]:


model.compile(loss='categorical_crossentropy',
optimizer=optimizers.RMSprop(lr=1e-5),
metrics=['acc'])
history = model.fit_generator(
train_generator,
steps_per_epoch=100,
epochs=50,
validation_data=validation_generator,
validation_steps=50)


# In[ ]:


model.save("/kaggle/input/fine_tuned.h5")


# In[ ]:


import pickle


# In[ ]:


filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[ ]:


test_generator = test_datagen.flow_from_directory(
validation_dir,
target_size=(84, 84),
batch_size=20,
class_mode='categorical')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)


# In[ ]:


os.listdir('/kaggle/input')


# In[ ]:


submission = pd.read_csv('/kaggle/input/plant-seedlings-classification/sample_submission.csv')


# In[ ]:


submission.shape


# In[ ]:


from keras.models import load_model
from keras.preprocessing import image
import numpy as np


# In[ ]:



def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(84, 84))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor


# In[ ]:


img_width, img_height = 84, 84
model = load_model('/kaggle/input/fine_tuned.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[ ]:


my_dict = train_generator.class_indices


# In[ ]:


key_list = list(my_dict.keys()) 
val_list = list(my_dict.values()) 


# In[ ]:


from IPython.display import FileLink
FileLink('/kaggle/input/fine_tuned.h5')


# In[ ]:


count = 0
for sub_img in submission['file']:
    img_path = test_dir+'/'+sub_img
    print (img_path)
    new_image = load_image(img_path)
    pred = model.predict(new_image)
    labels = np.argmax(pred, axis=-1)  
    name = key_list[val_list.index(labels)]
    print (name)
    submission['species'][count] = name
    count += 1
    


# In[ ]:


submission['species'][793]


# In[ ]:


submission.to_csv("submission.csv")


# In[ ]:


from IPython.display import FileLink
FileLink('submission.csv')

