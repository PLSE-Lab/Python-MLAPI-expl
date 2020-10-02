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


# **Prepare Dataset**

# In[ ]:


get_ipython().system('git clone https://github.com/spMohanty/PlantVillage-Dataset')


# In[ ]:


import shutil
import glob

if (os.path.exists(os.getcwd()+"/TomatoDataset") is False):
    os.mkdir("TomatoDataset")

    for path in glob.glob(os.getcwd()+"/PlantVillage-Dataset/raw/color/Tomato*"):
        shutil.move(path, os.getcwd()+"/TomatoDataset/"+path.split('/')[-1])


# In[ ]:


get_ipython().system('rm -r PlantVillage-Dataset #delete the cloned repository to free space')


# In[ ]:


# I am using split-folders for the task, you could use other methods
get_ipython().system(' pip install split-folders')

get_ipython().system(' split_folders ./TomatoDataset --ratio .6 .2 .2')

# rename the folder to for better description
# split-folder output folder is named output, we had to change that
get_ipython().system(' mv output TomatoDatasetTrainTestValSplit')

# the folder structure will be
# |--TomatoDatasetTrainTestValSplit
# |  |--train
# |     |--{the tomato classes folder1}
# |     |--{the tomato classes folder2}  
# |     |             .
# |     |             .
# |     |             .
# |     |--{the tomato classes folderN}
# |  |--test
# |     |--{the tomato classes folder1}
# |     |--{the tomato classes folder2}
# |     |             .
# |     |             .
# |     |             .
# |     |--{the tomato classes folderN}
# |  |--val
# |     |--{the tomato classes folder1}
# |     |--{the tomato classes folder2}
# |     |             .
# |     |             .
# |     |             .
# |     |--{the tomato classes folderN}


# **Start Building Model**

# In[ ]:


train_data_dir = './TomatoDatasetTrainTestValSplit/train'
val_data_dir = './TomatoDatasetTrainTestValSplit/val'
test_data_dir = './TomatoDatasetTrainTestValSplit/test'


# In[ ]:


import math 

batch_size = 64
image_size = (224,224)

num_train_samples = sum([len(files) for root, directory, files in os.walk(train_data_dir)])
num_train_steps = math.floor(num_train_samples/batch_size)

num_val_samples = sum([len(files) for root, directory, files in os.walk(val_data_dir)])
num_val_steps = math.floor(num_val_samples/batch_size)


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data_generator = ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
     height_shift_range=0.1,
    rescale=1./255,
     fill_mode='constant'
)

train_generator = train_data_generator.flow_from_directory(
    train_data_dir, # the location of the training folder
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse', # there are 2 types, sparse or binary (if we are predicting only disease and non disease, thats when we use binary, else we use sparse when its more than 2)
    color_mode='rgb', # we can skip this line cos our image is rgb, but if we want ti to be grey we can change it
    shuffle=True # this just determines the order in which the generator will give us the images, random is ususlly better
)
train_generator.class_indices

val_data_generator = ImageDataGenerator(rescale=1./255)

val_generator = val_data_generator.flow_from_directory(
    val_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    color_mode='rgb',
    shuffle=True
)


# In[ ]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense ,BatchNormalization, Dropout
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import EarlyStopping

input_tensor = Input(shape=(224,224,3))

base_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

for i, layer in enumerate(base_model.layers):
  print(i, layer.name)


# In[ ]:


for layer in base_model.layers:
   layer.trainable=True
 
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(10,activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[ ]:


epochs = 1000

early_stopping =  EarlyStopping(monitor='val_accuracy', patience=15, mode='auto', min_delta=0.95)


print('starting model training and saving history')
history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[early_stopping]
)


# In[ ]:


history.history.keys()


# In[ ]:


import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()


# In[ ]:


import pandas as pd

if (os.path.exists(os.getcwd()+"/models") is False):
  os.mkdir("models")
  pd.DataFrame(history.history).to_csv('./models/inveption_v3_tl_history_0_10.csv')
  model.save("./models/inveption_v3_tl_model_0_10.h5")

