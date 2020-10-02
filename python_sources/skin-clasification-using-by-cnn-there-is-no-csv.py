#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense ,LeakyReLU
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from glob import glob
from keras.applications import InceptionV3


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# 
# 
# 
# **Firt of all  define test path and train path **

# In[ ]:


train_path = "/kaggle/input/skin-cancer9-classesisic/skin cancer isic the international skin imaging collaboration/Skin cancer ISIC The International Skin Imaging Collaboration/Train/"
test_path = "/kaggle/input/skin-cancer9-classesisic/skin cancer isic the international skin imaging collaboration/Skin cancer ISIC The International Skin Imaging Collaboration/Test/"


# **Testing paths and images **
# 
# 

# In[ ]:


img = load_img(train_path + "nevus/ISIC_0000041.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()


# **convert images to array** 
# 
# 
# **show to shape** 

# In[ ]:


x = img_to_array(img)
print(x.shape)


# 
# **Using the glob function, we learn how many different folders there are in the dataset.**

# In[ ]:



className = glob(train_path + '/*' )
numberOfClass = len(className)
print("NumberOfClass: ",numberOfClass)


# **we are building the cnn structure**

# In[ ]:




model = Sequential()
model.add(InceptionV3(include_top=False, input_shape=(299,299,3)))
model.add(Flatten())
model.add(Dense(32))
model.add(LeakyReLU(0.001))
model.add(Dense(16))
model.add(LeakyReLU(0.001))
model.add(Dense(numberOfClass, activation='softmax'))
model.layers[0].trainable = False

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()


# **define loss and optimizer method ...**

# In[ ]:


model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])
batch_size = 250


# **We get various images by zooming and rotating and flipping **

# In[ ]:


train_datagen = ImageDataGenerator(rescale= 1./255,
                   shear_range = 0.3,
                   horizontal_flip=True,
                   zoom_range = 0.3)

test_datagen = ImageDataGenerator(rescale= 1./255)

train_generator = train_datagen.flow_from_directory(
        train_path, 
        target_size=(299,299),
        batch_size = batch_size,
        color_mode= "rgb",
        class_mode= 'categorical')

test_generator = test_datagen.flow_from_directory(
        test_path, 
        target_size=(299,299),
        batch_size = batch_size,
        color_mode= "rgb",
        class_mode= 'categorical')

hist = model.fit_generator(
        generator = train_generator,
        steps_per_epoch = 5000,
        epochs=1,
        validation_data = test_generator,
        validation_steps = 250)


# **if you want to save the train weights like me, you must push to internet button in setting on right vertical menu ** 

# In[ ]:


model.save_weights("weights.h5")


# accuracy= 0,20 this is mean that dataset is not corrrect separetaly pehh ;)
