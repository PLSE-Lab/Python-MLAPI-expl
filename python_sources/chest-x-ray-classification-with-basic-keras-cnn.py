#!/usr/bin/env python
# coding: utf-8

# In[38]:


from keras.preprocessing.image import load_img, ImageDataGenerator
from keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Dropout, Flatten
from keras.models import Sequential

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

print(os.listdir("../input"))
print(os.listdir("../input/chest_xray/chest_xray/"))


# In[39]:


path_to_data = "../input/chest_xray/chest_xray/"


# Displaying Normail Image

# In[40]:


img_normal_path = path_to_data + "train/NORMAL/NORMAL2-IM-0927-0001.jpeg"
img_normal = load_img(img_normal_path)
print("Normal Image")
plt.imshow(img_normal)
plt.show()


# Displaying PNEUMONIA Image

# In[41]:


img_other_path = path_to_data + "train/PNEUMONIA/person478_virus_975.jpeg"
img_other = load_img(img_other_path)
print("PNEUMONIA Image")
plt.imshow(img_other)
plt.show()


# ### Setting Up Hyper-parameters

# In[42]:


img_width, img_height = 128, 128
batch_size = 16
epochs = 20


# In[43]:


# Getting number of training and test samples
nb_train_samples = len(os.listdir(path_to_data + "train/NORMAL")) + len(os.listdir(path_to_data + "train/PNEUMONIA"))
nb_test_samples = len(os.listdir(path_to_data + "test/NORMAL")) + len(os.listdir(path_to_data + "test/PNEUMONIA"))

print("Training samples: " + str(nb_train_samples))
print("Testing samples: " + str(nb_test_samples))


# ## Creating Basic CNN Model

# In[44]:


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape= (img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.summary()


# In[45]:


model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])


# ## Setting Up Data Generators with some augmentations

# In[46]:


train_datagen = ImageDataGenerator(
        horizontal_flip= True,
        rescale= 1. / 255,
        shear_range= 0.25,
        zoom_range= 0.2,
)


# In[47]:


test_datagen = ImageDataGenerator(
        rescale= 1. / 255,
)


# In[48]:


train_gen = train_datagen.flow_from_directory(
    path_to_data + "train/",
    target_size= (img_width, img_height),
    batch_size= batch_size,
    shuffle= True,
    class_mode= "binary"
)


# In[49]:


valid_gen = test_datagen.flow_from_directory(
    path_to_data + "val/",
    target_size= (img_width, img_height),
    batch_size= batch_size,
    shuffle= True,
    class_mode= "binary"
)


# In[50]:


test_gen = test_datagen.flow_from_directory(
    path_to_data + "test/",
    target_size= (img_width, img_height),
    batch_size= batch_size,
    shuffle= True,
    class_mode= "binary",
)


# In[51]:


model.fit_generator(
    train_gen,
    epochs= epochs,
    steps_per_epoch= nb_train_samples // batch_size,
    validation_data= valid_gen,
    validation_steps = nb_test_samples // batch_size
)


# In[52]:


model.save_weights("model_2.h5")


# In[53]:


# Evaluating the model
scores = model.evaluate_generator(generator=test_gen, steps=nb_test_samples // batch_size)
print("Test accuracy is {}".format(scores[1] * 100))


# In[ ]:




