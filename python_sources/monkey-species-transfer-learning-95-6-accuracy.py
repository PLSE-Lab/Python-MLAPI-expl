#!/usr/bin/env python
# coding: utf-8

# # Monkey species, using convnets
# ## This notebook is done mainly for self-learning, I hope others might find it useful.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Lets do a simple exploration of the files...

# In[ ]:


import pandas as pd
raw = pd.read_csv("../input/monkey_labels.txt", skipinitialspace=True)
raw


# In[ ]:


raw.columns


# ### Looks like the columns are looking funny... lets strip the names

# In[ ]:


raw = raw.rename(columns=lambda x: x.strip())
raw.columns


# In[ ]:


labels = pd.DataFrame()
labels["id"] = raw["Label"].str.strip()
labels["name"] = raw["Common Name"].str.strip()


# In[ ]:


labels


# In[ ]:


from IPython.display import Image, display
from keras.preprocessing.image import ImageDataGenerator
from os import listdir
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


TRAIN_DIR = "../input/training/training/"
VALIDATION_DIR = "../input/validation/validation/"
print(os.listdir(TRAIN_DIR))


# ### Time to show a couple of images

# In[ ]:


all_ids = labels["id"]

for my_id in labels["id"]:
    images_to_show = 5
    image_dir = TRAIN_DIR + "%s/" % my_id
    image_name = listdir(image_dir)[0]
    image_path = image_dir  + image_name
    print("Labels is %s" % my_id)
    display(Image(filename=image_path, width=300, height=300))


# ### Now we are going to use the ImageDataGenerator objects to prepare our datasets. This step is very important, and the arguments we provide are incredibly sensitive, we NEED to understand what is going on here, so spend time in this cell

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
BATCH_SIZE = 16


# ### Lets play with ImageDataGenerator and see what it can... generate for us

# In[ ]:




train_datagen = ImageDataGenerator(rescale=1./255,      # We need to normalize the data
                                    rotation_range=40,      # The rest of params will generate us artificial data by manipulating the image
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True, 
                                    fill_mode='nearest'
                                  )

validation_datagen = ImageDataGenerator(rescale=1./255, # We need to normalize the data
                                  )

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, 
                                                    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), 
                                                    batch_size = BATCH_SIZE, 
                                                    shuffle=True, # By shuffling the images we add some randomness and prevent overfitting
                                                    class_mode="categorical")

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
                                                    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), 
                                                    batch_size = BATCH_SIZE, 
                                                    shuffle=True,
                                                    class_mode="categorical")


# In[ ]:


training_samples = 1097
validation_samples = 272
total_steps = training_samples // BATCH_SIZE


# ### We will use the VGG16 network with pretrained weights, this will allow us to go quite quickly, it is important to mark every layer as NON-trainable, so we do not spend GPU cycles on them

# In[ ]:


from keras.applications import vgg16
model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), pooling="max")

for layer in model.layers[:-5]:
        layer.trainable = False
        
for layer in model.layers:
    print(layer, layer.trainable)


# In[ ]:


model.summary()


# ### Note how the output (final layer) is just a MaxPooling, this is of no use to use, we need now to add our OWN final layer which will consist on a Dense layer with 10 outputs (remember, we have 10 monkey species)

# In[ ]:


from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model, Sequential

# Although this part can be done also with the functional API, I found that for this simple models, this becomes more intuitive
transfer_model = Sequential()
for layer in model.layers:
    transfer_model.add(layer)
transfer_model.add(Dense(512, activation="relu"))  # Very important to use relu as activation function, search for "vanishing gradiends" :)
transfer_model.add(Dropout(0.5))
transfer_model.add(Dense(10, activation="softmax")) # Finally our activation layer! we use 10 outputs as we have 10 monkey species (labels)


# In[ ]:


transfer_model.summary()


# ### Note how the output is now a Dense layer of 10 units!
# ### Next thing, lets define our optimizer, I will go with Adam and a learning rate of 0.0001

# In[ ]:


from keras import optimizers
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)

transfer_model.compile(loss="categorical_crossentropy",
                      optimizer=adam,
                      metrics=["accuracy"])


# In[ ]:


model_history = transfer_model.fit_generator(train_generator, steps_per_epoch=training_samples // BATCH_SIZE,
                                            epochs=25,
                                            validation_data=validation_generator,
                                            validation_steps=validation_samples // BATCH_SIZE)


# ### Now, lets plot our results, that will give us and idea on how we were improving over time.

# In[ ]:


for metric in model_history.history.keys():
    print(metric)


# In[ ]:


import pandas as pd
history = pd.DataFrame()
history["acc"] = model_history.history["acc"]
history["val_acc"] = model_history.history["val_acc"]
history.head()


# In[ ]:


history.plot(figsize=(12, 6))


# ### Now, at this point seems like more epochs might help us, but we seem to have reach some sort of plateau, so the question is, can we do better? There are a few things to test
#  
# * Try bigger images
# * Bottleneck features
# * Everything else :)
# 
# ### However this notebook/kernel just wants to ilustrate how to reuse an existing network (in this case VGG16), so we will leave it like this.

# In[ ]:




