#!/usr/bin/env python
# coding: utf-8

# # You must catch them all.
# **A code-focused introduction to neural networks, with Pokemon.**  
# 
# - This tutorial demonstrates how to implement [transfer-learning](https://en.wikipedia.org/wiki/Transfer_learning) to quickly train an image recognition network with Keras.  
# - We'll use the magnificently curated [Complete Pokemon Image Dataset.](https://www.kaggle.com/mrgravelord/complete-pokemon-image-dataset)  
# - At the end of this notebook, you'll have a Pokemon-recognizing network ready to use on any `.jpg` or `.png` image.

# In[ ]:


import os
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.models import Model,load_model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense,Input,GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ReduceLROnPlateau


# # Prepare a data pipeline
# The objective here is to load the images from the `pokemon` folder. [Image data generators](https://keras.io/preprocessing/image/#imagedatagenerator-class) make this especially easy. They also provide quick access to many simple data augmentation options, like randomly flipping or rotating images. This helps simulate a larger dataset than we actually have, and thereby yields a higher-performing network.  
#   
# Fortunately, this dataset's folder structure is already in a great format to feed into Keras. It's something like this:  
# ```
# ../complete-pokemon-image-dataset/pokemon/
#                                         Phanpy/
#                                             Phanpy_1.jpg
#                                             Phanpy_2.jpg
#                                             ...
#                                         Charmander/
#                                             Charmander_1.jpg
#                                             Charmander_2.jpg
#                                             ...
#                                         ...
# ```
# 

# In[ ]:


batch_size = 24
num_classes = 928 # this many classes of Pokemon in the dataset

data_generator = ImageDataGenerator(rescale=1./255,
                                    horizontal_flip=True,
                                    vertical_flip=False,
                                    brightness_range=(0.5,1.5),
                                    rotation_range=10,
                                    validation_split=0.2) # use the `subset` argument in `flow_from_directory` to access

train_generator = data_generator.flow_from_directory('../input/complete-pokemon-image-dataset/pokemon',
                                                    target_size=(160,160), # chosen because this is size of the images in dataset
                                                    batch_size=batch_size,
                                                    subset='training')

val_generator = data_generator.flow_from_directory('../input/complete-pokemon-image-dataset/pokemon',
                                                    target_size=(160,160),
                                                    batch_size=batch_size,
                                                    subset='validation')


# # Transfer Learning
# [Keras Applications](https://keras.io/applications/) provides a variety of popular network architectures with pre-trained weights. Even though the weights are for detecting objects different from ours, starting from here (rather than some systematic [initialization](https://keras.io/initializers/)) saves a lot of time. After all, a tree is visually more similar to a spatula than to whiteness.

# In[ ]:


# import the base model and pretrained weights
custom_input = Input(shape=(160,160,3,))
base_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=custom_input, input_shape=None, pooling=None, classes=num_classes)


# Like our custom input, we need a few layers on the end. First, we need to collapse dimensions, which can be done with a global [pooling](https://keras.io/layers/pooling/) layer, or a `Flatten()` layer, and then cap it off with a prediction layer.

# In[ ]:


x = base_model.layers[-1].output # snag the last layer of the imported model

x = GlobalMaxPooling2D()(x)
x = Dense(1024,activation='relu')(x) # an optional extra layer
x = Dense(num_classes,activation='softmax',name='predictions')(x) # our new, custom prediction layer

model = Model(input=base_model.input,output=x)
# new model begins from the beginning of the imported model,
# and the predictions come out of `x` (our new prediction layer)

# let's train all the layers
for layer in model.layers:
    layer.training = True


# ### Optional: load pretrained model/weights
# [This](https://www.kaggle.com/kwisatzhaderach/inceptionv3-pokemon-modelweights) is this model and dataset, pretrained.

# In[ ]:


model = load_model('../input/inceptionv3-pokemon-modelweights/InceptionV3_Pokemon.h5')


# In[ ]:


# these are utilities to maximize learning, while preventing over-fitting
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=12, cooldown=6, rate=0.6, min_lr=1e-18, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=24, verbose=1)


# In[ ]:


# compile and train the network
model.compile(optimizer=Adam(1e-8),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_generator,
                    validation_data=val_generator,
                    steps_per_epoch=2000//batch_size,
                    validation_steps=800//batch_size,
                    epochs=1, # increase this if actually training
                    shuffle=True,
                    callbacks=[reduce_lr,early_stop],
                    verbose=0)


# In[ ]:


# here's how to save the model after training. Use ModelCheckpoint callback to save mid-training.
model.save('InceptionV3_Pokemon.h5')


# In[ ]:


# preprocessing and predicting function for test images:
def predict_this(this_img):
    im = this_img.resize((160,160)) # size expected by network
    img_array = np.array(im)
    img_array = img_array/255 # rescale pixel intensity as expected by network
    img_array = np.expand_dims(img_array, axis=0) # reshape from (160,160,3) to (1,160,160,3)
    pred = model.predict(img_array)
    return np.argmax(pred, axis=1).tolist()[0]

classes = [_class for _class in os.listdir('../input/complete-pokemon-image-dataset/pokemon/')]
classes.sort() # they were originally converted to number when loaded by folder, alphabetically


# In[ ]:


url = 'https://i.imgur.com/5Nycvcx.jpg'
response = requests.get(url)
img_1 = Image.open(BytesIO(response.content))

print("A wild {} appears!".format(classes[predict_this(img_1)]))
display(img_1)


# In[ ]:


# the same thing, as a reusable function
def identify(url):
    response = requests.get(url)
    _img = Image.open(BytesIO(response.content))
    print("A wild {} appears!".format(classes[predict_this(_img)]))
    display(_img)

identify("https://bit.ly/2VQ32fd")


# # Go forth, and catch them all!
