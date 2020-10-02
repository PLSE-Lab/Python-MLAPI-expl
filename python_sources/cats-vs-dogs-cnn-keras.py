#!/usr/bin/env python
# coding: utf-8

# ## Using Keras to do image classification
# This kernel uses Keras within Tensorflow to build CNN to build a model to do image classification.
# 
# Here are the steps:
# 1. Load dataset, both training and testing(validation) dataset
# 2. Image preprocess with data argumentation API - ImageDataGenerator
# 3. Setup model parameters
# 4. Train the model
# 5. Plot accuracy line to check how the model is working
# 6. Save model to a file
# 
# These are typical steps for image classification.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

#ignore all warning informations
import warnings 
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.models import load_model


# In[ ]:


train_dir='../input/cat-and-dog/training_set/training_set/'
test_dir='../input/cat-and-dog/test_set/test_set/'


train_cat_dir=train_dir+'cats'
train_dog_dir=train_dir+'dogs'

train_cat_fnames=os.listdir(train_cat_dir)
train_dog_fnames=os.listdir(train_dog_dir)

# test_cat_fnames=os.listdir(test_dir+'cats')
# test_dog_fnames=os.listdir(test_dir+'dogs')


# In[ ]:


nrows=4
ncols=4
pic_index=0


# Let's see how the pictures look like

# In[ ]:


fig=plt.gcf()
fig.set_size_inches(nrows*4,ncols*4)

pic_index+=8

next_cat_pix=[os.path.join(train_cat_dir,fname) for fname in train_cat_fnames[pic_index-8:pic_index]]
next_dog_pix=[os.path.join(train_dog_dir,fname) for fname in train_dog_fnames[pic_index-8:pic_index]]

for i,img_path in enumerate(next_cat_pix+next_dog_pix):
    sp=plt.subplot(nrows,ncols,i+1)
    sp.axis('off')
    
    img=mpimage.imread(img_path)
    plt.imshow(img)
    


# We use ImageDataGenerator to do data argumentation,please note we only apply argumentation to training data, and only rescalling with testing(validation) data

# In[ ]:


train_datagen=ImageDataGenerator(rescale=1.0/255.0,#rescaling factor
                                rotation_range=45, #Int. Degree range for random rotations.
                                width_shift_range=0.2, #fraction of total width
                                height_shift_range=0.2,#fraction of total height
                                shear_range=0.2, #Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
                                zoom_range=0.2,#Float or [lower, upper]. Range for random zoom
                                horizontal_flip=True,#Boolean. Randomly flip inputs horizontally.
                                fill_mode='nearest' #Points outside the boundaries of the input are filled according to the given mode
                                )

test_datagen=ImageDataGenerator(rescale=1.0/255.0)

#ImageDataGenarator will generate dataset, as well as generate training labels based on the path,for example,
#our training directory is ../input/cat-and-dog/training_set/training_set/, there are two directory under it, cats and dogs,
#then the generator will generate data with label 'cats' and 'dogs', and with all pictures under each directory respectively
train_generator=train_datagen.flow_from_directory(train_dir,
                                                target_size=(150,150),
                                                batch_size=20,
                                                class_mode='binary')

test_generator=test_datagen.flow_from_directory(test_dir,
                                                target_size=(150,150),
                                                batch_size=20,
                                                class_mode='binary')


# We will use three CNN with three Maxpooling layers first, then get it flattened,then one fully connected layer with 512 neurons.  
# for input layer and all hidden layers, we will use relu as activation function. 
# In the output layer, we will use sigmoid to activate it.

# In[ ]:


model=tf.keras.models.Sequential([
    keras.layers.Conv2D(32,(3,3),input_shape=(150,150,3),activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64,(3,3),activation="relu"),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(128,(3,3),activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(128,(3,3),activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')    
])


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


# In[ ]:


history=model.fit_generator(train_generator,
                  epochs=150,
                  steps_per_epoch=100,
                  validation_data=test_generator,
                  validation_steps=50,
                  verbose=1
                 )


# Based the result above, the training is quite good because accurary of validation set is higher than that of training set.
# I only trained for 150 epochs and you can try to set it o 200 or more, to see how accurate you can get at the end.
# 
# I plot the result as below, as you can see, the accuracy of both training and validation are still growing,which means we defenity can improve that by adding more epochs, and at the sametime, there's no sigh of overfitting, which is great.

# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')


# In[ ]:


#Save model to a file
model.save('cat-dogs-model.h5') 


# # If you think this kernel helped you, I will be gratefull if you upvote it, and any comments are welcome.
