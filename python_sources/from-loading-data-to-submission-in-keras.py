#!/usr/bin/env python
# coding: utf-8

# This Notebook contains begginer's code to get started with this Kaggle Competition using Keras.  
# Contents of Notebook:  
# 1. Load Data using ImageGenerator (Data Augmentation)  
# 2. Transfer Learning  
# 3. Build Model  
# 4. Train the Model  
# 5. Analyze Training  
# 6. Use the Trained Model to predict Test set and generate the Submission.csv file  

# Import Dependencies

# In[ ]:


#import dependencies
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop


# Create Dataframes for reading csv files

# In[ ]:


df=pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv') #training set list
columns=["healthy", "multiple_diseases", "rust", "scab"]
df_test=pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv') #testing set list

#appending .jpg so that we can dirctly read the images from "image_id" column
df['image_id'] = df['image_id'].astype(str)+".jpg"
df_test['image_id'] = df_test['image_id'].astype(str)+".jpg"


# Load Data using ImageGenerator (Data Augmentation)   
# 80:20 split between Training and Validation Data

# In[ ]:


#ImageGenerator

#data augmentation
datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_datagen=ImageDataGenerator(rescale=1./255.)

#training data 
train_generator=datagen.flow_from_dataframe(
    dataframe=df[:1460],
    directory='../input/plant-pathology-2020-fgvc7/images',
    x_col="image_id",
    y_col=columns,
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=(300,300))

#validation data
valid_generator=test_datagen.flow_from_dataframe(
    dataframe=df[1460:],
    directory='../input/plant-pathology-2020-fgvc7/images',
    x_col="image_id",
    y_col=columns,
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=(300,300))

#test data
test_generator=test_datagen.flow_from_dataframe(
    dataframe=df[:],
    directory='../input/plant-pathology-2020-fgvc7/images',
    x_col="image_id",
    batch_size=1,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(300,300))


# Transfer Learning (InceptionNet V3)

# In[ ]:


#download model
get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5     -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
  
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (300, 300, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)
  
#pre_trained_model.summary()


# Build Model

# In[ ]:


for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed10')
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 64 hidden units and ReLU activation
x = layers.Dense(64, activation='relu')(x)
# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x) 

# Add a final sigmoid layer for classification
x = layers.Dense  (4, activation='softmax')(x)           


model = Model( pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

#model.summary()


# Train the Model

# In[ ]:


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

history=model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=5)


# Analyze Training

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib as mpl
import matplotlib.pyplot as plt


print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))


# In[ ]:


f, ax = plt.subplots(figsize=(12,4)) # set the size that you'd like (width, height)

plt.title('Traing Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')

plt.plot(epochs,loss,label='Training Loss')
plt.plot(epochs, val_loss,label='Validation Loss')

plt.legend()
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(12,4)) # set the size that you'd like (width, height)

plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')

plt.plot(epochs,acc,label='Training Accuracy')
plt.plot(epochs, val_acc,label='Validation Accuracy')

plt.legend()
plt.show()


# Use the Trained Model to predict Test set and generate the Submission.csv file! 

# In[ ]:


SUB_PATH = "../input/plant-pathology-2020-fgvc7/sample_submission.csv"
sub = pd.read_csv(SUB_PATH)


# In[ ]:


test_generator.reset()
pred=model.predict_generator(test_generator,steps=STEP_SIZE_TEST,verbose=1)

sub.loc[:, 'healthy':] = pred
sub.to_csv('submission.csv', index=False) #submit this file
sub.head()


# Congratulation! You have come to the end of this notebook.

# Please do give suggestions and improvements regarding this Notebook! Thank You.
