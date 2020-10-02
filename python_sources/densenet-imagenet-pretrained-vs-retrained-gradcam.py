#!/usr/bin/env python
# coding: utf-8

# # DenseNet Imagenet pretrained vs retrined & GradCam
# 
# Background : The objectives of the analysis in this notebook is to find the answers of simple question.
# 
# **Adding a couple of fully connected (Dense) Layer to the imagenet pretrained DenseNet model is good enough or do we need to train all layers of DenseNet?**
# 
# Steps
# 
# 1. Train the DenseNet with trainset with the same hyperparameters. The only difference would be setting all DenseNet layers "Trainable" or "Not Trainable"
# 2. Compare the performance between two (One is about 90% accuracy and the other is about 98%)
# 3. See feature outputs (avegarge all outputs) for all training samples.
# 
# Additional
# 
# Invesigate the error sample employing GradCam (ref : https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/) 

# #### Step 1. Train DenseNet + two fullyconnected Layers

# Includes Packages

# In[ ]:


import os
import gc
import re

import cv2
import math
import numpy as np
import scipy as sp
import pandas as pd

import tensorflow as tf

from kaggle_datasets import KaggleDatasets
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Conv2DTranspose, concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam


import warnings
import os 
import pandas as pd
import plotly.graph_objs as go
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


# In[ ]:


data_dir = '../input/plant-pathology-2020-fgvc7'
img_path = os.path.join(data_dir,'images')
IMG_DIM = 256

os.listdir(data_dir)


# load data

# In[ ]:


train_total=pd.read_csv(os.path.join(data_dir,"train.csv"))
test=pd.read_csv(os.path.join(data_dir,"test.csv"))
train_total['image_id']=train_total['image_id']+'.jpg'
test['image_id']=test['image_id']+'.jpg'
train_total['label'] = train_total.iloc[:,1:5].idxmax(1)


# In[ ]:


train_total.head()


# Split the Dataset into Train and Validation set using the option (stratify) for have the same percentage smaple by category

# In[ ]:


train, val = train_test_split(train_total, test_size = 0.15,stratify = train_total.label)
print(train.label.value_counts()/train_total.label.value_counts()) # TO check the percentage by category


# Create Keras Imagegenerator with augmentation for trainset / No augmentation for Validation

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator( horizontal_flip=True,
    vertical_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    fill_mode='nearest',
    shear_range=0.1,
    rescale=1/255,
    brightness_range=[0.5, 1.5])

val_datagen = ImageDataGenerator( 
    rescale=1/255)


# Define Generator

# In[ ]:


train_generator=train_datagen.flow_from_dataframe(train,directory=img_path,
                                                      target_size=(IMG_DIM,IMG_DIM),
                                                      x_col="image_id",
                                                      y_col=['healthy','multiple_diseases','rust','scab'],
                                                      class_mode='raw',
                                                      shuffle=False,
                                                       subset='training',
                                                      batch_size=16)
val_generator=val_datagen.flow_from_dataframe(val,directory=img_path,
                                                      target_size=(IMG_DIM,IMG_DIM),
                                                      x_col="image_id",
                                                      y_col=['healthy','multiple_diseases','rust','scab'],
                                                      class_mode='raw',
                                                      shuffle=False,
                                                      batch_size=16,
                                                  )

test_generator=val_datagen.flow_from_dataframe(test,directory='/kaggle/input/plant-pathology-2020-fgvc7/images/',
                                                      target_size=(IMG_DIM,IMG_DIM),
                                                      x_col="image_id",
                                                      y_col=None,
                                                      class_mode=None,
                                                      shuffle=False,
                                                      batch_size=16)


# Change to tf.data format (performance improve?)

# In[ ]:


ds = tf.data.Dataset.from_generator(lambda: train_generator,
                     output_types=(tf.float32,tf.float32),
                     output_shapes=([None, IMG_DIM, IMG_DIM, 3],[None, 4])
                     )
val_ds = tf.data.Dataset.from_generator(lambda: val_generator ,
                     output_types=(tf.float32,tf.float32),
                     output_shapes=([None, IMG_DIM, IMG_DIM, 3],[None, 4])
                     )


# In[ ]:


len(train_generator)


# In[ ]:


base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(IMG_DIM,IMG_DIM,3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
predictions = Dense(4, activation="softmax")(x)

model_updated = Model(inputs=base_model.input, outputs=predictions)
model_updated.compile(optimizer='adam',
                  loss = 'categorical_crossentropy',
                  metrics=['accuracy'])


# In[ ]:


# from tensorflow.keras.callbacks import ReduceLROnPlateau

# history = model_updated.fit(ds,                                    
#                                   steps_per_epoch=len(train_generator),
#                                   epochs=30,validation_data=val_ds,validation_steps=len(val_generator)
#                                   ,verbose=1,callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.3,patience=3, min_lr=0.000001)])


# Check loss and accuracy bu epochs

# In[ ]:


# loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross Entropy')
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.show()


# In[ ]:


# model_updated.save('/content/drive/My Drive/Colab Notebooks/Plant_256/DenseNet pretrained.h5')
# model_updated.save('/content/drive/My Drive/Colab Notebooks/Plant_256/DenseNet pretrained.h5')


# Two models were trained in Colab
# 
# Now load two models

# #### Step 2 Compare the performance between two (One is about 92% accuracy and the other is about 98%)

# In[ ]:


from tensorflow.keras.models import load_model
model_updated = load_model('/kaggle/input/trainedmodel/DenseNet/DenseNet retrained.h5')
model_pretrained = load_model('/kaggle/input/trainedmodel/DenseNet/DenseNet pretrained.h5')


# In[ ]:


model_updated.summary()


# In[ ]:


model_pretrained.summary()


# Prepare the dataset for total Training Dataset provided from Kaggle

# In[ ]:


train_total_generator=val_datagen.flow_from_dataframe(train_total,
                                                        directory= img_path,
                                                      target_size=(IMG_DIM,IMG_DIM),
                                                      x_col="image_id",
                                                      y_col=['healthy','multiple_diseases','rust','scab'],
                                                      class_mode='raw',
                                                      shuffle=False,
                                                      batch_size=32)

ds_total = tf.data.Dataset.from_generator(lambda: train_total_generator,
                     output_types=(tf.float32,tf.float32),
                     output_shapes=([None, IMG_DIM, IMG_DIM, 3],[None, 4])
                     )


# Evaluate Two models (92% vs 98%)

# In[ ]:


train_total_generator.reset()
model_pretrained.evaluate(ds_total,steps=len(train_total_generator),verbose=1)


# In[ ]:


train_total_generator.reset()
model_updated.evaluate(ds_total,steps=len(train_total_generator),verbose=1)


# Now, we are trying to see the level of difference (?) of feature output after average pooling layers (1024 outputs)of each model 

# #### Step 3. See feature outputs (avegarge all outputs) for all training samples.

# In[ ]:


# Redefine model too get 1024 feature output
model_updated_feature = Model(inputs=model_updated.input,
                              outputs=model_updated.get_layer("global_average_pooling2d_2").output)

model_pretrained_feature = Model(inputs=model_pretrained.input,
                                 outputs=model_pretrained.get_layer("global_average_pooling2d_3").output)

train_total_generator.reset()
preds_updated = model_updated_feature.predict(ds_total,steps=len(train_total_generator),verbose=1)
train_total_generator.reset()
preds_pretrained = model_pretrained_feature.predict(ds_total,steps=len(train_total_generator),verbose=1)


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le.fit(train_total['label'])
preds_label = le.transform(train_total['label'])


# Calculate the average 1024 feature output 

# In[ ]:


preds_updated_avg = [preds_updated[np.where(preds_label == x)].mean(axis = 0) for x in range(4)]
preds_pretrained_avg = [preds_pretrained[np.where(preds_label == x)].mean(axis = 0) for x in range(4)]


# As you can see the plot in below, 1024 feature output of retrined model (set all layers "Trainable") provides more information as everyone can expect.

# In[ ]:


plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
for x in range(4):
    plt.plot(preds_pretrained_avg[x], label = le.classes_[x])
    plt.legend(loc='upper left')
plt.subplot(1,2,2)
for x in range(4):
    plt.plot(preds_updated_avg[x], label = le.classes_[x])
    plt.legend(loc='upper left')


# #### Additional Step. Compare the ground truth labels and the predicted labels and investiagte the source of error by GradCam

# In[ ]:


train_total_generator.reset()
preds = model_updated.predict(ds_total,steps=len(train_total_generator),verbose=1)


# In[ ]:


preds_df = pd.DataFrame(preds)
preds_df = preds_df.rename(columns={0:'healthy',1:'multiple_diseases',2:'rust',3:'scab'})

preds_df['p_label'] = preds_df.iloc[:,:4].idxmax(1)

result = pd.concat([train_total, preds_df], axis=1, sort=False)

wrong_result = result[result.label != result.p_label]

wrong_result.image_id.count()


# Load image files of error sample

# In[ ]:


def load_image(image_id):
    image = load_img(os.path.join(data_dir,'images',image_id), target_size=(IMG_DIM,IMG_DIM ))
    return img_to_array(image)

id_list = wrong_result['image_id'].tolist()
suspisous_images = []


for ids in id_list:
    suspisous_images.append(load_image(ids))

suspisous_images = np.array(suspisous_images,dtype='float32')


# In[ ]:


suspisous_images = suspisous_images/256.
p_Label = wrong_result['p_label'].to_list()
Label = wrong_result['label'].to_list()


# In[ ]:


import sys
sys.path.append('/kaggle/input/gradcam')


# In[ ]:


from gradcam import GradCAM

heatmaps = []

for orig in suspisous_images:
    image = np.expand_dims(orig, axis=0)
    preds = model_updated.predict(image)
    i = np.argmax(preds)
    cam = GradCAM(model_updated, i)
    heatmap = cam.compute_heatmap(image)
    heatmap = cv2.resize(heatmap, (IMG_DIM, IMG_DIM))
    heatmaps.append(heatmap)

heatmaps = np.array(heatmaps,dtype='float32')


# In[ ]:


plt.figure(figsize=(15,400))
for i in range(wrong_result.image_id.count()):
    ax = plt.subplot(40,2,2*i+1)

    ax.text(0.5, 1, 'GT : {} '.format(Label[i]),
        verticalalignment='top', horizontalalignment='center',
        transform=ax.transAxes,
        color='White', fontsize=15)
    ax.set_title(id_list[i])

    plt.imshow(suspisous_images[i])
        
    ax = plt.subplot(40,2,2*i+2)
    ax.text(0.5, 1, ' vs PRED : {}'.format(p_Label[i]),
    verticalalignment='top', horizontalalignment='center',
    transform=ax.transAxes,
    color='White', fontsize=15)
    ax.set_title(id_list[i])
    
    plt.imshow(suspisous_images[i])
    plt.imshow(heatmaps[i],alpha=0.5)


# In[ ]:


wrong_result


# Observation:
# 
# 1. Transfer Learning approach would be a good starting point. By setting all layers as "Trainnable" would improve the model accuracy as 1024 feature output provide the more information
# 2. Trained model with ImageNet which comprises general total population of images for image identification should be retrained for this type of pupose with specific populations
# 3. Not sure why around 500 channels from 1024 is not activley responsive (may be more training epoch requried?)
# 4. Most errors is coming from "multiple diseases" Segemnt ==> What is exact definition of "multiple diseases". Is it both "rust" and "scab"?

# In[ ]:




