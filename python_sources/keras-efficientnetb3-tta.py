#!/usr/bin/env python
# coding: utf-8

# **I would like to share a notebook that i used to train a model which i used in ensembling which resulted in the final score of 0.810 on the public LB, also i would like to share the TTA i used which gave me an lb boost of about 0.02-0.03**

# Configurations :
# * MODEL - EfficientNetB3
# * IMAGE SIZE - 300x300
# * OPTIMIZER - Adam
# * LEARNING RATE - 5e-5
# * EPOCHS - 6
# * BATCH SIZE - 8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import h5py
import keras
import matplotlib.pyplot as plt
import cv2
from keras.optimizers import Adam
from tqdm import tqdm_notebook
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import random
from statistics import mode


# Defining paths to directories

# In[ ]:


model_path = '../input/size300-v1-old-data-aptos/old_data_effnet.h5'
train_csv = "../input/aptos2019-blindness-detection/train.csv"
test_csv = "../input/aptos2019-blindness-detection/test.csv"
train_dir = "../input/aptos2019-blindness-detection/train_images/"
test_dir = "../input/aptos2019-blindness-detection/test_images/"
size = 300,300 # input image size


# Loading Model : 

# In[ ]:


get_ipython().system("pip install -U '../input/install/efficientnet-0.0.3-py2.py3-none-any.whl'")
from efficientnet import EfficientNetB3

model = keras.models.load_model(model_path)

optimizer=Adam(lr = 5e-5)
loss = "binary_crossentropy"
model.compile(loss = loss, optimizer = optimizer, metrics = ["accuracy"])


# In[ ]:


def get_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    return image


# Below code converts the labels into multilabel, this code is taken from the kernel [APTOS 2019: DenseNet Keras Starter](https://www.kaggle.com/xhlulu/aptos-2019-densenet-keras-starter)

# In[ ]:


df = pd.read_csv(train_csv)

train_paths = [train_dir + str(x) + str(".png") for x in df["id_code"]]

labels = pd.get_dummies(df["diagnosis"]).values
y_train_multi = np.empty(labels.shape, dtype=labels.dtype)
y_train_multi[:, 4] = labels[:, 4]
for i in range(3, -1, -1):
    y_train_multi[:, i] = np.logical_or(labels[:, i], y_train_multi[:, i+1])
train_labels = y_train_multi


# In[ ]:


# loading images
images = np.empty((len(df),300,300,3), dtype = np.uint8)
for i, im in tqdm_notebook(enumerate(train_paths)):
    images[i,:,:,:] = get_image(im)


# In[ ]:


images, x_val, train_labels, y_val = train_test_split(images, train_labels, test_size = 0.15)


# #### TRAINING : 

# In[ ]:


train_aug = ImageDataGenerator(
        zoom_range=0.25,
        rotation_range = 360,
        vertical_flip=True,
        horizontal_flip=True)

train_generator = train_aug.flow(images, train_labels, batch_size = 8)

model.fit_generator(train_generator, epochs = 6, steps_per_epoch = len(train_generator), validation_data = (x_val, y_val))

#training process is same for the old data as well.


# In[ ]:


del train_generator, images, x_val
gc.collect


# PREDICTIONS AND TTA :

# In[ ]:


def get_predictions(test, model):
    predictions = model.predict(test) > 0.5
    predictions = predictions.astype(int).sum(axis=1) - 1
    return predictions

def rotate_image(image, degree):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2,height/2), degree, 1)
    image_transformed = cv2.warpAffine(image, rotation_matrix, (width, height))
    return image_transformed


# In[ ]:


# loading test images

test_df = pd.read_csv(test_csv)
test_paths = [test_dir + str(x) + str(".png") for x in test_df["id_code"]]

images = np.empty((len(test_df),300,300,3), dtype = np.uint8)
for i, im in tqdm_notebook(enumerate(test_paths)):
    images[i,:,:,:] = get_image(im)


# ### TTA

# In[ ]:


predictions1 = get_predictions(images, model)

# rotate image and then make predictions
temp_images = np.empty((len(images), 300,300,3), dtype = np.uint8)
for i in range(len(images)):
    temp_images[i,:,:,:] = rotate_image(images[i], random.randint(1,6))
predictions2 = get_predictions(temp_images, model)

# rotate image and make predictions
temp_images = np.empty((len(images), 300,300,3), dtype = np.uint8)
for i in range(len(images)):
    temp_images[i,:,:,:] = rotate_image(images[i], random.randint(7,12))

predictions3 = get_predictions(temp_images, model)


# In[ ]:


# stack all predictions for images and then find the mode of prediction.
final_predictions = []
for i in range(len(predictions1)):
    temp = [predictions1[i], predictions2[i], predictions3[i]]
    curr_mode = None
    if len(set(temp)) == 3:
        curr_mode = temp[0]
    else:
        curr_mode = mode(temp)
    final_predictions.append(curr_mode)


# In[ ]:


id_code = test_df["id_code"].values.tolist()
subfile = pd.DataFrame({"id_code":id_code, "diagnosis":final_predictions})
subfile.to_csv('submission.csv',index=False)

