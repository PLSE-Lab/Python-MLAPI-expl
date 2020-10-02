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

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Resources:
# 
# https://www.kaggle.com/xhlulu/aptos-2019-densenet-keras-starter
# 
# https://www.kaggle.com/wouterbulten/getting-started-with-the-panda-dataset
# 
# https://www.kaggle.com/yeayates21/densenet-keras-starter-fork-v2
# 
# https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta
# 
# https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
# 
# https://stackoverflow.com/questions/55328355/keras-flow-from-directory-read-only-from-selected-sub-directories
# 
# https://kylewbanks.com/blog/train-validation-split-with-imagedatagenerator-keras
# 
# https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720

# # Imports

# In[ ]:


import os
import gc
import json
import math
import cv2
import PIL
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import tensorflow as tf
from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing import image


# # Config Settings

# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# # Fixed Constants

# In[ ]:


# class directories
cover_dir = "../input/alaska2-image-steganalysis/Cover/"
JMiPOD_dir = "../input/alaska2-image-steganalysis/JMiPOD/"
JUNIWARD_dir = "../input/alaska2-image-steganalysis/JUNIWARD/"
UERD_dir = "../input/alaska2-image-steganalysis/UERD/"
# add to list
class_locs = [cover_dir, JMiPOD_dir, JUNIWARD_dir, UERD_dir]


# # Variable Constants

# In[ ]:


BATCH_SIZE = 8
TRAIN_VAL_SPLIT_RATIO = 0.50
EPOCHS = 4
# sample size for each class
cN = 2000


# # Training Data

# In[ ]:


def preprocess_image(image_path, desired_size=224):
    im = Image.open(image_path)
    im = im.resize((desired_size, desired_size))
    im = np.array(im) / 255
    return im


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# total samles \nN = cN * len(class_locs)\n# initialize y_train\ny_train = []\n# initalize x_train\nx_train = np.empty((N, 224, 224, 3), dtype=np.float16)\n# initialize class counter\nimClassNum = 0\n\n# run loop to grab training data\nfor imClass_dir in class_locs:\n    imClassNum += 1\n    print("running class", str(imClassNum), "...")\n    for i, filename in enumerate(os.listdir(imClass_dir)):\n        x_train[i, :, :, :] = preprocess_image(imClass_dir+filename)\n        print(str(round(i/75000,2))+"% of total images processed, "+str(i)+" images in total", end="\\r")\n        if i == cN:\n            print("")\n            print("finished with class " + str(imClassNum) + "..")\n            break\n    y_train.extend([imClassNum] * cN)\n\n# convert y_train to numpy    \ny_train = np.array(y_train)    \n\nprint("final training dataset shape..")\nprint("x_train shape: ", x_train.shape)\nprint("y_train len: ", len(y_train))\nprint(str(round(N/(75000*4),2))+"% of total images being used for training")')


# In[ ]:


# pre-processing the target (i.e. one-hot encoding the target)
y_train = pd.get_dummies(y_train).values


# In[ ]:


y_train.shape


# # Train Validation Split

# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, 
    test_size=TRAIN_VAL_SPLIT_RATIO, 
    random_state=2020
)


# # Image Gen

# In[ ]:


# define gen
img_gen = ImageDataGenerator(
        zoom_range=0.15,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )

# create generator
data_generator = img_gen.flow(x_train, y_train, batch_size=BATCH_SIZE, seed=2019)


# # Create Model

# In[ ]:


densenet = DenseNet121(
    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(224,224,3)
)


# In[ ]:


def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.80))
    model.add(layers.Dense(4, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00010509613402110064),
        metrics=['accuracy']
    )
    
    return model


# In[ ]:


model = build_model()
model.summary()


# # Train Model

# In[ ]:


history = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val)
)


# # Training Plots

# In[ ]:


history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['accuracy', 'val_accuracy']].plot()


# # Submission

# In[ ]:


sub = pd.read_csv("../input/alaska2-image-steganalysis/sample_submission.csv")
sub.head()


# In[ ]:


# test data directory
test_dir = "../input/alaska2-image-steganalysis/Test/"

# do similar data loading we did on train but on the test\holdout set
N = len([name for name in os.listdir(test_dir)])
x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)
for i, filename in enumerate(tqdm(os.listdir(test_dir))):
    x_test[i, :, :, :] = preprocess_image(test_dir+filename)

x_test.shape


# In[ ]:


y_test = model.predict(x_test)
y_test = y_test[:,0]
sub['Label'] = y_test


# In[ ]:


sub.to_csv("submission.csv", index=False)


# In[ ]:


sub.head()


# In[ ]:




