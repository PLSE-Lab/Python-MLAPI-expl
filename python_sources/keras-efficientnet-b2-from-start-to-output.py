#!/usr/bin/env python
# coding: utf-8

# This competition provides an exciting and challenging task of doing multi-label classification on a dataset with well over half a million images. There are multiple very nice notebooks which perform only 2 or 3 epochs with all the training data. In this notebook I will try out and see what the effect is of using more epochs but less steps per epoch. By averaging the predictions made during the last few epochs we should be able to achieve a nice LB score. This also should provide some alternative ways to experiment for the Kagglers that don't have the adequate computing resources available and are dependent on Kaggle Kernels.
# 
# As model I will be using the EfficientNet B2 model. It should be able to provide highly accurate predictions while still being able to run within the kernel limits. With 9 hours max time for a GPU kernel you have to make some trade-offs ;-)
# 
# I hope this kernel will be usefull and may'be will provide you with some new and alternative ideas to try out. If you like it..then please upvote it ;-)
# Any feedback or remarks are appreciated.
# 
# Lets start by importing all the necessary modules.
# 
# Note!! This kernel is now updated for Stage2 Training and Test data..altough with less epochs because of the increase in train and test data.

# In[ ]:


import numpy as np
import pandas as pd
import pydicom
import os
import collections
import sys
import glob
import random
import cv2
import tensorflow as tf
import multiprocessing

from math import ceil, floor
from copy import deepcopy
from tqdm import tqdm_notebook as tqdm
from imgaug import augmenters as iaa

import keras
import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model, load_model
from keras.utils import Sequence
from keras.losses import binary_crossentropy
from keras.optimizers import Adam


# Install and import the efficientnet and iterative-stratification packages from the internet. The iterative-stratification package provides a very nice implementation of multi-label stratification. I've used it in a few competitions now with good results. There are offcourse more packages that provide implementations for it.

# In[ ]:


# Install Modules from internet
get_ipython().system('pip install efficientnet')
get_ipython().system('pip install iterative-stratification')


# In[ ]:


# Import Custom Modules
import efficientnet.keras as efn 
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# Next we will set the random_state, some constants and folders that will be used later on. I've specified a rather small test size as I want to maximize the training time available and minimize the time used for validation. I'am not using methods like early stopping...when the kernel time limit is approaching we could still increase the results on the LB if we were allowed to continue.

# In[ ]:


# Seed
SEED = 12345
np.random.seed(SEED)
tf.set_random_seed(SEED)

# Constants
TEST_SIZE = 0.02
HEIGHT = 256
WIDTH = 256
CHANNELS = 3
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 64
SHAPE = (HEIGHT, WIDTH, CHANNELS)

# Folders
DATA_DIR = '/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/'
TEST_IMAGES_DIR = DATA_DIR + 'stage_2_test/'
TRAIN_IMAGES_DIR = DATA_DIR + 'stage_2_train/'


# Next the code for the DICOM windowing and the Data Generators. After seeing the effect of different versions of windowing as presented in this very nice [kernel](https://www.kaggle.com/akensert/inceptionv3-prev-resnet50-keras-baseline-model) I decided to also update my kernel with it. Lets see what the effect will be.

# In[ ]:


def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000

def window_image(dcm, window_center, window_width):    
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    
    # Resize
    img = cv2.resize(img, SHAPE[:2], interpolation = cv2.INTER_LINEAR)
   
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img

def bsb_window(dcm):
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    soft_img = window_image(dcm, 40, 380)
    
    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)
    return bsb_img

def _read(path, SHAPE):
    dcm = pydicom.dcmread(path)
    try:
        img = bsb_window(dcm)
    except:
        img = np.zeros(SHAPE)
    return img


# I'll specify some light image augmentation. Some horizontal and vertical flipping and some cropping. I haven't yet tried out more augmentation but will do so in future versions of the kernel. Also the code for Data Generators for train and test data.

# In[ ]:


# Image Augmentation
sometimes = lambda aug: iaa.Sometimes(0.25, aug)
augmentation = iaa.Sequential([ iaa.Fliplr(0.25),
                                iaa.Flipud(0.10),
                                sometimes(iaa.Crop(px=(0, 25), keep_size = True, sample_independently = False))   
                            ], random_order = True)       
        
# Generators
class TrainDataGenerator(keras.utils.Sequence):
    def __init__(self, dataset, labels, batch_size = 16, img_size = SHAPE, img_dir = TRAIN_IMAGES_DIR, augment = False, *args, **kwargs):
        self.dataset = dataset
        self.ids = dataset.index
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X, Y = self.__data_generation(indices)
        return X, Y

    def augmentor(self, image):
        augment_img = augmentation        
        image_aug = augment_img.augment_image(image)
        return image_aug

    def on_epoch_end(self):
        self.indices = np.arange(len(self.ids))
        np.random.shuffle(self.indices)

    def __data_generation(self, indices):
        X = np.empty((self.batch_size, *self.img_size))
        Y = np.empty((self.batch_size, 6), dtype=np.float32)
        
        for i, index in enumerate(indices):
            ID = self.ids[index]
            image = _read(self.img_dir+ID+".dcm", self.img_size)
            if self.augment:
                X[i,] = self.augmentor(image)
            else:
                X[i,] = image
            Y[i,] = self.labels.iloc[index].values        
        return X, Y
    
class TestDataGenerator(keras.utils.Sequence):
    def __init__(self, dataset, labels, batch_size = 16, img_size = SHAPE, img_dir = TEST_IMAGES_DIR, *args, **kwargs):
        self.dataset = dataset
        self.ids = dataset.index
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X = self.__data_generation(indices)
        return X

    def on_epoch_end(self):
        self.indices = np.arange(len(self.ids))
    
    def __data_generation(self, indices):
        X = np.empty((self.batch_size, *self.img_size))
        
        for i, index in enumerate(indices):
            ID = self.ids[index]
            image = _read(self.img_dir+ID+".dcm", self.img_size)
            X[i,] = image              
        return X


# Import the training and test datasets.

# In[ ]:


def read_testset(filename = DATA_DIR + "stage_2_sample_submission.csv"):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    return df

def read_trainset(filename = DATA_DIR + "stage_2_train.csv"):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    duplicates_to_remove = [56346, 56347, 56348, 56349,
                            56350, 56351, 1171830, 1171831,
                            1171832, 1171833, 1171834, 1171835,
                            3705312, 3705313, 3705314, 3705315,
                            3705316, 3705317, 3842478, 3842479,
                            3842480, 3842481, 3842482, 3842483 ]
    df = df.drop(index = duplicates_to_remove)
    df = df.reset_index(drop = True)    
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    return df

# Read Train and Test Datasets
test_df = read_testset()
train_df = read_trainset()


# The training data contains some class inbalance. Multiple kernels explored the use of undersampling..so let's try the opposite and oversample the minority class 'epidural' one additional time.

# In[ ]:


# Oversampling
epidural_df = train_df[train_df.Label['epidural'] == 1]
train_oversample_df = pd.concat([train_df, epidural_df])
train_df = train_oversample_df

# Summary
print('Train Shape: {}'.format(train_df.shape))
print('Test Shape: {}'.format(test_df.shape))


# Some methods for predictions on the test data, a callback method and a method to create the EfficientNet B2 model. For the EfficientNet we use the pretrained imagenet weights. Also a Dropout layer is added with a small value to prevent some overfitting. 

# In[ ]:


def predictions(test_df, model):    
    test_preds = model.predict_generator(TestDataGenerator(test_df, None, 8, SHAPE, TEST_IMAGES_DIR), verbose = 1)
    return test_preds[:test_df.iloc[range(test_df.shape[0])].shape[0]]

def ModelCheckpointFull(model_name):
    return ModelCheckpoint(model_name, 
                            monitor = 'val_loss', 
                            verbose = 1, 
                            save_best_only = False, 
                            save_weights_only = True, 
                            mode = 'min', 
                            period = 1)

# Create Model
def create_model():
    K.clear_session()
    
    base_model =  efn.EfficientNetB2(weights = 'imagenet', include_top = False, pooling = 'avg', input_shape = SHAPE)
    x = base_model.output
    x = Dropout(0.15)(x)
    y_pred = Dense(6, activation = 'sigmoid')(x)

    return Model(inputs = base_model.input, outputs = y_pred)


# Next we setup the multi label stratification. I've specified multiple splits but only using the first one for train data and validation data. Optionally you can also loop through the different splits and use a different train and validation set for each epoch. 

# In[ ]:


# Submission Placeholder
submission_predictions = []

# Multi Label Stratified Split stuff...
msss = MultilabelStratifiedShuffleSplit(n_splits = 10, test_size = TEST_SIZE, random_state = SEED)
X = train_df.index
Y = train_df.Label.values

# Get train and test index
msss_splits = next(msss.split(X, Y))
train_idx = msss_splits[0]
valid_idx = msss_splits[1]


# Now we can train the model for a number of epochs. All epochs we train the full model but each time on only 1/6 of the train data. With each epoch only a subset of the train data will allow us to make more epochs and allows todo averaging over more then just 1 or 2 epochs (compared to using all data every epoch).
# 
# Note that I recreate the data generators and model on each epoch. This is only necessary when using the different Multi-label stratified splits since the data generators will get a totally different set of data on each epoch then. I left it in so that you can try it out.
# 
# Starting with the 6th epoch a prediction for the test set is made on each epoch. In total predictions from the last 6 epochs will be averaged this way for the final submission.

# In[ ]:


# Loop through Folds of Multi Label Stratified Split
#for epoch, msss_splits in zip(range(0, 9), msss.split(X, Y)): 
#    # Get train and test index
#    train_idx = msss_splits[0]
#    valid_idx = msss_splits[1]
for epoch in range(0, 8):
    print('=========== EPOCH {}'.format(epoch))

    # Shuffle Train data
    np.random.shuffle(train_idx)
    print(train_idx[:5])    
    print(valid_idx[:5])

    # Create Data Generators for Train and Valid
    data_generator_train = TrainDataGenerator(train_df.iloc[train_idx], 
                                                train_df.iloc[train_idx], 
                                                TRAIN_BATCH_SIZE, 
                                                SHAPE,
                                                augment = True)
    data_generator_val = TrainDataGenerator(train_df.iloc[valid_idx], 
                                            train_df.iloc[valid_idx], 
                                            VALID_BATCH_SIZE, 
                                            SHAPE,
                                            augment = False)

    # Create Model
    model = create_model()
    
    # Full Training Model
    for base_layer in model.layers[:-1]:
        base_layer.trainable = True
    TRAIN_STEPS = int(len(data_generator_train) / 6)
    LR = 0.000125

    if epoch != 0:
        # Load Model Weights
        model.load_weights('model.h5')    

    model.compile(optimizer = Adam(learning_rate = LR), 
                  loss = 'binary_crossentropy',
                  metrics = ['acc', tf.keras.metrics.AUC()])
    
    # Train Model
    model.fit_generator(generator = data_generator_train,
                        validation_data = data_generator_val,
                        steps_per_epoch = TRAIN_STEPS,
                        epochs = 1,
                        callbacks = [ModelCheckpointFull('model.h5')],
                        verbose = 1)
    
    # Starting with the 4th epoch we create predictions for the test set on each epoch
    if epoch >= 3:
        preds = predictions(test_df, model)
        preds[:100]
        submission_predictions.append(preds)
        
model.save('Final.h5')


# And finally we create the submission file by averaging all submission_predictions.

# In[ ]:


test_df.iloc[:, :] = np.average(submission_predictions, axis = 0, weights = [2**i for i in range(len(submission_predictions))])
test_df = test_df.stack().reset_index()
test_df.insert(loc = 0, column = 'ID', value = test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])
test_df = test_df.drop(["Image", "Diagnosis"], axis=1)
test_df.to_csv('submission.csv', index = False)
print(test_df.head(12))


# In[ ]:




