#!/usr/bin/env python
# coding: utf-8

# Hi, I am a semantic segmentation beginner.(I'm sorry for my poor English in advance)<br/>
# (I refered to many part of this [site](https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/submission.py))

# In[ ]:


import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split, KFold

import keras
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

K.set_image_data_format('channels_last')


# In[ ]:


print(os.listdir("../input"))


# ## Building the training dataset.
# Let's look at the train image list

# In[ ]:


path = "../input/train/"
file_list = os.listdir(path)
file_list[:20]


# **Sort the file list in ascending order and seperate it into images and masks**<br/>
# Each file has the form of either "subject_imageNum.tif" or "subject_imageNum_mask.tif", so we can extract `subject` and `imageNum` from each file name by using regular expression. `"[0-9]+"` means to find the first consecutive number.<br/>

# In[ ]:


reg = re.compile("[0-9]+")

temp1 = list(map(lambda x: reg.match(x).group(), file_list)) 
temp1 = list(map(int, temp1))

temp2 = list(map(lambda x: reg.match(x.split("_")[1]).group(), file_list))
temp2 = list(map(int, temp2))

file_list = [x for _,_,x in sorted(zip(temp1, temp2, file_list))]
file_list[:20]


# In[ ]:


train_image = []
train_mask = []
for idx, item in enumerate(file_list):
    if idx % 2 == 0:
        train_image.append("../input/train/"+item)
    else:
        train_mask.append("../input/train/"+item)
        
print(train_image[:10],"\n" ,train_mask[:10])


# In[ ]:


# Display the first image and mask of the first subject.
image1 = np.array(Image.open(path+"1_1.tif"))
image1_mask = np.array(Image.open(path+"1_1_mask.tif"))
image1_mask = np.ma.masked_where(image1_mask == 0, image1_mask)

fig, ax = plt.subplots(1,3,figsize = (16,12))
ax[0].imshow(image1, cmap = 'gray')

ax[1].imshow(image1_mask, cmap = 'gray')

ax[2].imshow(image1, cmap = 'gray', interpolation = 'none')
ax[2].imshow(image1_mask, cmap = 'jet', interpolation = 'none', alpha = 0.7)


# Now, I try to load all image files and store them variables X and y. Afther doing this, I recognize that it takes very much memory.<br/>
# Please let me know if there are several efficient ways to store image file

# ## How to deal with train_masks.csv ?

# In[ ]:


mask_df = pd.read_csv("../input/train_masks.csv")
mask_df.head()


# **How to deal with `pixels` column ?**<br/>
# Let me try to convert the first `pixels` column to the mask image.<br/>
# Actually, this work could be not necessary, since we are provided mask_image. But other competition that I want to join provide only run length encoded data, so I do this to practice. 

# In[ ]:


width = 512
height = 512

temp = mask_df["pixels"][0]
temp = temp.split(" ")


# In[ ]:


mask1 = np.zeros(height * width)
for i, num in enumerate(temp):
    if i % 2 == 0:
        run = int(num) -1             # very first pixel is 1, not 0
        length = int(temp[i+1])
        mask1[run:run+length] = 255 

#Since pixels are numbered from top to bottom, then left to right, we are careful to change the shape
mask1 = mask1.reshape((width, height))
mask1 = mask1.T 


# Let's check that I did well

# Let's modularize this work.

# In[ ]:


# RLE : run-length-encoding
def RLE_to_image(rle):
    '''
    rle : array in mask_df["pixels"]
    '''
    width, height = 580, 420
    
    if rle == 0:
        return np.zeros((height,width))
    
    else:
        rle = rle.split(" ")
        mask = np.zeros(width * height)
        for i, num in enumerate(rle):
            if i % 2 == 0:
                run = int(num) - 1
                length = int(rle[i+1])
                mask[run:run+length] = 255

        mask = mask.reshape((width, height))
        mask = mask.T 

        return mask


# ## Exploratory data analysis
# First of all, let's check how many train data we have.

# One can find the number of subjects in train data by `groupby` method on `mask_df`.

# In[ ]:


mask_df.head()
subject_df = mask_df[['subject', 'img']].groupby(by = 'subject').agg('count').reset_index()
subject_df.columns = ['subject', 'N_of_img']
subject_df.sample(10)


# In[ ]:


pd.value_counts(subject_df['N_of_img']).reset_index()


# There are total 47 subjects and almost almost all subjects have 120 images except for 5 subjects who have 119 images.<br/>
# I want to know whether test dataset has similar distribution or not. Let's check this by using the similar way when we listed the train data.

# In[ ]:


print(os.listdir("../input/test")[0:15])


# Each test image name is numbered in different way, so we cannot exploit subject information when we predict test data.

# In[ ]:


from keras.models import Model, Input, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[ ]:


smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred, smooth = 100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.square(y_true), axis = -1) + K.sum(K.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


# In[ ]:


def unet(input_size=(256,256,1)):
    
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])


# In[ ]:


def train_generator(data_frame, batch_size, train_path, aug_dict,
        image_color_mode="grayscale",
        mask_color_mode="grayscale",
        image_save_prefix="image",
        mask_save_prefix="mask",
        save_to_dir=None,
        target_size=(256,256),
        seed=1):
    '''
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        directory = train_path,
        x_col = "filename",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        directory = train_path,
        x_col = "mask",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator)
    
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img,mask)

def adjust_data(img,mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return (img, mask)


# In[ ]:


df = pd.DataFrame(data={"filename": train_image, 'mask' : train_mask})

kf = KFold(n_splits = 5, shuffle=False)


# In[ ]:


train_generator_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest')

histories = []
losses = []
accuracies = []
dicecoefs = []
ious = []

EPOCHS = 50
BATCH_SIZE = 16

for k, (train_index, test_index) in enumerate(kf.split(df)):
    train_data_frame = df.iloc[train_index]
    test_data_frame = df.iloc[test_index]
    
    train_gen = train_generator(train_data_frame, BATCH_SIZE,
                                None,
                                train_generator_args,
                                target_size=(height, width))

    test_gener = train_generator(test_data_frame, BATCH_SIZE,
                                None,
                                train_generator_args,
                                target_size=(height, width))

    model = unet(input_size=(height,width, 1))
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss,                       metrics=[iou, dice_coef, 'binary_accuracy'])
#    model.summary()

    model_checkpoint = ModelCheckpoint(str(k+1) + '_unet_lung_seg.hdf5', 
                                       monitor='loss', 
                                       verbose=1, 
                                       save_best_only=True)

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=len(train_data_frame) / BATCH_SIZE, 
                                  epochs=EPOCHS, 
                                  callbacks=[model_checkpoint],
                                  validation_data = test_gener,
                                  validation_steps=len(test_data_frame) / BATCH_SIZE)
    
    #test_gen = test_generator(test_files, target_size=(512,512))
    test_gen = train_generator(test_data_frame, BATCH_SIZE,
                                None,
                                train_generator_args,
                                target_size=(height, width))
    results = model.evaluate_generator(test_gen, steps=len(test_data_frame))
    results = dict(zip(model.metrics_names,results))
    
    histories.append(history)
    accuracies.append(results['binary_accuracy'])
    losses.append(results['loss'])
    dicecoefs.append(results['dice_coef'])
    ious.append(results['iou'])


# In[ ]:


for h, history in enumerate(histories):

    keys = history.history.keys()
    fig, axs = plt.subplots(1, 4, figsize = (25, 5))
    fig.suptitle('No. ' + str(h+1) + ' Fold Results', fontsize=30)

    for k, key in enumerate(list(keys)[len(keys)//2:]):
        training = history.history[key]
        validation = history.history['val_' + key]

        epoch_count = range(1, len(training) + 1)

        axs[k].plot(epoch_count, training, 'r--')
        axs[k].plot(epoch_count, validation, 'b-')
        axs[k].legend(['Training ' + key, 'Validation ' + key])


# In[ ]:


print('average accuracy : ', np.mean(np.array(accuracies)), '+-', np.std(np.array(accuracies)))
print('average loss : ', np.mean(np.array(losses)), '+-', np.std(np.array(losses)))
print('average iou : ', np.mean(np.array(ious)), '+-', np.std(np.array(ious)))
print('average dice_coe : ', np.mean(np.array(dice_cos)), '+-', np.std(np.array(dice_cos)))

