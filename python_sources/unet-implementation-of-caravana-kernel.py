#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob # For pathname matching
from skimage.transform import resize
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model 
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten,concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import cv2

from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.misc import imresize

from time import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from matplotlib.pyplot import rc
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 12}
rc('font', **font)  # pass in the font dict as kwargs
K.set_image_dim_ordering('th')

import os
from os.path import basename
print(os.listdir("../input"))
print(os.listdir("../"))
# Any results you write to the current directory are saved as output.


# In[ ]:


input_folder = '../input'

train= glob('/'.join([input_folder,'train/*.jpg']))
train_masks= glob('/'.join([input_folder,'train_masks/*.gif']))
test= glob('/'.join([input_folder,'test/*.jpg']))
print('Number of training images: ', len(train), 'Number of corresponding masks: ', len(train_masks), 'Number of test images: ', len(test))


# In[ ]:


tt_ratio = 0.8
img_rows, img_cols = 1024,1024
batch_size = 8
def dice_coef(y_true, y_pred, smooth=0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection=K.sum(y_true_f * y_pred_f)
    return(2. * intersection + smooth) / ((K.sum(y_true_f) + K.sum(y_pred_f)) + smooth)


# In[ ]:


#split the training set into train and validation samples
train_images, validation_images = train_test_split(train, train_size=tt_ratio, test_size=1-tt_ratio)
print('Size of the training sample=', len(train_images), 'and size of the validation sample=', len(validation_images), ' images')


# In[ ]:


#utility function to convert greyscale inages to rgb
def grey2rgb(img):
    new_img = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img.append(list(img[i][j])*3)
    new_img = np.array(new_img).reshape(img.shape[0], img.shape[1], 3)
    return new_img

#generator that will be used to read data from the directory
def data_generator(data_dir, masks, images, dims, batch_size=batch_size):
    while True:
        ix=np.random.choice(np.arange(len(images)), batch_size)
        imgs = []
        labels = []
        for i in ix:
            # images
            original_img = cv2.imread(images[i])
            resized_img = imresize(original_img, dims + [3]) 
            array_img = resized_img/255
            array_img = array_img.swapaxes(0, 2)
            imgs.append(array_img)
            #imgs is a numpy array with dim: (batch size X 128 X 128 3)
            #print('shape of imgs ', array_img.shape)
            # masks
            try:
                mask_filename = basename(images[i])
                file_name = os.path.splitext(mask_filename)[0]
                correct_mask = '/'.join([input_folder,'train_masks',file_name+'_mask.gif'])
                original_mask = Image.open(correct_mask).convert('L')
                data = np.asarray(original_mask, dtype="int32")
                resized_mask = imresize(original_mask, dims+[3])
                array_mask = resized_mask / 255
                labels.append(array_mask)
            except Exception as e:
                labels=None
            
        imgs = np.array(imgs)
        labels = np.array(labels)
        try:
            relabel = labels.reshape(-1, dims[0], dims[1], 1)
            relabel = relabel.swapaxes(1, 3)
        except Exception as e:
            relabel=labels
        yield imgs, relabel


# In[ ]:


train_gen = data_generator('train/', train_masks, train_images, dims=[img_rows, img_cols])
img, msk = next(train_gen)
train_img = img[0].swapaxes(0,2)
train_msk = msk.swapaxes(1,3)

fig, ax = plt.subplots(1,2, figsize=(16, 16))
ax = ax.ravel()
ax[0].imshow(train_img, cmap='gray') 
ax[0].set_title('Training Image')
ax[1].imshow(grey2rgb(train_msk[0]), cmap='gray')
ax[1].set_title('Training Image mask')


# In[ ]:


# create an instance of a validation generator:
validation_gen = data_generator('train/', train_masks, validation_images, dims=[img_rows, img_cols]) 


# In[ ]:


def unet(input_size = (3,img_rows,img_cols)):
    input_ = Input(input_size)
    conv0 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_)
    conv0 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0)
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
    
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool0)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv4,up6], axis = 1)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 1)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 1)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 1)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    up10 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))

    conv10 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up10)
    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)

    model = Model(input = input_, outputs = conv11)
    
    model.compile(optimizer= Adam(lr=0.0005), loss='binary_crossentropy', metrics=[dice_coef])
    
    return model


# In[ ]:


# Build and compile the model
model = unet()
model.summary()

model.load_weights('saved_model.h5')


# In[ ]:


# Fit the model to the training set and compute dice coefficient at each validation set
model_save = ModelCheckpoint('best_model.hdf5', save_best_only=True, monitor='val_loss', mode='min')

model_run = model.fit_generator(train_gen, steps_per_epoch=50, epochs=60, validation_data=validation_gen, validation_steps=50, callbacks=[model_save])

model.save("saved_model.h5")


# In[ ]:


pd.DataFrame(model_run.history)[['dice_coef','val_dice_coef']].plot()


# In[ ]:


img, msk = next(validation_gen)
predicted_mask = model.predict(img).swapaxes(1,3)
validation_image = img[0].swapaxes(0,2)

fig, ax = plt.subplots(1,2, figsize=(16, 16))
ax = ax.ravel()
ax[0].imshow(validation_image, cmap='gray') 
ax[0].set_title('Validation Image')
ax[1].imshow(grey2rgb(predicted_mask[0]), cmap='gray')
ax[1].set_title('Validation Image mask')


# In[ ]:


test_set = data_generator('test/', train_masks, test, dims=[img_rows, img_cols]) 
img_tst, msk_tst = next(test_set)
predicted_mask_tst = model.predict(img_tst)
predicted_mask_tst = predicted_mask_tst.swapaxes(1,3)
test_mask = grey2rgb(predicted_mask_tst[0])

test_image = img_tst[0].swapaxes(0,2)

fig, ax = plt.subplots(1,2, figsize=(16, 16))
ax = ax.ravel()
ax[0].imshow(test_image, cmap='gray') 
ax[0].set_title('Test Image')
ax[1].imshow(test_mask, cmap='gray')
ax[1].set_title('Test Image mask')


# In[ ]:




