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
from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf
from keras import backend as K

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
        train_image.append(item)
    else:
        train_mask.append(item)
        
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

# In[ ]:


## Storing data
X = []
y = []
for image, mask in zip(train_image, train_mask):
    X.append(np.array(Image.open(path+image)))
    y.append(np.array(Image.open(path+mask)))


# If you load images by using cv2.imread(filepath), it gives you image as data type "np.darray"<br/>

# In[ ]:


X = np.array(X)
y = np.array(y)

print("X_shape : ", X.shape)
print("y_shape : ", y.shape)


# ## How to deal with train_masks.csv ?

# In[ ]:


mask_df = pd.read_csv("../input/train_masks.csv")
mask_df.head()


# **How to deal with `pixels` column ?**<br/>
# Let me try to convert the first `pixels` column to the mask image.<br/>
# Actually, this work could be not necessary, since we are provided mask_image. But other competition that I want to join provide only run length encoded data, so I do this to practice. 

# In[ ]:


width = 580
height = 420

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

# In[ ]:


(mask1 != y[0]).sum()


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

# In[ ]:


print("The number of train data : ", X.shape[0])


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

# ## Let's define U-net and train our model by using 100 data
# Since the whole data size is quite big, it may lead to over-memory if we load whole data on X and y as we did earier. <br/>
# So our strategy is to use `data generator` that allow us to load a few of data and to use them to train our model.<br/>
# Before that, we first use only 100 data to check our model works well

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


# Randomly choose the indices of data used to train our model.
indices = np.random.choice(range(len(train_image)), replace = False ,size = 100)
train_image_sample = np.array(train_image)[indices]
train_mask_sample = np.array(train_mask)[indices]


# In[ ]:


# Build the dataset.
IMG_HEIGHT = 96
IMG_WIDTH = 96

X = np.empty(shape = (len(indices), IMG_HEIGHT, IMG_WIDTH), dtype = 'float32')
y = np.empty(shape = (len(indices), IMG_HEIGHT, IMG_WIDTH), dtype = 'float32')

for i, (image_path, mask_path) in enumerate(zip(train_image_sample, train_mask_sample)):
    image = cv2.imread("../input/train/" + image_path, 0)
    mask = cv2.imread("../input/train/" + mask_path, 0)
    
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
    
    X[i] = image
    y[i] = mask

X = X[:,:,:,np.newaxis] / 255
y = y[:,:,:,np.newaxis] / 255
print("X shape : ", X.shape)
print("y shape : ", y.shape)


# Now we define the dice loss and metrics. I refered to this [site](https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py).

# In[ ]:


smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# Let's build U-net model. I also refered to this [site](https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py). 

# In[ ]:


inputs = Input((IMG_HEIGHT, IMG_WIDTH, 1))

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

model = Model(inputs=[inputs], outputs=[conv10])
model.compile(optimizer=Adam(lr = 1e-5), loss=dice_coef_loss, metrics=[dice_coef])


# In[ ]:


results = model.fit(X, y, validation_split=0.1, batch_size=4, epochs=20)


# It worked !! This was my first semantic segmentation models :)<br/>
# Let's keep going on !

# ## Define image_generator

# In order to define data generator, I refer this [site](https://towardsdatascience.com/a-keras-pipeline-for-image-segmentation-part-1-6515a421157d)  

# In[ ]:


def Generator(X_list, y_list, batch_size = 16):
    c = 0

    while(True):
        X = np.empty((batch_size, IMG_HEIGHT, IMG_WIDTH), dtype = 'float32')
        y = np.empty((batch_size, IMG_HEIGHT, IMG_WIDTH), dtype = 'float32')
        
        for i in range(c,c+batch_size):
            image = cv2.imread("../input/train/" + X_list[i], 0)
            mask = cv2.imread("../input/train/" + y_list[i], 0)
    
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)
    
            X[i - c] = image
            y[i - c] = mask
        
        X = X[:,:,:,np.newaxis] / 255
        y = y[:,:,:,np.newaxis] / 255
        
        c += batch_size
        if(c+batch_size >= len(X_list)):
            c = 0
        yield X, y    


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(train_image, train_mask, test_size = 0.3, random_state = 1)

epochs = 10
batch_size = 8
steps_per_epoch = int(len(X_train) / batch_size)
validation_steps = int(len(X_val) / batch_size)

train_gen = Generator(X_train, y_train, batch_size = batch_size)
val_gen = Generator(X_val, y_val, batch_size = batch_size)

model = Model(inputs=[inputs], outputs=[conv10])
model.compile(optimizer=Adam(lr = 1e-5), loss=dice_coef_loss, metrics=[dice_coef])


# In[ ]:


history = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs = epochs,
                             validation_data = val_gen, validation_steps = validation_steps)


# ## Let's predict the test data set.

# In[ ]:


sub = pd.read_csv("../input/sample_submission.csv")
test_list = os.listdir("../input/test")

print("The number of test data : ", len(test_list))

# Sort the test set in ascending order.
reg = re.compile("[0-9]+")

temp1 = list(map(lambda x: reg.match(x).group(), test_list)) 
temp1 = list(map(int, temp1))

test_list = [x for _,x in sorted(zip(temp1, test_list))]

test_list[:15]


# In[ ]:


X_test = np.empty((len(test_list), IMG_HEIGHT, IMG_WIDTH), dtype = 'float32')
for i, item in enumerate(test_list):
    image = cv2.imread("../input/test/" + item, 0)
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)
    X_test[i] = image
X_test = X_test[:,:,:,np.newaxis] / 255

y_pred = model.predict(X_test)


# I refered to the this [site](https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/submission.py) to submit my prediction

# In[ ]:


def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


# In[ ]:


rles = []
for i in range(X_test.shape[0]):
    img = y_pred[i, :, :, 0]
    img = img > 0.5
    img = resize(img, (420, 580), preserve_range=True)
    rle = run_length_enc(img)
    rles.append(rle)
    if i % 100 == 0:
            print('{}/{}'.format(i, X_test.shape[0]), end = "\r")


# In[ ]:


sub['pixels'] = rles
sub.to_csv("submission.csv", index = False)


# It gives me 0.44690 score.

# In[ ]:




