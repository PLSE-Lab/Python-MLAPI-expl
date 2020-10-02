#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import the dataset
import pandas as pd
sub = pd.read_csv("../input/understanding_cloud_organization/sample_submission.csv")
train = pd.read_csv("../input/understanding_cloud_organization/train.csv")


# In[ ]:


# Run-length decoder
def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    '''
    Decode rle encoded mask.
    
    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


# # Initial inspection of the dataset:

# In[ ]:


# Understanding the dataset
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import patches
# read the csv file using read_csv function of pandas
train.head()


# # Dataset size:

# In[ ]:


import os
path = '../input/understanding_cloud_organization'
os.listdir(path)
n_train = len(os.listdir(f'{path}/train_images'))
n_test = len(os.listdir(f'{path}/test_images'))
print(f'There are {n_train} images in train dataset')
print(f'There are {n_test} images in test dataset')


# # How many classes:

# In[ ]:


classes = pd.DataFrame(train['Image_Label'].str.split('_',1).tolist(),
                                   columns = ['img_name','class'])
print('There are ' + str(classes['class'].nunique()) + ' classes')


# # Data distribution:

# In[ ]:


# Number of images per class
for class_ in classes['class'].unique():
    nr_imgs = sum([off_class for off_class, no_data in zip(train['Image_Label'].str.endswith(str(class_)), train['EncodedPixels'].isna()) if off_class == True and no_data == False])
    print("Number of images of class " + str(class_) + ": " + str(nr_imgs))


# # Images of each class:

# In[ ]:


import numpy as np
# Images known to contain the classes
fish_imgs = ['0011165', '002be4f', '0031ae9', '00dec6a']
flower_imgs = ['0011165', '002be4f', '0031ae9', '00dec6a']
gravel_imgs = ['00a0954', '00b81e1', '00cedfa', '00dec6a']
sugar_imgs = ['00a0954', '00b81e1', '00cedfa', '00dec6a']

columns = 4
rows = 4
fig, ax = plt.subplots(rows, columns, figsize=(18, 13))
ax[0, 0].set_title('Fish', fontsize=20)
ax[0, 1].set_title('Flower', fontsize=20)
ax[0, 2].set_title('Gravel', fontsize=20)
ax[0, 3].set_title('Sugar', fontsize=20)
for i in range(len(fish_imgs)):
    fish_img = plt.imread(f"{path}/train_images/{fish_imgs[i]}.jpg")
    ax[i, 0].imshow(fish_img)
    image_label = f'{fish_imgs[i]}.jpg_Fish'
    mask_rle = train.loc[train['Image_Label'] == image_label, 'EncodedPixels'].values[0]
    mask = rle_decode(mask_rle)
    ax[i, 0].imshow(mask, alpha=0.5, cmap='gray')
    
    flower_img = plt.imread(f"{path}/train_images/{flower_imgs[i]}.jpg")
    ax[i, 1].imshow(flower_img)
    image_label = f'{flower_imgs[i]}.jpg_Flower'
    mask_rle = train.loc[train['Image_Label'] == image_label, 'EncodedPixels'].values[0]
    mask = rle_decode(mask_rle)
    ax[i, 1].imshow(mask, alpha=0.5, cmap='gray')
    
    gravel_img = plt.imread(f"{path}/train_images/{gravel_imgs[i]}.jpg")
    ax[i, 2].imshow(gravel_img)
    image_label = f'{gravel_imgs[i]}.jpg_Gravel'
    mask_rle = train.loc[train['Image_Label'] == image_label, 'EncodedPixels'].values[0]
    mask = rle_decode(mask_rle)
    ax[i, 2].imshow(mask, alpha=0.5, cmap='gray')
    
    sugar_img = plt.imread(f"{path}/train_images/{sugar_imgs[i]}.jpg")
    ax[i, 3].imshow(sugar_img)
    image_label = f'{sugar_imgs[i]}.jpg_Sugar'
    mask_rle = train.loc[train['Image_Label'] == image_label, 'EncodedPixels'].values[0]
    mask = rle_decode(mask_rle)
    ax[i, 3].imshow(mask, alpha=0.5, cmap='gray')
plt.show()


# # Based on "Satellite Clouds: U-Net with ResNet Encoder" by xhlulu

# In[ ]:


get_ipython().system('pip install segmentation-models --quiet')
get_ipython().system('pip install tensorflow')
import tensorflow as tf
print(tf.__version__)
import os
import json

import albumentations as albu
import cv2
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Cropping2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
from keras.optimizers import Adam, Nadam
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import segmentation_models as sm


# # Pre-process the dataset:

# In[ ]:


# Read training data
train_df = pd.read_csv('../input/understanding_cloud_organization/train.csv')
# Split Image_Label into Image(ImageId) and Label(ClassId)
train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
train_df['ClassId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])
# Determine if an Image_Label contains a mask or not
train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()

train_df.head()


# In[ ]:


# Determine number of masks pr. image
mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()
mask_count_df.sort_values('hasMask', ascending=False, inplace=True)


# In[ ]:


sub_df = pd.read_csv('../input/understanding_cloud_organization/sample_submission.csv')
sub_df['ImageId'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])
test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])


# In[ ]:


train_imgs = pd.DataFrame(train_df['ImageId'].unique(), columns=['ImageId'])


# # Utility Functions:

# In[ ]:


# Utility functions
def np_resize(img, input_shape):
    """
    Reshape a numpy array, which is input_shape=(height, width), 
    as opposed to input_shape=(width, height) for cv2
    """
    height, width = input_shape
    return cv2.resize(img, (width, height))
    
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(rle, input_shape):
    width, height = input_shape[:2]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return mask.reshape(height, width).T

def build_masks(rles, input_shape, reshape=None):
    depth = len(rles)
    if reshape is None:
        masks = np.zeros((*input_shape, depth))
    else:
        masks = np.zeros((*reshape, depth))
    
    for i, rle in enumerate(rles):
        if type(rle) is str:
            if reshape is None:
                masks[:, :, i] = rle2mask(rle, input_shape)
            else:
                mask = rle2mask(rle, input_shape)
                reshaped_mask = np_resize(mask, reshape)
                masks[:, :, i] = reshaped_mask
    
    return masks

def build_rles(masks, reshape=None):
    width, height, depth = masks.shape
    
    rles = []
    
    for i in range(depth):
        mask = masks[:, :, i]
        
        if reshape:
            mask = mask.astype(np.float32)
            mask = np_resize(mask, reshape).astype(np.int64)
        
        rle = mask2rle(mask)
        rles.append(rle)
        
    return rles


# # Loss Function:

# In[ ]:


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# # Data Generator:

# In[ ]:


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, df, target_df=None, mode='fit',
                 base_path="../input/understanding_cloud_organization/train_images",
                 batch_size=32, dim=(1400, 2100), n_channels=3, reshape=None,
                 augment=False, n_classes=4, random_state=2019, shuffle=True, normalize=False, mean=(0.25664523, 0.27471591, 0.32296003), std=(0.24326022, 0.23920952, 0.23920952)):
#(array([0.32296003]), array([0.27471591]), array([0.25664523]))
#(array([0.23920952]), array([0.23920952]), array([0.24326022]))
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
        self.base_path = base_path
        self.target_df = target_df
        self.list_IDs = list_IDs
        self.reshape = reshape
        self.n_channels = n_channels
        self.augment = augment
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = random_state
        self.normalize = normalize
        self.mean = mean
        self.std = std
        
        self.on_epoch_end()
        np.random.seed(self.random_state)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]
        
        X = self.__generate_X(list_IDs_batch)
        
        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch)
            
            if self.augment:
                X, y = self.__augment_batch(X, y)
                
            if self.normalize:
                X, y = self.__normalize_batch(X, y)
            
            return X, y
        
        elif self.mode == 'predict':
            #X = self.__normalize_batch_predict(X)
            return X

        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')
            
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)
    
    def __generate_X(self, list_IDs_batch):
        'Generates data containing batch_size samples'
        # Initialization
        if self.reshape is None:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
        else:
            X = np.empty((self.batch_size, *self.reshape, self.n_channels))
        
        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            img_path = f"{self.base_path}/{im_name}"
            img = self.__load_rgb(img_path)
            
            if self.reshape is not None:
                img = np_resize(img, self.reshape)
            
            # Store samples
            X[i,] = img

        return X
    
    def __generate_y(self, list_IDs_batch):
        if self.reshape is None:
            y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)
        else:
            y = np.empty((self.batch_size, *self.reshape, self.n_classes), dtype=int)
        
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            image_df = self.target_df[self.target_df['ImageId'] == im_name]
            
            rles = image_df['EncodedPixels'].values
            
            if self.reshape is not None:
                masks = build_masks(rles, input_shape=self.dim, reshape=self.reshape)
            else:
                masks = build_masks(rles, input_shape=self.dim)
            
            y[i, ] = masks

        return y

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)

        return img
    
    def __load_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.

        return img
    
    def __random_transform(self, img, masks):
        composition = albu.Compose([
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.ShiftScaleRotate(rotate_limit=45, shift_limit=0.15, scale_limit=0.15),
            #albu.RandomSizedCrop(min_max_height=(192,288), height=320, width=480, w2h_ratio=0.666, interpolation=1, always_apply=False, p=0.5),
            #albu.CoarseDropout(max_holes=4, max_height=32, max_width=48, min_holes=2, min_height=16, min_width=24, fill_value=0, always_apply=False, p=0.5),
            #albu.Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0, always_apply=False, p=1.0)
        ])
        
        composed = composition(image=img, mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']
        
        return aug_img, aug_masks
    
    def __normalize(self, img, masks):
        composition = albu.Compose([
            albu.Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0, always_apply=False, p=1.0)
        ])
        
        composed = composition(image=img, mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']
        
        return aug_img, aug_masks
    
    def __normalize_predict(self, img):
        composition = albu.Compose([
            albu.Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0, always_apply=False, p=1.0)
        ])
        
        composed = composition(image=img)
        aug_img = composed['image']
        
        return aug_img
    
    def __augment_batch(self, img_batch, masks_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i, ], masks_batch[i, ] = self.__random_transform(
                img_batch[i, ], masks_batch[i, ])
        
        return img_batch, masks_batch
    
    def __normalize_batch(self, img_batch, masks_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i, ], masks_batch[i, ] = self.__normalize(
                img_batch[i, ], masks_batch[i, ])

        return img_batch, masks_batch
    
    def __normalize_batch_predict(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i, ] = self.__normalize_predict(
                img_batch[i, ])

        return img_batch


# # Vanilla U-Net model:

# In[ ]:


def vanilla_unet(input_shape):
    """
    Unet vanilla
    """
    inputs = Input(input_shape)

    c1 = Conv2D(8, (3, 3), activation='elu', padding='same') (inputs)
    c1 = Conv2D(8, (3, 3), activation='elu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2), padding='same') (c1)

    c2 = Conv2D(16, (3, 3), activation='elu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='elu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2), padding='same') (c2)

    c3 = Conv2D(32, (3, 3), activation='elu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='elu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2), padding='same') (c3)

    c4 = Conv2D(64, (3, 3), activation='elu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='elu', padding='same') (c4)
    p4 = MaxPooling2D((2, 2), padding='same') (c4)

    c5 = Conv2D(64, (3, 3), activation='elu', padding='same') (p4)
    c5 = Conv2D(64, (3, 3), activation='elu', padding='same') (c5)
    p5 = MaxPooling2D((2, 2), padding='same') (c5)

    c55 = Conv2D(128, (3, 3), activation='elu', padding='same') (p5)
    c55 = Conv2D(128, (3, 3), activation='elu', padding='same') (c55)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c55)
    u6 = concatenate([u6, c5])
    c6 = Conv2D(64, (3, 3), activation='elu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='elu', padding='same') (c6)

    u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u71 = concatenate([u71, c4])
    c71 = Conv2D(32, (3, 3), activation='elu', padding='same') (u71)
    c61 = Conv2D(32, (3, 3), activation='elu', padding='same') (c71)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c61)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='elu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='elu', padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='elu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='elu', padding='same') (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='elu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='elu', padding='same') (c9)

    outputs = Conv2D(4, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model


# # Vanilla U-Net model w/ batch normalization:

# In[ ]:


def unet_bn(input_shape):
    """
    Unet with batch normalization
    """
    inputs = Input(input_shape)

    c1 = Conv2D(8, (3, 3), activation=None, padding='same') (inputs)
    c1 = Conv2D(8, (3, 3), activation=None, padding='same') (c1)
    bn1 = BatchNormalization()(c1)
    a1 = Activation('elu')(bn1)
    p1 = MaxPooling2D((2, 2), padding='same') (a1)

    c2 = Conv2D(16, (3, 3), activation=None, padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation=None, padding='same') (c2)
    bn2 = BatchNormalization()(c2)
    a2 = Activation('elu')(bn2)
    p2 = MaxPooling2D((2, 2), padding='same') (a2)

    c3 = Conv2D(32, (3, 3), activation=None, padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation=None, padding='same') (c3)
    bn3 = BatchNormalization()(c3)
    a3 = Activation('elu')(bn3)
    p3 = MaxPooling2D((2, 2), padding='same') (a3)

    c4 = Conv2D(64, (3, 3), activation=None, padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation=None, padding='same') (c4)
    bn4 = BatchNormalization()(c4)
    a4 = Activation('elu')(bn4)
    p4 = MaxPooling2D((2, 2), padding='same') (a4)

    c5 = Conv2D(64, (3, 3), activation=None, padding='same') (p4)
    c5 = Conv2D(64, (3, 3), activation=None, padding='same') (c5)
    bn5 = BatchNormalization()(c5)
    a5 = Activation('elu')(bn5)
    p5 = MaxPooling2D((2, 2), padding='same') (a5)

    c55 = Conv2D(128, (3, 3), activation=None, padding='same') (p5)
    c55 = Conv2D(128, (3, 3), activation=None, padding='same') (c55)
    bn55 = BatchNormalization()(c55)
    a55 = Activation('elu')(bn55)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (a55)
    u6 = concatenate([u6, a5])
    c6 = Conv2D(64, (3, 3), activation=None, padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation=None, padding='same') (c6)
    bn6 = BatchNormalization()(c6)
    a6 = Activation('elu')(bn6)

    u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (a6)
    u71 = concatenate([u71, a4])
    c71 = Conv2D(32, (3, 3), activation=None, padding='same') (u71)
    c61 = Conv2D(32, (3, 3), activation=None, padding='same') (c71)
    bn61 = BatchNormalization()(c61)
    a61 = Activation('elu')(bn61)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (a61)
    u7 = concatenate([u7, a3])
    c7 = Conv2D(32, (3, 3), activation=None, padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation=None, padding='same') (c7)
    bn7 = BatchNormalization()(c7)
    a7 = Activation('elu')(bn7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (a7)
    u8 = concatenate([u8, a2])
    c8 = Conv2D(16, (3, 3), activation=None, padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation=None, padding='same') (c8)
    bn8 = BatchNormalization()(c8)
    a8 = Activation('elu')(bn8)
    
    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (a8)
    u9 = concatenate([u9, a1], axis=3)
    c9 = Conv2D(8, (3, 3), activation=None, padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation=None, padding='same') (c9)
    bn9 = BatchNormalization()(c9)
    a9 = Activation('elu')(bn9)

    outputs = Conv2D(4, (1, 1), activation='sigmoid') (a9)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model


# # U-Net model w/ batch normalization & dropout:

# In[ ]:


def vanilla_unet_bnd(input_shape):
    """
    Unet with batch normalization and dropout
    """
    inputs = Input(input_shape)

    c1 = Conv2D(8, (3, 3), activation=None, padding='same') (inputs)
    d1 = Dropout(0.5)(c1)
    c1 = Conv2D(8, (3, 3), activation=None, padding='same') (d1)
    d1 = Dropout(0.5)(c1)
    bn1 = BatchNormalization()(d1)
    a1 = Activation('elu')(bn1)
    p1 = MaxPooling2D((2, 2), padding='same') (a1)

    c2 = Conv2D(16, (3, 3), activation=None, padding='same') (p1)
    d2 = Dropout(0.5)(c2)
    c2 = Conv2D(16, (3, 3), activation=None, padding='same') (d2)
    d2 = Dropout(0.5)(c2)
    bn2 = BatchNormalization()(d2)
    a2 = Activation('elu')(bn2)
    p2 = MaxPooling2D((2, 2), padding='same') (a2)

    c3 = Conv2D(32, (3, 3), activation=None, padding='same') (p2)
    d3 = Dropout(0.5)(c3)
    c3 = Conv2D(32, (3, 3), activation=None, padding='same') (d3)
    d3 = Dropout(0.5)(c3)
    bn3 = BatchNormalization()(d3)
    a3 = Activation('elu')(bn3)
    p3 = MaxPooling2D((2, 2), padding='same') (a3)

    c4 = Conv2D(64, (3, 3), activation=None, padding='same') (p3)
    d4 = Dropout(0.5)(c4)
    c4 = Conv2D(64, (3, 3), activation=None, padding='same') (d4)
    d4 = Dropout(0.5)(c4)
    bn4 = BatchNormalization()(d4)
    a4 = Activation('elu')(bn4)
    p4 = MaxPooling2D((2, 2), padding='same') (a4)

    c5 = Conv2D(64, (3, 3), activation=None, padding='same') (p4)
    d5 = Dropout(0.5)(c5)
    c5 = Conv2D(64, (3, 3), activation=None, padding='same') (d5)
    d5 = Dropout(0.5)(c5)
    bn5 = BatchNormalization()(d5)
    a5 = Activation('elu')(bn5)
    p5 = MaxPooling2D((2, 2), padding='same') (a5)

    c55 = Conv2D(128, (3, 3), activation=None, padding='same') (p5)
    d55 = Dropout(0.5)(c55)
    c55 = Conv2D(128, (3, 3), activation=None, padding='same') (d55)
    d55 = Dropout(0.5)(c55)
    bn55 = BatchNormalization()(d55)
    a55 = Activation('elu')(bn55)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (a55)
    u6 = concatenate([u6, a5])
    c6 = Conv2D(64, (3, 3), activation=None, padding='same') (u6)
    d6 = Dropout(0.5)(c6)
    c6 = Conv2D(64, (3, 3), activation=None, padding='same') (d6)
    d6 = Dropout(0.5)(c6)
    bn6 = BatchNormalization()(d6)
    a6 = Activation('elu')(bn6)

    u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (a6)
    u71 = concatenate([u71, a4])
    c71 = Conv2D(32, (3, 3), activation=None, padding='same') (u71)
    d71 = Dropout(0.5)(c71)
    c61 = Conv2D(32, (3, 3), activation=None, padding='same') (d71)
    d61 = Dropout(0.5)(c61)
    bn61 = BatchNormalization()(d61)
    a61 = Activation('elu')(bn61)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (a61)
    u7 = concatenate([u7, a3])
    c7 = Conv2D(32, (3, 3), activation=None, padding='same') (u7)
    d7 = Dropout(0.5)(c7)
    c7 = Conv2D(32, (3, 3), activation=None, padding='same') (d7)
    d7 = Dropout(0.5)(c7)
    bn7 = BatchNormalization()(d7)
    a7 = Activation('elu')(bn7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (a7)
    u8 = concatenate([u8, a2])
    c8 = Conv2D(16, (3, 3), activation=None, padding='same') (u8)
    d8 = Dropout(0.5)(c8)
    c8 = Conv2D(16, (3, 3), activation=None, padding='same') (d8)
    d8 = Dropout(0.5)(c8)
    bn8 = BatchNormalization()(d8)
    a8 = Activation('elu')(bn8)
    
    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (a8)
    u9 = concatenate([u9, a1], axis=3)
    c9 = Conv2D(8, (3, 3), activation=None, padding='same') (u9)
    d9 = Dropout(0.5)(c9)
    c9 = Conv2D(8, (3, 3), activation=None, padding='same') (d9)
    d9 = Dropout(0.5)(c9)
    bn9 = BatchNormalization()(d9)
    a9 = Activation('elu')(bn9)

    outputs = Conv2D(4, (1, 1), activation='sigmoid') (a9)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model


# # U-Net model w/ batch normalization & dropout revised: 

# In[ ]:


def unet_bnd_r(input_shape):
    """
    Unet with batch normalization and dropout (revised)
    """
    inputs = Input(input_shape)

    c1 = Conv2D(8, (3, 3), activation=None, padding='same') (inputs)
    c1 = Conv2D(8, (3, 3), activation=None, padding='same') (c1)
    bn1 = BatchNormalization()(c1)
    a1 = Activation('elu')(bn1)
    p1 = MaxPooling2D((2, 2), padding='same') (a1)
    d1 = Dropout(0.25)(p1)

    c2 = Conv2D(16, (3, 3), activation=None, padding='same') (d1)
    c2 = Conv2D(16, (3, 3), activation=None, padding='same') (c2)
    bn2 = BatchNormalization()(c2)
    a2 = Activation('elu')(bn2)
    p2 = MaxPooling2D((2, 2), padding='same') (a2)
    d2 = Dropout(0.25)(p2)

    c3 = Conv2D(32, (3, 3), activation=None, padding='same') (d2)
    c3 = Conv2D(32, (3, 3), activation=None, padding='same') (c3)
    bn3 = BatchNormalization()(c3)
    a3 = Activation('elu')(bn3)
    p3 = MaxPooling2D((2, 2), padding='same') (a3)
    d3 = Dropout(0.25)(p3)

    c4 = Conv2D(64, (3, 3), activation=None, padding='same') (d3)
    c4 = Conv2D(64, (3, 3), activation=None, padding='same') (c4)
    bn4 = BatchNormalization()(c4)
    a4 = Activation('elu')(bn4)
    p4 = MaxPooling2D((2, 2), padding='same') (a4)
    d4 = Dropout(0.25)(p4)

    c5 = Conv2D(64, (3, 3), activation=None, padding='same') (d4)
    c5 = Conv2D(64, (3, 3), activation=None, padding='same') (c5)
    bn5 = BatchNormalization()(c5)
    a5 = Activation('elu')(bn5)
    p5 = MaxPooling2D((2, 2), padding='same') (a5)
    d5 = Dropout(0.25)(p5)

    c55 = Conv2D(128, (3, 3), activation=None, padding='same') (d5)
    c55 = Conv2D(128, (3, 3), activation=None, padding='same') (c55)
    bn55 = BatchNormalization()(c55)
    a55 = Activation('elu')(bn55)
    d55 = Dropout(0.25)(a55)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (d55)
    u6 = concatenate([u6, a5])
    d6 = Dropout(0.25)(u6)
    c6 = Conv2D(64, (3, 3), activation=None, padding='same') (d6)
    c6 = Conv2D(64, (3, 3), activation=None, padding='same') (c6)
    bn6 = BatchNormalization()(c6)
    a6 = Activation('elu')(bn6)

    u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (a6)
    u71 = concatenate([u71, a4])
    d71 = Dropout(0.25)(u71)
    c71 = Conv2D(32, (3, 3), activation=None, padding='same') (d71)
    c61 = Conv2D(32, (3, 3), activation=None, padding='same') (c71)
    bn61 = BatchNormalization()(c61)
    a61 = Activation('elu')(bn61)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (a61)
    u7 = concatenate([u7, a3])
    d7 = Dropout(0.25)(u7)
    c7 = Conv2D(32, (3, 3), activation=None, padding='same') (d7)
    c7 = Conv2D(32, (3, 3), activation=None, padding='same') (c7)
    bn7 = BatchNormalization()(c7)
    a7 = Activation('elu')(bn7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (a7)
    u8 = concatenate([u8, a2])
    d8 = Dropout(0.25)(u8)
    c8 = Conv2D(16, (3, 3), activation=None, padding='same') (d8)
    c8 = Conv2D(16, (3, 3), activation=None, padding='same') (c8)
    bn8 = BatchNormalization()(c8)
    a8 = Activation('elu')(bn8)
    
    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (a8)
    u9 = concatenate([u9, a1], axis=3)
    d9 = Dropout(0.25)(u9)
    c9 = Conv2D(8, (3, 3), activation=None, padding='same') (d9)
    c9 = Conv2D(8, (3, 3), activation=None, padding='same') (c9)
    bn9 = BatchNormalization()(c9)
    a9 = Activation('elu')(bn9)

    outputs = Conv2D(4, (1, 1), activation='sigmoid') (a9)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model


# # U-Net model w/ batch normalization & ReLU:

# In[ ]:


def unet_bn_relu(input_shape):
    """
    Unet with batch normalization and ReLu
    """
    inputs = Input(input_shape)

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    bn1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2), padding='same') (bn1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    bn2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2), padding='same') (bn2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    bn3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2), padding='same') (bn3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    bn4 = BatchNormalization()(c4)
    p4 = MaxPooling2D((2, 2), padding='same') (bn4)

    c5 = Conv2D(64, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same') (c5)
    bn5 = BatchNormalization()(c5)
    p5 = MaxPooling2D((2, 2), padding='same') (bn5)

    c55 = Conv2D(128, (3, 3), activation='relu', padding='same') (p5)
    c55 = Conv2D(128, (3, 3), activation='relu', padding='same') (c55)
    bn55 = BatchNormalization()(c55)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (bn55)
    u6 = concatenate([u6, bn5])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)
    bn6 = BatchNormalization()(c6)

    u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (bn6)
    u71 = concatenate([u71, bn4])
    c71 = Conv2D(32, (3, 3), activation='relu', padding='same') (u71)
    c61 = Conv2D(32, (3, 3), activation='relu', padding='same') (c71)
    bn61 = BatchNormalization()(c61)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (bn61)
    u7 = concatenate([u7, bn3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)
    bn7 = BatchNormalization()(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (bn7)
    u8 = concatenate([u8, bn2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)
    bn8 = BatchNormalization()(c8)
    
    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (bn8)
    u9 = concatenate([u9, bn1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

    outputs = Conv2D(4, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model


# # U-Net model w/ batch normalization & ReLU & 1-deeper:

# In[ ]:


def unet_bn_relu_deeper(input_shape):
    """
    Unet with batch normalization and ReLu
    """
    inputs = Input(input_shape)

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    bn1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2), padding='same') (bn1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    bn2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2), padding='same') (bn2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    bn3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2), padding='same') (bn3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    bn4 = BatchNormalization()(c4)
    p4 = MaxPooling2D((2, 2), padding='same') (bn4)

    c5 = Conv2D(64, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same') (c5)
    bn5 = BatchNormalization()(c5)
    p5 = MaxPooling2D((2, 2), padding='same') (bn5)

    c6 = Conv2D(128, (3, 3), activation='relu', padding='same') (p5)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same') (c6)
    bn6 = BatchNormalization()(c6)
    p6 = MaxPooling2D((2, 2), padding='same') (bn6)
    
    c66 = Conv2D(256, (3, 3), activation='relu', padding='same') (p6)
    c66 = Conv2D(256, (3, 3), activation='relu', padding='same') (c66)
    bn66 = BatchNormalization()(c66)

    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (bn66)
    u7 = Cropping2D(cropping=((0, 0), (0, 1)))(u7)
    u7 = concatenate([u7, bn6])
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same') (c7)
    bn7 = BatchNormalization()(c7)
    
    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (bn7)
    u8 = concatenate([u8, bn5])
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same') (c8)
    bn8 = BatchNormalization()(c8)

    u91 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (bn8)
    u91 = concatenate([u91, bn4])
    c91 = Conv2D(32, (3, 3), activation='relu', padding='same') (u91)
    c91 = Conv2D(32, (3, 3), activation='relu', padding='same') (c91)
    bn91 = BatchNormalization()(c91)

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (bn91)
    u9 = concatenate([u9, bn3])
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same') (c9)
    bn9 = BatchNormalization()(c9)

    u10 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (bn9)
    u10 = concatenate([u10, bn2])
    c10 = Conv2D(16, (3, 3), activation='relu', padding='same') (u10)
    c10 = Conv2D(16, (3, 3), activation='relu', padding='same') (c10)
    bn10 = BatchNormalization()(c10)
    
    u11 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (bn10)
    u11 = concatenate([u11, bn1], axis=3)
    c11 = Conv2D(8, (3, 3), activation='relu', padding='same') (u11)
    c11 = Conv2D(8, (3, 3), activation='relu', padding='same') (c11)

    outputs = Conv2D(4, (1, 1), activation='sigmoid') (c11)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model


# # Training:

# In[ ]:


train_idx, val_idx = train_test_split(
    mask_count_df.index, random_state=2019, test_size=0.2
)
training = False
if training:
    data_path = "../input/understanding_cloud_organization/"
    train_csv_path = os.path.join(data_path,'train.csv')
    train_image_path = os.path.join(data_path,'train_images')
    df = pd.read_csv(train_csv_path)
    # drop the rows where at least one element is missing. 
    df.dropna(inplace=True)

    #  split Image_Label in Image_id and Label
    df['Image'] = df['Image_Label'].apply(lambda x: x.split('_')[0])
    df['Label'] = df['Image_Label'].apply(lambda x: x.split('_')[1])

    # drop Image_Label column
    df.drop(columns='Image_Label', inplace=True)
    mean_b = 0
    mean_g = 0
    mean_r = 0
    std_b = 0
    std_g = 0
    std_r = 0
    nr_imgs = 0
    for i, train_id in enumerate(train_idx):
        image_path = os.path.join(train_image_path, df['Image'].iloc[train_id])
        img = cv2.imread(image_path)
        mean, std = cv2.meanStdDev(img)
        mean_b += mean[0]
        mean_g += mean[1]
        mean_r += mean[2]
        std_b += std[1]
        std_g += std[1]
        std_r += std[2]
        nr_imgs += 1

    mean = ((mean_r[0]/nr_imgs)/255, (mean_g[0]/nr_imgs)/255, (mean_b[0]/nr_imgs)/255)
    std = ((std_r[0]/nr_imgs)/255, (std_g[0]/nr_imgs)/255, (std_b[0]/nr_imgs)/255)


# In[ ]:


BATCH_SIZE = 32
if training:
    train_generator = DataGenerator(
        train_idx, 
        df=mask_count_df,
        target_df=train_df,
        batch_size=BATCH_SIZE,
        reshape=(320, 480),
        augment=True,
        n_channels=3,
        n_classes=4,
        normalize=False,
        mean=mean,
        std=std
    )

    val_generator = DataGenerator(
        val_idx, 
        df=mask_count_df,
        target_df=train_df,
        batch_size=BATCH_SIZE, 
        reshape=(320, 480),
        augment=False,
        n_channels=3,
        n_classes=4,
        normalize=False,
        mean=mean,
        std=std
    )


# In[ ]:


if training:
    model = unet_bn_relu(input_shape=(320, 480, 3))
    model.compile(optimizer=Nadam(lr=0.0002), loss=bce_dice_loss, metrics=[dice_coef])
    model.summary()


# In[ ]:


if training:
    checkpoint = ModelCheckpoint('model_unet_bn_relu.h5', save_best_only=True)

    history = model.fit_generator(
        train_generator,
        validation_data=val_generator,
        callbacks=[checkpoint],
        epochs=20
    )


# # Evaluation:

# In[ ]:


if training:
    with open('history.json', 'w') as f:
        json.dump(str(history.history), f)

    history_df = pd.DataFrame(history.history)
    history_df[['loss', 'val_loss']].plot()
    history_df[['dice_coef', 'val_dice_coef']].plot()


# In[ ]:


if training:
    print(history_df.to_string())


# In[ ]:


def calc_dice_one_class(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_class1(y_true, y_pred, smooth=1):
    y_true_1 = y_true[:,:,:,0]
    y_pred_1 = y_pred[:,:,:,0]
    calc_dice_one_class(y_true_1, y_pred_1)
    return calc_dice_one_class(y_true_1, y_pred_1)

def dice_coef_class2(y_true, y_pred, smooth=1):
    y_true_2 = y_true[:,:,:,1]
    y_pred_2 = y_pred[:,:,:,1]
    calc_dice_one_class(y_true_2, y_pred_2)
    return calc_dice_one_class(y_true_2, y_pred_2)

def dice_coef_class3(y_true, y_pred, smooth=1):
    y_true_3 = y_true[:,:,:,2]
    y_pred_3 = y_pred[:,:,:,2]
    calc_dice_one_class(y_true_3, y_pred_3)
    return calc_dice_one_class(y_true_3, y_pred_3)

def dice_coef_class4(y_true, y_pred, smooth=1):
    y_true_4 = y_true[:,:,:,3]
    y_pred_4 = y_pred[:,:,:,3]
    calc_dice_one_class(y_true_4, y_pred_4)
    return calc_dice_one_class(y_true_4, y_pred_4)


# In[ ]:


model = unet_bn_relu(input_shape=(320, 480, 3))
if training:
    model.load_weights("model_unet_bn_relu.h5")
else:
    # Load pretrained model
    model.load_weights("../input/premodel/model_unet_bn_relu_pretrain.h5")

    
# Dice meteric for each class
model.compile(optimizer=Nadam(lr=0.0002), loss=bce_dice_loss, metrics=[dice_coef_class1, dice_coef_class2, dice_coef_class3, dice_coef_class4, dice_coef])


# In[ ]:


val_df = pd.DataFrame(val_idx)


# In[ ]:


val_df = mask_count_df.iloc[val_idx]


# # Dice for each class

# In[ ]:


Fish_df = []
Flower_df = []
Sugar_df = []
Gravel_df = []
All_class_df = []
for i in enumerate(val_idx):
    img = mask_count_df.iloc[i[1]]['ImageId']
    sample_generator = DataGenerator(
        [i[1]], 
        df=mask_count_df,
        target_df=train_df,
        batch_size=1, 
        reshape=(320, 480),
        augment=False,
        n_channels=3,
        n_classes=4,
        normalize=False,
    )

    score = model.evaluate(sample_generator)
    Fish_df.append([score[1], img, 'Fish'])
    Flower_df.append([score[2], img, 'Flower'])
    Gravel_df.append([score[3], img, 'Gravel'])
    Sugar_df.append([score[4], img, 'Sugar'])
    All_class_df.append([score[5], img, 'All'])


# In[ ]:


Fish_df = pd.DataFrame(Fish_df) 
Flower_df = pd.DataFrame(Flower_df) 
Sugar_df = pd.DataFrame(Sugar_df) 
Gravel_df = pd.DataFrame(Gravel_df) 
All_class_df = pd.DataFrame(All_class_df)


# In[ ]:


Fish_mean_score = np.asarray(Fish_df.iloc[:,0]).mean()
Flower_mean_score = np.asarray(Flower_df.iloc[:,0]).mean()
Sugar_mean_score = np.asarray(Sugar_df.iloc[:,0]).mean()
Gravel_mean_score = np.asarray(Gravel_df.iloc[:,0]).mean()
Class_comp_mean_score = np.asarray(pd.concat([Fish_df.iloc[:,0], Flower_df.iloc[:,0], Sugar_df.iloc[:,0], Gravel_df.iloc[:,0]]).mean())
All_class_mean_score = np.asarray(All_class_df.iloc[:,0]).mean()


# In[ ]:


print(f'Fish mean score: {Fish_mean_score}')
print(f'Flower mean score: {Flower_mean_score}')
print(f'Sugar mean score: {Sugar_mean_score}')
print(f'Gravel mean score: {Gravel_mean_score}')
print(f'Classes combined mean score: {Class_comp_mean_score}')
print(f'All class mean score: {All_class_mean_score}')


# In[ ]:


frames = [Fish_df, Flower_df, Sugar_df, Gravel_df]

val_total_df = pd.concat(frames)
val_total_df.columns = ['Score', 'img', 'Type']


# In[ ]:


All_class_df.columns = ['Score', 'img', 'Type']


# # Images with largest and smallest Dice coeff

# In[ ]:


val_largest_df = All_class_df.nlargest(5, 'Score')
val_smallest_df = All_class_df.nsmallest(5, 'Score')
print('Largest')
print(val_largest_df)
print('\nSmallest')
print(val_smallest_df)


# # Display images, predicetions and GT for largest Dice

# In[ ]:


columns = 4
rows = 5
fig, ax = plt.subplots(rows, columns, figsize=(18, 13))
ax[0, 0].set_title('Fish', fontsize=20)
ax[0, 1].set_title('Flower', fontsize=20)
ax[0, 2].set_title('Gravel', fontsize=20)
ax[0, 3].set_title('Sugar', fontsize=20)
j = 0
for i, row in val_largest_df.iterrows():
    idx = val_idx[i]
    img_name = row['img']
    img = plt.imread(f"{path}/train_images/{img_name}")
    ax[j, 0].imshow(img)
    ax[j, 1].imshow(img)
    ax[j, 2].imshow(img)
    ax[j, 3].imshow(img)
    
    
    label = row['Type']
    for k in range(4):
        if k == 0:
            label = 'Fish'
        elif k == 1:
            label = 'Flower'
        elif k == 2:
            label = 'Gravel'
        elif k == 3:
            label = 'Sugar'
        image_label = f'{img_name}_{label}'
        true_mask_rle = train.loc[train['Image_Label'] == image_label, 'EncodedPixels'].values[0]
        if true_mask_rle != true_mask_rle: 
            true_mask = np.zeros((1400, 2100))
        else:
            true_mask = rle_decode(true_mask_rle).astype('float32')  
        ax[j, k].imshow(true_mask, alpha=0.5, cmap='gray')
        
    if label == 'Fish':
        mask_idx = 0
    elif label == 'Flower':
        mask_idx = 1
    elif label == 'Gravel':
        mask_idx = 2
    elif label == 'Sugar':
        mask_idx = 3
    sample_generator = DataGenerator(
        [idx], 
        df=mask_count_df,
        target_df=train_df,
        batch_size=1, 
        reshape=(320, 480),
        augment=False,
        n_channels=3,
        n_classes=4,
        normalize=False,
    )
    mask = model.predict_generator(sample_generator)
    for k in range(4):
        pred_mask = mask[0,:,:,k]
        pred_mask = np_resize(pred_mask, (1400, 2100))
        ax[j, k].imshow(pred_mask, alpha=0.5, cmap='autumn')    
    j += 1


# # Display images, predicetions and GT for smallest Dice

# In[ ]:


columns = 4
rows = 5
fig, ax = plt.subplots(rows, columns, figsize=(18, 13))
ax[0, 0].set_title('Fish', fontsize=20)
ax[0, 1].set_title('Flower', fontsize=20)
ax[0, 2].set_title('Gravel', fontsize=20)
ax[0, 3].set_title('Sugar', fontsize=20)
j = 0
for i, row in val_smallest_df.iterrows():
    idx = val_idx[i]
    img_name = row['img']
    img = plt.imread(f"{path}/train_images/{img_name}")
    ax[j, 0].imshow(img)
    ax[j, 1].imshow(img)
    ax[j, 2].imshow(img)
    ax[j, 3].imshow(img)
    
    
    label = row['Type']
    for k in range(4):
        if k == 0:
            label = 'Fish'
        elif k == 1:
            label = 'Flower'
        elif k == 2:
            label = 'Gravel'
        elif k == 3:
            label = 'Sugar'
        image_label = f'{img_name}_{label}'
        true_mask_rle = train.loc[train['Image_Label'] == image_label, 'EncodedPixels'].values[0]
        if true_mask_rle != true_mask_rle: 
            true_mask = np.zeros((1400, 2100))
        else:
            true_mask = rle_decode(true_mask_rle).astype('float32')  
        ax[j, k].imshow(true_mask, alpha=0.5, cmap='gray')
        
    if label == 'Fish':
        mask_idx = 0
    elif label == 'Flower':
        mask_idx = 1
    elif label == 'Gravel':
        mask_idx = 2
    elif label == 'Sugar':
        mask_idx = 3
    sample_generator = DataGenerator(
        [idx], 
        df=mask_count_df,
        target_df=train_df,
        batch_size=1, 
        reshape=(320, 480),
        augment=False,
        n_channels=3,
        n_classes=4,
        normalize=False,
    )
    mask = model.predict_generator(sample_generator)
    for k in range(4):
        pred_mask = mask[0,:,:,k]
        pred_mask = np_resize(pred_mask, (1400, 2100))
        ax[j, k].imshow(pred_mask, alpha=0.5, cmap='autumn')    
    j += 1


# In[ ]:


if not training:
    model.load_weights("../input/premodel/model_unet_bn_relu_pretrain.h5")
else:
    model.load_weights("model_unet_bn_relu.h5")
    
test_df = []

for i in range(0, test_imgs.shape[0], 500):
    batch_idx = list(
        range(i, min(test_imgs.shape[0], i + 500))
    )

    test_generator = DataGenerator(
        batch_idx,
        df=test_imgs,
        shuffle=False,
        mode='predict',
        dim=(350, 525),
        reshape=(320, 480),
        n_channels=3,
        base_path='../input/understanding_cloud_organization/test_images',
        target_df=sub_df,
        batch_size=1,
        n_classes=4
    )

    batch_pred_masks = model.predict_generator(
        test_generator, 
        workers=1,
        verbose=1
    )

    for j, b in enumerate(batch_idx):
        filename = test_imgs['ImageId'].iloc[b]
        image_df = sub_df[sub_df['ImageId'] == filename].copy()

        pred_masks = batch_pred_masks[j, ].round().astype(int)
        pred_rles = build_rles(pred_masks, reshape=(350, 525))

        image_df['EncodedPixels'] = pred_rles
        test_df.append(image_df)


# # Submission

# In[ ]:


test_df = pd.concat(test_df)
test_df.drop(columns='ImageId', inplace=True)
test_df.to_csv('submission.csv', index=False)

