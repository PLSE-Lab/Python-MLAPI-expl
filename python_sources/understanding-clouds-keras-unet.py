#!/usr/bin/env python
# coding: utf-8

# # Understanding Clouds

# ## General Information
# 
# If you like my notes please upvote this Kernel. Furthermore, if you find some mistakes or any problems and any suggestions, please leave a comment in the comments section to help me out. Many thanks in advance!

# ### Change log
# - V34: PR AUC Callback 
# - V30:vgg16, 50 epochs, threshold = 0.5
# - V26 : Droped some augmentations, 50 epochs, threshold = 0.5
# - V20: Added new augmentataions, optimize thresholds,
# - PSPNet
# - Gray Images + Unet1 ; 0.48
# - Unet + resnet50 + dice_conf, epochs up to 35, added some albumentions tools 
# - Unet + resnet34 + sm.losses.bce_jaccard_loss
# - Version 11: Using segmentation-models: Linknet + vgg16 
# - Version 8: Using segmentation-models: Unet + resnet34
# 

# ## References
# 
# - Most idea from [Satellite Clouds: U-Net with ResNet Boilerplate](https://www.kaggle.com/xhlulu/satellite-clouds-u-net-with-resnet-boilerplate)
# - Some from [segmentation in pytorch using convenient tools](https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools/notebook)

# ## Score Log
# 
# - [segmentation-moddels](https://github.com/qubvel/segmentation_models#models-and-backbones), Unet, resnet34, epochs = 30, BATCH_SIZE=32, score = 0.562

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.data import imread
import matplotlib.pyplot as plt
import cv2
import json

import multiprocessing

import albumentations as albu

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc

import keras
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import Adam

get_ipython().system('pip install segmentation-models --quiet')
import segmentation_models as sm

import warnings
warnings.filterwarnings('ignore')

import os, glob


# In[ ]:


num_cpu_cores = multiprocessing.cpu_count()


# ### Utilities Functions
# 
# - `np_resize` : Reize the image to a specified shape
# - `mask2rle` : Convert a mask image array to a Run Length Encoding 
# - `rle2mask` : Convert a Run length encoding to a mask image array 
# - `build_masks`
# - `build_rles`

# In[ ]:


def np_resize(img, input_shape,graystyle=False):
    """
    Reshape a numpy array, which is input_shape=(height, width), 
    as opposed to input_shape=(width, height) for cv2
    """

    height, width = input_shape
    resized_img = cv2.resize(img, (width, height))
    
    # keep dimension
    if graystyle:
        resized_img = resized_img[..., None]
        
    return resized_img
    
def mask2rle(img):
    '''
    img: a mask image, numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    
    img.T.flatten()
    image(width x height x channel), 
    from width -> height -> channel flatten a one dimension array 
    
    '''
    pixels= img.T.flatten() 
    pixels = np.concatenate([[0], pixels, [0]])  # add 0 to the beginning and end of array
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


# **Convert a image (width x height x channel) three dimension array to a one dimension array**
# 
# Example: 
# 
# There is a image (4x2x3 = 24 pixles)
# 
# ![https://github.com/gsaneryeeb/ARTS/blob/master/data/4x2x3.png?raw=true](https://github.com/gsaneryeeb/ARTS/blob/master/data/4x2x3.png?raw=true)
# 
# ```python
# image = np.array([[[1,9,17],
#                [5,13,21]],
#               
#               [[2,10,18],
#                [6,14,22]],
#               
#               [[3,11,19],
#                [7,15,23]],
#               
#               [[4,12,20],
#                [8,16,24]]])
# ```
# 
# [flatten()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flatten.html) Return a copy of the array collapsed in to one dimension.
# 
# ```
# image.flatten()
# 
# array([ 1,  9, 17,  5, 13, 21,  2, 10, 18,  6, 14, 22,  3, 11, 19,  7, 15,
#        23,  4, 12, 20,  8, 16, 24])
# ```
# 
# In Run length encoding, We need to expand a image into a one-dimensional array in order of width, height and channel. So we use the function `image.T.flatten()`
# 
# ```
# image.T.flatten()
# 
# array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
#        18, 19, 20, 21, 22, 23, 24])
# ```

# ## Loss function
# $$
# \mathrm{DC}=\frac{2 T P}{2 T P+F P+F N}=\frac{2|X \cap Y|}{|X|+|Y|}
# $$
# 
# $$
# \begin{array}{c}{\mathrm{DL}(p, \hat{p})=1-\frac{2 p \hat{p}+1}{p+\hat{p}+1}} \\ {\text { where } p \in\{0,1\} \text { and } 0 \leq \hat{p} \leq 1}\end{array}
# $$
#  
# Loss Function: $$B C E+D i c e$$

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


# ## Preprocessing and EDA

# In[ ]:


train_images = os.listdir('../input/train_images')
print(len(train_images))

test_images = os.listdir('../input/test_images')
print(len(test_images))


# - Split Image_Label into ImageId and Label
# - Add `hasMask` column

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df.head()

# Split Image_Label into ImageId and Label
train_df['ImageId'] = train_df['Image_Label'].apply(lambda x : x.split('_')[0])
train_df['Label'] = train_df['Image_Label'].apply(lambda x : x.split('_')[1])
train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()

print(train_df.shape)
train_df.head()


# In[ ]:


train_df.loc[train_df['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[1]).value_counts()


# Total training images is 5546, every images have four masks.(Fish, Flower, Gravel, Sugar), Total training dataset is 5546 * 4 = 22184

# In[ ]:


train_df.loc[train_df['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().value_counts()


# Only 266 images have all four masks.

# In[ ]:


mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()
mask_count_df.sort_values('hasMask', ascending=False, inplace=True)
print(mask_count_df.shape)
mask_count_df.head()


# In[ ]:


submission_df = pd.read_csv('../input/sample_submission.csv')
submission_df['ImageId'] = submission_df['Image_Label'].apply(lambda x : x.split('_')[0])
test_images = pd.DataFrame(submission_df['ImageId'].unique(), columns=['ImageId'])


# ## One-hot encoding classed

# In[ ]:


train_df.head()


# Mark: `train_ohe_df = train_ohe_df.groupby('ImageId')['Label'].agg(set).reset_index()` 
# 
# 1. Groupby *ImageId*
# 2. Set *Label* value

# In[ ]:


train_ohe_df = train_df[~train_df['EncodedPixels'].isnull()]
classes = train_ohe_df['Label'].unique()
train_ohe_df = train_ohe_df.groupby('ImageId')['Label'].agg(set).reset_index()
for class_name in classes:
    train_ohe_df[class_name] = train_ohe_df['Label'].map(lambda x: 1 if class_name in x else 0)
print(train_ohe_df.shape)
train_ohe_df.head()


# In[ ]:


# dictionary for fast access to ohe vectors
# key: ImageId
# value: ohe value
# {'0011165.jpg': array([1, 1, 0, 0]),
# '002be4f.jpg': array([1, 1, 1, 0]),...}
img_to_ohe_vector = {img: vec for img, vec in zip(train_ohe_df['ImageId'], train_ohe_df.iloc[:, 2:].values)}


# ## Stratified split into train and val

# In[ ]:


train_ohe_df['Label'].map(lambda x: str(sorted(list(x))))


# In[ ]:


train_idx, val_idx = train_test_split(mask_count_df.index, 
                                      random_state=42,
                                      stratify = train_ohe_df['Label'].map(lambda x: str(sorted(list(x)))),# sorting present classes in lexicographical order, just to be sure
                                      test_size = 0.2)


# In[ ]:


print('train length:',len(train_idx))
print('val length:',len(val_idx))


# ## Data Generator

# In[ ]:


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, df, target_df=None, mode='fit',
                 base_path='../input/train_images',
                 batch_size=32, dim=(1400, 2100), n_channels=3, reshape=None,
                 augment=False, n_classes=4, random_state=42, shuffle=True, graystyle=False):
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
        self.graystyle = graystyle
        
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
            
            return X, y
        
        elif self.mode == 'predict':
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
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.
            
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
    
    
    def __load_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.

        return img
    
    def __random_transform(self, img, masks):
        composition = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            # albu.RandomRotate90(p=1),
            # albu.RandomBrightness(),
            #albu.ElasticTransform(p=1,alpha=120,sigma=120*0.05,alpha_affine=120*0.03),
            albu.GridDistortion(p=0.5)])
            #albu.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5),
            #albu.ShiftScaleRotate(scale_limit=0.5,rotate_limit=30, shift_limit=0.1, p=1, border_mode=0)])
        
        composed = composition(image=img, mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']
        
        return aug_img, aug_masks
    
    def __augment_batch(self, img_batch, masks_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i, ], masks_batch[i, ] = self.__random_transform(
                img_batch[i, ], masks_batch[i, ])
        
        return img_batch, masks_batch
    
    def get_labels(self):
        if self.shuffle:
            images_current = self.list_IDs[:self.len * self.batch_size]
            labels = [img_to_ohe_vector[img] for img in images_current]
        else:
            labels = self.labels
        return np.array(labels)


# Generator instances

# In[ ]:


BATCH_SIZE = 32
BACKBONE = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
EPOCHS = 30
LEARNING_RATE = 3e-4
HEIGHT =320
WIDTH = 480
CHANNELS = 3
ES_PATIENCE = 5
RLROP_PATIENCE = 3
DECAY_DROP = 0.5
N_CLASSES = train_df['Label'].nunique()


# In[ ]:


train_generator = DataGenerator(train_idx, 
                                df=mask_count_df, 
                                target_df=train_df, 
                                batch_size=BATCH_SIZE,
                                reshape=(HEIGHT, WIDTH),
                                augment=True,
                                graystyle=False,
                                shuffle = True,
                                n_channels=CHANNELS,
                                n_classes=N_CLASSES)

train_eval_generator = DataGenerator(train_idx, 
                              df=mask_count_df, 
                              target_df=train_df, 
                              batch_size=BATCH_SIZE, 
                              reshape=(HEIGHT, WIDTH),
                              augment=False,
                              graystyle=False,
                              shuffle = False,
                              n_channels=CHANNELS,
                              n_classes=N_CLASSES)

val_generator = DataGenerator(val_idx, 
                              df=mask_count_df, 
                              target_df=train_df, 
                              batch_size=BATCH_SIZE, 
                              reshape=(HEIGHT, WIDTH),
                              augment=False,
                              graystyle=False,
                              shuffle = False,
                              n_channels=CHANNELS,
                              n_classes=N_CLASSES)


# In[ ]:


print("train_generator lengh is ", len(train_generator))
print("train_eval_generator lengh is ", len(train_eval_generator))
print("val_generator lengh is ", len(val_generator))


# ## PR(Pro)-AUC-based Callback
# 
# from [https://www.kaggle.com/samusram/cloud-classifier-for-post-processing](https://www.kaggle.com/samusram/cloud-classifier-for-post-processing)
# 
# Using Sklearn Precision Recall Curve and AUC
# 
# 1. to estimate AUC under precision recall curve for each class,
# 2. to early stop after 5 epochs of no improvement in mean PR AUC,
# 3. save a model with the best PR AUC in validation,
# 4. to reduce learning rate on PR AUC plateau.

# In[ ]:


class PrAucCallback(Callback):
    def __init__(self, data_generator, num_workers=num_cpu_cores, 
                 early_stopping_patience=5, 
                 plateau_patience=3, reduction_rate=0.5,
                 stage='train', checkpoints_path='checkpoints/'):
        super(Callback, self).__init__()
        self.data_generator = data_generator
        self.num_workers = num_workers
        self.class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']
        self.history = [[] for _ in range(len(self.class_names) + 1)] # to store per each class and also mean PR AUC
        self.early_stopping_patience = early_stopping_patience
        self.plateau_patience = plateau_patience
        self.reduction_rate = reduction_rate
        self.stage = stage
        self.best_pr_auc = -float('inf')
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)
        self.checkpoints_path = checkpoints_path
    
    
    def compute_pr_auc(self, y_true, y_pred):
        pr_auc_mean = 0
        print(f"\n{'#'*30}\n")
        for class_i in range(len(self.class_names)):
            precision, recall, _ = precision_recall_curve(y_true[:, class_i], y_pred[:, class_i])
            pr_auc = auc(recall, precision)
            pr_auc_mean += pr_auc/len(self.class_names)
            print(f"PR AUC {self.class_names[class_i]}, {self.stage}: {pr_auc:.3f}\n")
            self.history[class_i].append(pr_auc)        
        print(f"\n{'#'*20}\n PR AUC mean, {self.stage}: {pr_auc_mean:.3f}\n{'#'*20}\n")
        self.history[-1].append(pr_auc_mean)
        return pr_auc_mean
              
    def is_patience_lost(self, patience):
        if len(self.history[-1]) > patience:
            best_performance = max(self.history[-1][-(patience + 1):-1])
            return best_performance == self.history[-1][-(patience + 1)] and best_performance >= self.history[-1][-1]    
              
    def early_stopping_check(self, pr_auc_mean):
        if self.is_patience_lost(self.early_stopping_patience):
            self.model.stop_training = True    
              
    def model_checkpoint(self, pr_auc_mean, epoch):
        if pr_auc_mean > self.best_pr_auc:
            # remove previous checkpoints to save space
            for checkpoint in glob.glob(os.path.join(self.checkpoints_path, 'classifier_epoch_*')):
                os.remove(checkpoint)
        self.best_pr_auc = pr_auc_mean
        self.model.save(os.path.join(self.checkpoints_path, f'classifier_epoch_{epoch}_val_pr_auc_{pr_auc_mean}.h5'))              
        print(f"\n{'#'*20}\nSaved new checkpoint\n{'#'*20}\n")
              
    def reduce_lr_on_plateau(self):
        if self.is_patience_lost(self.plateau_patience):
            new_lr = float(keras.backend.get_value(self.model.optimizer.lr)) * self.reduction_rate
            keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"\n{'#'*20}\nReduced learning rate to {new_lr}.\n{'#'*20}\n")
        
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_generator(self.data_generator,
                                              verbose=1,
                                              workers=self.num_workers)
        y_true = self.data_generator.get_labels()
        # estimate AUC under precision recall curve for each class
        pr_auc_mean = self.compute_pr_auc(y_true, y_pred)
              
        if self.stage == 'val':
            # early stop after early_stopping_patience=4 epochs of no improvement in mean PR AUC
            self.early_stopping_check(pr_auc_mean)

            # save a model with the best PR AUC in validation
            self.model_checkpoint(pr_auc_mean, epoch)

            # reduce learning rate on PR AUC plateau
            self.reduce_lr_on_plateau()            
        
    def get_pr_auc_history(self):
        return self.history


# Instances callback

# In[ ]:


train_metric_callback = PrAucCallback(train_eval_generator)
val_callback = PrAucCallback(val_generator, stage='val')


# ## Model Architecture

# In[ ]:


def unet1(input_shape):
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


# ## Training

# In[ ]:


preprocess_input = sm.backbones.get_preprocessing(BACKBONE)

model = sm.Unet(encoder_name=BACKBONE,
                classes=N_CLASSES, 
                input_shape=(HEIGHT,WIDTH,CHANNELS), 
                activation=ACTIVATION)
"""

model = unet1((320,480,3))
"""


# In[ ]:


earlystopping = EarlyStopping(monitor='val_loss', 
                             mode='min', 
                             patience=ES_PATIENCE,
                             restore_best_weights=True,
                             verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              mode='min',
                              patience=RLROP_PATIENCE,
                              factor=DECAY_DROP,
                              min_lr=1e-6,
                              verbose=1)

metric_list = [dice_coef]
callback_list = [earlystopping, reduce_lr]
optimizer = Adam(lr = LEARNING_RATE)

model.compile(optimizer=optimizer, 
              loss=bce_dice_loss, 
              metrics=metric_list)
model.summary()


# In[ ]:


checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

history = model.fit_generator(
    train_generator,
    validation_data=val_generator,
    callbacks=callback_list,
    epochs=1,
    verbose=2
)


# ## Evaluation

# In[ ]:


with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['dice_coef', 'val_dice_coef']].plot()


# ### Visualizing train and val PR AUC

# In[ ]:


def plot_with_dots(ax, np_array):
    ax.scatter(list(range(1, len(np_array) + 1)), np_array, s=50)
    ax.plot(list(range(1, len(np_array) + 1)), np_array)


# Training and Validation PR AUC

# In[ ]:


pr_auc_history_train = train_metric_callback.get_pr_auc_history()
pr_auc_history_val = val_callback.get_pr_auc_history()

plt.figure(figsize=(10, 7))
plot_with_dots(plt, pr_auc_history_train[-1])
plot_with_dots(plt, pr_auc_history_val[-1])

plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Mean PR AUC', fontsize=15)
plt.legend(['Train', 'Val'])
plt.title('Training and Validation PR AUC', fontsize=20)
plt.savefig('pr_auc_hist.png')


# Training and Validation Loss

# In[ ]:


plt.figure(figsize=(10, 7))
plot_with_dots(plt, history_0.history['loss']+history_1.history['loss'])
plot_with_dots(plt, history_0.history['val_loss']+history_1.history['val_loss'])

plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Binary Crossentropy', fontsize=15)
plt.legend(['Train', 'Val'])
plt.title('Training and Validation Loss', fontsize=20)
plt.savefig('loss_hist.png')


# ## Predicting
# 
# 
# - try threshold = 0.9 size =25000
# 
# ### Optimize thresholds
# 
# Using predictions on validation dataset and validation label to optimize thresholds

# In[ ]:



"""
model.load_weights('model.h5')
valid_masks = []
encoded_pixels = []
VALID_BATCH_SIZE = 10
probabilities = np.zeros((len(val_idx),350,525)) # Train = 90% train_dataset, Valid = 10% train_dataset.

print("length val_idx =", len(val_idx))
#for i in range(0, len(val_idx), VALID_BATCH_SIZE):
val_gen = DataGenerator(val_idx, 
                        df=mask_count_df, 
                        target_df=train_df, 
                        batch_size=1, 
                        reshape=(320, 480),
                        augment=False,
                        graystyle=False,
                        n_channels=3,
                        n_classes=4)

pred_valid_masks = model.predict_generator(
    val_gen, 
    workers=1,
    verbose=1    
)




    image, masks = batch
    
    for mask in masks:
        if mask.shape != (350, 525):
            mask = cv2.resize(mask.astype('float32'), dsize=(525, 350), 
                              interpolation=cv2.INTER_LINEAR)
            valid_masks.append(mask)
            
    print("valid_masks length:", len(valid_masks))
        
    for j, probability in enumerate(output):
        print("probability shape:",probability.shape)
        if probability.shape != (350, 525):
            probability = cv2.resize(probability, dsize=(525, 350), 
                                     interpolation=cv2.INTER_LINEAR)
        print("after probability shape:",probability.shape)
        probabilities[i * 4 + j, :, :] = probability
"""


# In[ ]:


best_threshold = 0.5
best_size = 25000


# In[ ]:


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


# In[ ]:


sigmoid = lambda x: 1 / (1 + np.exp(-x))


# Predict output shape is (320,480,4). 4 is classes(Fish, Flower, Gravel, Surger)
# 
# Using `pred_masks[...,k] ` or `pred_masks[:,:,k]` get one class of mask image.

# In[ ]:


model.load_weights('model.h5')
test_df = []
encoded_pixels = []
TEST_BATCH_SIZE = 500

for i in range(0, test_images.shape[0], TEST_BATCH_SIZE):
    batch_idx = list(
        range(i, min(test_images.shape[0], i + TEST_BATCH_SIZE))
    )

    test_generator = DataGenerator(
        batch_idx,
        df=test_images,
        shuffle=False,
        mode='predict',
        dim=(350, 525),
        reshape=(320, 480),
        n_channels=3,
        graystyle=False,
        base_path='../input/test_images',
        target_df=submission_df,
        batch_size=1,
        n_classes=4
    )

    batch_pred_masks = model.predict_generator(
        test_generator, 
        workers=1,
        verbose=1
    ) 
    # Predict out put shape is (320X480X4)
    # 4  = 4 classes, Fish, Flower, Gravel Surger.
    
    for j, idx in enumerate(batch_idx):
        filename = test_images['ImageId'].iloc[idx]
        image_df = submission_df[submission_df['ImageId'] == filename].copy()
        
        # Batch prediction result set
        pred_masks = batch_pred_masks[j, ].round().astype(int)
        pred_rles = build_rles(pred_masks, reshape=(350, 525))
        
        image_df['EncodedPixels'] = pred_rles
        
        test_df.append(image_df)
        
        
        for k in range(pred_masks.shape[-1]):
            pred_mask = pred_masks[...,k].astype('float32') 
            
            if pred_mask.shape != (350, 525):
                pred_mask = cv2.resize(pred_mask, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                
            pred_mask, num_predict = post_process(sigmoid(pred_mask), best_threshold, best_size )
            
            if num_predict == 0:
                encoded_pixels.append('')
            else:
                r = mask2rle(pred_mask)
                encoded_pixels.append(r)
        """
        # pred_rles = build_rles(pred_masks, reshape=(350, 525))

            #image_df['EncodedPixels'] = encoded_pixels
            #test_df.append(image_df)
        """


# In[ ]:


submission_df['EncodedPixels'] = encoded_pixels
submission_df.to_csv('submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)


# encoded_pixels

# In[ ]:


#test_df = pd.concat(test_df)
#test_df.drop(columns='ImageId', inplace=True)
#test_df.to_csv('all_flower_submission.csv', index=False)


# ## TODO
# 
# -  Try threshold = 0.85, size =10000
