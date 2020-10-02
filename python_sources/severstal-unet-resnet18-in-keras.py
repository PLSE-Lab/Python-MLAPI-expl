#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">SEVERSTAL STEEL </font></center></h1>
# 
# 
# <img src="https://thumbs.dreamstime.com/b/roll-steel-sheet-factory-d-rendering-79415588.jpg" width="800"></img>
# 
# <br>

# This kernel makes use of **segmentation_models** library for **Keras**. This library makes building segmentation models with different architectures and different backbones really easy. Very friendly for beginners like me ! 
# 
# But to progress in this competition it will always help to know the intricacies of model, backbone and various training methods. 
# Source of the library : https://github.com/qubvel/segmentation_models
# 
# 
# **Tip : To change the architecture of the model, just import the particular model from segmentation_models library. The available architectures are FPN, LinkNet, PSPNet. The available backbones are many like ResNets, DenseNets, EfficientNets, etc.** 
# 
# **If you like the kernel, please upvote it. It motivates me . Happy Kaggling**

# Some code is borrowed from : https://www.kaggle.com/xhlulu/severstal-simple-2-step-pipeline
# 
# Thank you xhlulu. All beginners like me are learning a lot from you. 

# # Contents :
# * Importing Libraries
# * Arranging DataSet
# * Utility Functions
# * Building and Training Unet with ResNet18 Backbone
# * Plotting the History of Model

# In[ ]:


pip install segmentation-models


# ## Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd 
import cv2
import os
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import keras
import json
import tqdm
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
import gc
from segmentation_models import Unet
from sklearn.model_selection import train_test_split
from keras.utils import Sequence
from keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')
print(os.listdir('../input'))


# In[ ]:


seed = 2019
BATCH_SIZE = 8


# ## Arranging the Dataset

# In[ ]:


traindf = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')


# In[ ]:


traindf['ImageId'] = traindf['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
traindf['ClassId'] = traindf['ImageId_ClassId'].apply(lambda x: x.split('_')[1])
traindf['hasMask'] = ~traindf['EncodedPixels'].isna()


# In[ ]:


traindf.head()


# In[ ]:


mask_counts = traindf.groupby('ImageId')['hasMask'].sum().reset_index()
mask_counts.sort_values(by = 'hasMask', ascending = False).head()


# In[ ]:


mask_counts['hasMask'].value_counts().plot.bar()


# In[ ]:


mask_counts.shape


# In[ ]:


mask_counts = mask_counts.reset_index(drop = True)


# ## Utility Functions

# **Taken from** : https://www.kaggle.com/xhlulu/severstal-simple-2-step-pipeline

# In[ ]:


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

def build_masks(rles, input_shape):
    depth = len(rles)
    masks = np.zeros((*input_shape, depth))
    
    for i, rle in enumerate(rles):
        if type(rle) is str:
            masks[:, :, i] = rle2mask(rle, input_shape)
    
    return masks

def build_rles(masks):
    width, height, depth = masks.shape
    
    rles = [mask2rle(masks[:, :, i])
            for i in range(depth)]
    
    return rles


# ## DataGenerator on the Fly

# **Taken from** : https://www.kaggle.com/xhlulu/severstal-simple-2-step-pipeline
# 
# **Original work** : https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

# In[ ]:


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, df, target_df=None, mode='fit',
                 base_path='../input/severstal-steel-defect-detection/train_images',
                 batch_size=32, dim=(256, 1600), n_channels=3,
                 n_classes=4, random_state=2019, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
        self.base_path = base_path
        self.target_df = target_df
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = random_state
        
        self.on_epoch_end()

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
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        
        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            img_path = f"{self.base_path}/{im_name}"
            img = self.__load_rgb(img_path)
            
            # Store samples
            X[i,] = img

        return X
    
    def __generate_y(self, list_IDs_batch):
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)
        
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            image_df = self.target_df[self.target_df['ImageId'] == im_name]
            
            rles = image_df['EncodedPixels'].values
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


# In[ ]:


all_index = mask_counts.index
trn_idx, val_idx = train_test_split(all_index, test_size = 0.2, random_state = seed)


# In[ ]:


train_generator = DataGenerator(
    trn_idx, 
    df=mask_counts,
    target_df=traindf,
    batch_size=BATCH_SIZE, 
    n_classes=4,
    random_state = seed
)

val_generator = DataGenerator(
    val_idx, 
    df=mask_counts,
    target_df=traindf,
    batch_size=BATCH_SIZE, 
    n_classes=4,
    random_state = seed
)


# In[ ]:


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# ## Building and Training Unet with Resnet18 as Backbone

# Other backbones and architectures can be found here : https://github.com/qubvel/segmentation_models

# In[ ]:


model = Unet('resnet18', classes=4, activation='softmax', input_shape = (256,1600,3))


# In[ ]:


model.summary()


# **Jaccard Loss** : This loss is usefull when you have unbalanced classes within a sample such as segmenting each pixel of an image
# 
# **Read more** : https://segmentation-models.readthedocs.io/en/latest/api.html

# In[ ]:


model.compile(Adam(lr = 0.005), loss=bce_jaccard_loss, metrics=[iou_score, dice_coef])


# In[ ]:


checkpoint = ModelCheckpoint(
    'Unet_resnet18.h5', 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=False,
    mode='auto'
)

reducelr = ReduceLROnPlateau(monitor = 'val_loss', min_lr = 1e-6, factor = 0.1, verbose = 1, patience = 5)

history = model.fit_generator(
    train_generator,
    validation_data=val_generator,
    callbacks=[checkpoint, reducelr],
    use_multiprocessing=True,
    workers=6,
    epochs=15
)


# ## Plotting history of the Model

# In[ ]:


with open('history.json', 'w') as f:
    json.dump(history.history, f)
    

history_df = pd.DataFrame(history.history)

plt.figure(figsize = (15,5))
plt.title('Plot of Loss')
history_df[['loss', 'val_loss']].plot()

plt.figure(figsize = (15,5))
plt.title('Plot of Dice Coefficient')
history_df[['dice_coef', 'val_dice_coef']].plot()

plt.figure(figsize = (15,5))
plt.title('Plot of IOU score')
history_df[['iou_score', 'val_iou_score']].plot()

