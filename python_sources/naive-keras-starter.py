#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm_notebook
from operator import itemgetter
from functools import partial
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


n_images_train = 1000 # Limited for a kernel


# In[ ]:


df_train = pd.read_csv('../input/train.csv', nrows=n_images_train*4)
train_images_path = Path('../input/train_images')


# In[ ]:


# Transform class to col
df_train['fname'], df_train['cls'] = zip(*df_train['ImageId_ClassId'].str.split('_'))
df_train['cls'] = df_train['cls'].astype(int)
df_train_pivot = df_train.pivot('fname', 'cls', 'EncodedPixels')
df_train_pivot.head()


# In[ ]:


def rle2mask(rle, width, height):
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return np.flipud( np.rot90( mask.reshape(width, height), k=1 ) )


# In[ ]:


def row_to_xy(row:pd.Series, image_path, return_weights=False, weights_border_size=50):
    fname = row.name
    X = cv2.imread(str(image_path / fname), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 127.5 - 1
    X = X[..., np.newaxis]  # Add channel info

    height, width = X.shape[:2]

    y = np.zeros((height, width, 5), dtype=np.uint8)  # Add class-zero
    for cls in range(1, 5):
        if isinstance(row[cls], str):
            y[:, :, cls] = rle2mask(row[cls], width, height)

    assert y.sum(-1).max() <= 1, f"Invalid image anntation for {fname}"
    
    # Add zero-class channel
    y[..., 0] = 1-y.sum(axis=-1)
    
    # Return weights (will give wight to non-zero class)
    if return_weights:
        # Dilate and apply blur to focus on border of class
        sw = cv2.blur(cv2.dilate(y[..., 1:].max(-1) * 255, np.ones((weights_border_size, weights_border_size))), (weights_border_size//2, weights_border_size//2))
        sw = sw.astype(np.float32) / 255
    
        # Sample-weight will be used as input (trick to get it to work on latest channel)
        return [X, sw[..., np.newaxis]], y
    else:
        return X, y


# In[ ]:


X, y = row_to_xy(df_train_pivot.iloc[0], train_images_path)


# In[ ]:


plt.figure(figsize=(20, 5))
plt.imshow(X[..., 0], cmap='gray', interpolation='bilinear')
colors = 'rgbk'
for cls in range(1, 5):
    plt.contour(y[..., cls], colors=colors[cls-1]);


# In[ ]:


# Split df in train and valida
df_train_pivot, df_valid_pivot = train_test_split(df_train_pivot, test_size=0.15)


# ## Keras baseline

# In[ ]:


from keras.layers import *
from keras.models import Model
from keras.utils import Sequence
from keras.optimizers import *
from keras.callbacks import *
import keras.backend as K


# In[ ]:


from math import ceil

class SteelDataset(Sequence):
    def __init__(self, df_pivot, batch_size, image_path, with_sw=True):
        self.df_pivot = df_pivot
        self.batch_size = batch_size
        self.row_to_xy = partial(row_to_xy, image_path=image_path, return_weights=with_sw)
        self.with_sw = with_sw

    def __len__(self):
        return ceil(self.df_pivot.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        batch_df = self.df_pivot.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        n_batch = len(batch_df)
        
        X_batch = np.empty((n_batch, 256, 1600, 1), dtype=np.float32)
        y_batch = np.empty((n_batch, 256, 1600, 5), dtype=np.float32)
        
        if self.with_sw:
            sw_batch = np.empty((n_batch, 256, 1600, 1), dtype=np.float32)
        
            for i, (_, row) in enumerate(batch_df.iterrows()):
                (X_batch[i], sw_batch[i]), y_batch[i] = self.row_to_xy(row)
            
            # SW will be used as input (trick to make it work on output)
            return {'img': X_batch, 'sw': sw_batch}, y_batch
        else:
            for i, (_, row) in enumerate(batch_df.iterrows()):
                X_batch[i], y_batch[i] = self.row_to_xy(row)
                
            return X_batch, y_batch


# In[ ]:


seq_train = SteelDataset(df_train_pivot, 4, train_images_path)
seq_valid = SteelDataset(df_valid_pivot, 4, train_images_path)


# In[ ]:


Xsw, y = seq_train[0]
X = Xsw['img']
sw = Xsw['sw']


# In[ ]:


X.shape, y.shape, sw.shape


# In[ ]:


fig, ax = plt.subplots(3, 1, figsize=(20, 10))
ax[0].imshow(X[2, ..., 0], cmap='gray', vmin=-1, vmax=1)
ax[1].imshow(y[2, ..., 1:].max(-1), vmin=0, vmax=1)
ax[2].imshow(sw[2, ..., 0], vmin=0, vmax=1);


# In[ ]:


drop_p = 0.1
inp = Input((256, 1600, 1), name='img')
inp_sw = Input((256, 1600, 1), name='sw')

convs_configs = [
    [
        (32, 3, 1),
    ], [
        (64, 3, 1),
        (64, 3, 2),
        (64, 3, 3),
        (64, 3, 4),
    ], [
        (64, 3, 1),
        (64, 3, 2),
        (64, 3, 3),
        (64, 3, 4),
    ]
]

x = inp
for g, conv_group_config in enumerate(convs_configs):
    x_group = []
    for c, (n_filters, kernel_size, dilation_rate) in enumerate(conv_group_config):
        conv = Conv2D(n_filters, kernel_size, dilation_rate=dilation_rate, padding='same', name=f'conv_{g}_{c}')(x)
        x_group.append(conv)
        
    if len(x_group) == 1:
        x = x_group[0]
    else:
        x = Add(name=f'add_{g}')(x_group)
        
    x = BatchNormalization(name=f'bn_{g}')(x)
    x = Activation('relu', name=f'relu_{g}')(x)
    x = SpatialDropout2D(drop_p, name=f'drop_{g}')(x)

# Create one output for each model
x_outs = []
models_cls = []
for cls in range(5):
    x_out = Conv2D(1, 1, use_bias=False)(x)  # Remove bias in order to avoid overfit
    model_cls = Model(inp, x_out)
    x_outs.append(x_out)
    models_cls.append(model_cls)
x_out_all = concatenate(x_outs, -1)
x_out_all = Activation('softmax')(x_out_all)

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)


def dice_coef_loss_func(mask):
    def dice_coef_loss(y_true, y_pred):
        return 1-dice_coef(y_true * mask, y_pred * mask)
    return dice_coef_loss

model = Model([inp, inp_sw], x_out_all)
model_no_sw = Model(inp, x_out_all)

optim = Adam()
model.compile(optim, dice_coef_loss_func(inp_sw), metrics=['accuracy', dice_coef])

model.summary()


# In[ ]:


cbs = [
    ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.1, verbose=True)
]

history = model.fit_generator(seq_train, epochs=15, validation_data=seq_valid, callbacks=cbs)


# In[ ]:


pd.DataFrame(history.history)[['loss', 'val_loss']].plot()


# In[ ]:


# Based on https://gist.github.com/JDWarner/6730747
def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


# In[ ]:


seq_valid_pred = SteelDataset(df_valid_pivot, 16, train_images_path, with_sw=False)


# In[ ]:


n_valid = len(df_valid_pivot)
y_true_cls = np.empty((n_valid, 256, 1600), dtype=np.bool)
y_pred_cls = np.empty((n_valid, 256, 1600), dtype=np.float32)

# Use model cls to predict each class (will avoid out of memory)
for cls in range(2, 5):
    model_cls = models_cls[cls]
    i = 0
    for X, y in tqdm_notebook(seq_valid_pred):
        n_batch = len(X)
        y_pred = model_cls.predict(X)  # Note that predict will run BEFORE softmax. So output is free from 0-1 range
        
        y_true_cls[i:i+n_batch] = (y[..., cls] > 0)
        y_pred_cls[i:i+n_batch] = y_pred[..., 0]
        
        i += n_batch
    break


# In[ ]:


vmin, vmax = np.percentile(y_pred_cls.ravel(), [0.1, 99.9])
vmin, vmax


# In[ ]:


plt.hist(y_pred_cls.ravel(), bins=51, range=(vmin, vmax));


# In[ ]:


t0 = (vmin + vmax) / 2
dice(y_true_cls, y_pred_cls > t0)


# In[ ]:


thres = np.linspace(vmin, vmax, 51)
dices = np.array([dice(y_true_cls, y_pred_cls > t) for t in tqdm_notebook(thres)])


# In[ ]:


plt.plot(thres, dices);


# In[ ]:


from scipy.optimize import minimize
def loss(x):
    r = dice(y_true_cls, y_pred_cls > x[0])
    print(x,r)
    return -r
minimize(loss, [t0], method='Nelder-Mead')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'minimize')


# In[ ]:


y_pred_cls = y_pred.argmax(-1)
y_pred_cls.shape


# In[ ]:


np.bincount(y_pred_cls.ravel())


# *Work in progress... still need to make submission process*
