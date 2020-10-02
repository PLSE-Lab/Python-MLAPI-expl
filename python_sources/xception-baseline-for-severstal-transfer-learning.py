#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm_notebook
import cv2
from tensorflow.python.keras import backend as K
from sklearn.model_selection import train_test_split
from skimage.color import gray2rgb
import tensorflow as tf

import keras
from keras.layers import UpSampling2D, Conv2D, Activation, Conv2DTranspose
from keras import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.applications.xception import preprocess_input
from imgaug import augmenters as iaa


# In[ ]:


BATCH_SIZE = 32 # 8 # 16 # 4 # 64 # 128 # 8
EPOCHS = 80 # 95
IMG_SIZE = 256

train_dir = '../input/severstal-steel-defect-detection/train_images'


# In[ ]:


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_ids, image_dir, batch_size=32,
                 img_h=256, img_w=256, shuffle=False):
        
        self.list_ids = list_ids
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        'denotes the number of batches per epoch'
        return int(np.floor(len(self.list_ids)) / self.batch_size)
    
    def __getitem__(self, index):
        'generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # get list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]
        # generate data
        
        X, y = self.__data_generation(list_ids_temp)
#         X = self.augmentor(X)
        # return data 
        return X, y
    
    def on_epoch_end(self):
        'update ended after each epoch'
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_ids_temp):
        'generate data containing batch_size samples'
        X = np.empty((self.batch_size, self.img_h, self.img_w, 3))
        y = np.empty((self.batch_size, self.img_h, self.img_w, 1))
        
        for idx, id in enumerate(list_ids_temp):
            file_path =  os.path.join(self.image_dir, id)
            
            image = cv2.imread(file_path, 1)
            image_resized = cv2.resize(image, (self.img_w, self.img_h))
            image_resized = np.array(image_resized, dtype=np.float64)
            
            '''
            image = self.__load_grayscale(file_path)
            
            # Store samples
            image_resized = gray2rgb(image[:,:,0])
            '''
                        
            mask = np.empty((self.img_h, self.img_w, 1))
            
            rle_name = id + '_' + '4'
            rle = df_train[df_train['ImageId_ClassId'] == rle_name]['EncodedPixels'].values[0]
            
            class_mask = rle_to_mask(rle, width=1600, height=256) 
            class_mask_resized = cv2.resize(class_mask, (self.img_w, self.img_h))
            mask = class_mask_resized
            
            X[idx,] = image_resized
            y[idx,] = np.expand_dims(mask, -1)
        
#         X = self.augmentor(X)
        
        # normalize 
        X = X / 255
        y = (y > 0).astype(int)
            
        return X, y
    
    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_w, self.img_h))
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)

        return img
    
    def augmentor(self, images):
        'Apply data augmentation'
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
            [
                iaa.Sharpen((0.0, 1.0)),       # sharpen the image
                iaa.Fliplr(),
                iaa.Flipud(),
                iaa.ElasticTransformation(alpha=50, sigma=5)
                ],random_order=True
        )
        return seq.augment_images(images)


# In[ ]:


df_train = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')
print(len(df_train))
df_train.head()


# In[ ]:


'Only 4 class'
df_train = df_train[df_train['EncodedPixels'].notnull()].reset_index(drop=True)
df_train = df_train[df_train['ImageId_ClassId'].apply(lambda x: x.split('_')[1] == '4')].reset_index(drop=True)
print(len(df_train))
df_train.head()


# In[ ]:


df_train['ImageId'] = df_train['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
listdir = df_train['ImageId'].values
train, valid = train_test_split(listdir, train_size=0.8)
print(train[:2], valid[:2])
df_train.head()


# In[ ]:


def rle_to_mask(rle_string, height, width):
    
    rows, cols = height, width
    img = np.zeros(rows * cols, dtype=np.uint8)
    if len(str(rle_string)) > 1:
        rle_numbers = [int(numstring) for numstring in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1, 2)
        for index, length in rle_pairs:
            index -= 1
            img[index:index+length] = 255
    else: img = np.zeros(cols*rows)
    img = img.reshape(cols, rows)
    img = img.T
    return img


# In[ ]:


for x, y in DataGenerator(df_train['ImageId'], 
                          '../input/severstal-steel-defect-detection/train_images', 
                          batch_size=32, img_h=256, img_w=256, shuffle=True):
    break
    
print(x.shape, y.shape)


# In[ ]:


plt.imshow(np.squeeze(x[3]))


# In[ ]:


plt.imshow(np.squeeze(y[3]))


# In[ ]:


'metric and loss function for evaluation'
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def loss_dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return -K.log((2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


# In[ ]:


'load pretrained model'
from keras.applications import Xception
base_model = Xception(weights=None, input_shape=(IMG_SIZE,IMG_SIZE,3), include_top=False)
base_model.load_weights('../input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')


# In[ ]:


base_model.trainable = False


# In[ ]:


base_out = base_model.output # (8, 8)
conv1 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (base_out) # (8, 16, 16)
up = UpSampling2D(8, interpolation='bilinear')(conv1) # (8, 128, 128)
conv2 = Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same') (up) # (1, 256, 256)
conv3 = Conv2D(1, (1, 1))(conv2)
conv4 = Activation('sigmoid')(conv3)

lr=1e-6 # 1e-4 # 1e-6 # 1e-5 # 1e-4 # 0.0001 # 1e-2 # 1e-3
model = Model(input=base_model.input, output=conv4)
optimizer = keras.optimizers.RMSprop(lr=lr) # keras.optimizers.Adam(lr=lr) # keras.optimizers.Adam(lr=lr, decay=1e-6) #  # , decay = 1e-6
model.compile(optimizer=optimizer, loss=loss_dice_coef, metrics=[dice_coef]) # decay = 1e-6

# model.summary()
# for i, layer in enumerate(base_model.layers):
#     print("{} {}".format(i, layer.__class__.__name__))


# In[ ]:


train_generator = DataGenerator(train, train_dir, batch_size=BATCH_SIZE, shuffle=True)
train_size = len(train)
print(train_size)

val_generator = DataGenerator(valid, train_dir, batch_size=BATCH_SIZE)
train_size = len(valid)


# We use ModelCheckpoint and ReduceLROnPlateau callbacks. ModelCheckpoint monitors the loss metric  after each epoch and prints out whether the metric has improved. ReduceLROnPlateau reduces learning rate when a metric has stopped improving during a number of epochs.

# In[ ]:


from keras.callbacks import *


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(
        self,
        base_lr=0.001,
        max_lr=0.006,
        step_size=2000.0,
        mode="triangular",
        gamma=1.0,
        scale_fn=None,
        scale_mode="cycle",
    ):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == "triangular":
                self.scale_fn = lambda x: 1.0
                self.scale_mode = "cycle"
            elif self.mode == "triangular2":
                self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
                self.scale_mode = "cycle"
            elif self.mode == "exp_range":
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = "iterations"
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.0
        self.trn_iterations = 0.0
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.0

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == "cycle":
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations
            )

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault("lr", []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault("iterations", []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


# In[ ]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, # factor=0.1 #0.2
                                      patience=8, min_lr=1e-5) # 1e-6 # patience=5  # min_lr=1e-5 # 0.000001 # 0.0001

cyclic_lr = CyclicLR(
                mode="triangular",
                base_lr=1e-6,
                max_lr=1e-4,
                step_size=8 * (train_size / BATCH_SIZE),
            )

# Add model checkpoint
checkpoint = ModelCheckpoint("model_out.hdf5", monitor="val_loss", verbose=1, save_best_only=True)

es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=8)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model.fit_generator(generator=train_generator,\n                              validation_data=val_generator,\n                              epochs=EPOCHS,\n                              callbacks=[checkpoint, cyclic_lr],\n                              verbose=1)')


# In[ ]:


'''
# unfreeze the final set of CONV layers and make them trainable
for layer in base_model.layers[129:]: # 122:]: # 122 - 3 last Conv layers (Conv2D) # 126 - 2 last Conv layers (SeparableConv2D's) # 129
    layer.trainable = True

# lr=1e-6 #1e-2 # 1e-3 # 0.01 # 0.001
# optimizer = keras.optimizers.RMSprop(lr=lr) # keras.optimizers.Adam(lr=lr) # keras.optimizers.Adam(lr=lr) #  # 
model.compile(optimizer=optimizer, loss=loss_dice_coef, metrics=[dice_coef]) # 0.0001
'''


# In[ ]:


'''
%%time
history = model.fit_generator(generator=train_generator, 
                              epochs=80, #90, # 50, # 35, # 5 #EPOCHS,
                              validation_data=val_generator,
#                               steps_per_epoch=train_size//BATCH_SIZE,
                              callbacks=[checkpoint, reduce_lr], # cyclic_lr], # [ checkpoint, es],
                              verbose=1) #, shuffle=True)
'''


# In[ ]:


fig, ax = plt.subplots()

plt.plot(np.arange(len(history.history['loss'])) + 1, history.history['loss'], label='loss')
plt.plot(np.arange(len(history.history['val_loss'])) + 1, history.history['val_loss'], label='val_loss')

ax.legend()
plt.show()


# In[ ]:


pred = model.predict(x)
plt.imshow(np.squeeze(pred[3] > 0.5).astype(int))


# In[ ]:


testfiles=os.listdir("../input/severstal-steel-defect-detection/test_images/")
len(testfiles)


# In[ ]:


get_ipython().run_cell_magic('time', '', "test_img = []\nfor fn in tqdm_notebook(testfiles):\n        img = cv2.imread( '../input/severstal-steel-defect-detection/test_images/'+fn )\n        img = cv2.resize(img,(IMG_SIZE, IMG_SIZE))       \n        test_img.append(img)")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'predict = model.predict(np.array(test_img))\nprint(len(predict))')


# In[ ]:


def mask_to_rle(mask):
    '''
    Convert a mask into RLE
    
    Parameters: 
    mask (numpy.array): binary mask of numpy array where 1 - mask, 0 - background

    Returns: 
    sring: run length encoding 
    '''
    pixels= mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'pred_rle = []\nfor img in tqdm_notebook(predict):\n    img = cv2.resize(img, (1600, 256))\n    tmp = np.copy(img)\n    tmp[tmp<0.5] = 0\n    tmp[tmp>0] = 1\n    pred_rle.append(mask_to_rle(tmp))')


# In[ ]:


img_t = cv2.imread( '../input/severstal-steel-defect-detection/test_images/'+ testfiles[4])
plt.imshow(img_t)


# In[ ]:


mask_t = rle_to_mask(pred_rle[4], 256, 1600)
plt.imshow(mask_t)


# In[ ]:


sub = pd.read_csv( '../input/severstal-steel-defect-detection/sample_submission.csv', converters={'EncodedPixels': lambda e: ' '} )
sub.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "for fn, rle in zip(testfiles, pred_rle):\n    sub['EncodedPixels'][(sub['ImageId_ClassId'].apply(lambda x: x.split('_')[0]) == fn) & \\\n                        (sub['ImageId_ClassId'].apply(lambda x: x.split('_')[1] == '4'))] = rle")


# In[ ]:


img_s = cv2.imread( '../input/severstal-steel-defect-detection/test_images/'+ sub['ImageId_ClassId'][47].split('_')[0])
plt.imshow(img_s)


# In[ ]:


mask_s = rle_to_mask(sub['EncodedPixels'][47], 256, 1600)
plt.imshow(mask_s)


# In[ ]:


sub.to_csv('submission.csv', index=False)

