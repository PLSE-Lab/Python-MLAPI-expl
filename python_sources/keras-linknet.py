#!/usr/bin/env python
# coding: utf-8

# # Intro
# Basically a fork of the U-Net starter where U-Net is replaced with LinkNet. The success of [LinkNet for Carvana](http://slides.com/vladimiriglovikov/kaggle-deep-learning-to-create-a-model-for-binary-segmentation-of-car-images#/) and presentation by Vladimir Iglovikov made me think it might be worth a try for segmenting nuclei. 
# 
# The architecture used is the so-called [LinkNet](https://arxiv.org/abs/1707.03718), which is very common for image segmentation problems such as this. They also have a tendency to work quite well even on small datasets.
# 

# In[ ]:


import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed


# In[ ]:


# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]


# # Get the data
# Let's first import all the images and associated masks. I downsample both the training and test images to keep things light and manageable, but we need to keep a record of the original sizes of the test images to upsample our predicted masks and create correct run-length encodings later on. There are definitely better ways to handle this, but it works fine for now!

# In[ ]:


# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')


# Let's see if things look all right by drawing some random images and their associated masks.

# In[ ]:


# Check if training data looks all right
ix = random.randint(0, len(train_ids))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()


# Seems good!
# 
# # Create our Keras metric
# 
# Now we try to define the *mean average precision at different intersection over union (IoU) thresholds* metric in Keras. TensorFlow has a mean IoU metric, but it doesn't have any native support for the mean over multiple thresholds, so I tried to implement this. **I'm by no means certain that this implementation is correct, though!** Any assistance in verifying this would be most welcome! 
# 
# *Update: This implementation is most definitely not correct due to the very large discrepancy between the results reported here and the LB results. It also seems to just increase over time no matter what when you train ... *

# In[ ]:


# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# # Build and train our neural network
# Next we build our model, loosely based on [LinkNet](https://arxiv.org/abs/1707.03718)

# In[ ]:


from keras.models import Model
from keras.layers import Input, Conv2D, Deconv2D, MaxPool2D, concatenate, AvgPool2D
network_mode = 'bn'
s_c2 = lambda fc, k, s = 1, activation='elu', **kwargs: Conv2D(fc, kernel_size = (k,k), strides= (s,s),
                                       padding = 'same', activation = activation,
                                       **kwargs)
s_d2 = lambda fc, k, s = 1, activation='elu', **kwargs: Deconv2D(fc, kernel_size=(k,k), strides=(s,s), 
                                                       padding = 'same', activation=activation,
                                                       **kwargs)
if network_mode == 'bn':
    from keras.layers import BatchNormalization, Activation
    c2 = lambda fc, k, s = 1, **kwargs: lambda x: Activation('elu')(BatchNormalization()(
        Conv2D(fc, kernel_size = (k,k), strides= (s,s),
               padding = 'same', activation = 'linear', **kwargs)(x)))

    d2 = lambda fc, k, s = 1, **kwargs: lambda x: Activation('elu')(BatchNormalization()(
        Deconv2D(fc, kernel_size=(k,k), strides=(s,s), 
                 padding = 'same', activation='linear', **kwargs)(x)))
else:
    c2 = s_c2
    d2 = s_d2


# In[ ]:


# Build U-Net model
start_in = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name = 'Input')
start_scale = Lambda(lambda x: x / 255) (start_in)
# pre-processing
in_filt = c2(64, 7, 2)(start_scale)
in_mp = MaxPool2D((3,3), strides = (2,2), padding = 'same')(in_filt)


# In[ ]:


from keras import backend as K
from keras.regularizers import l2
from keras.layers import add

def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])

def enc_block(m, n):
    def block_func(x):
        cx = c2(n, 3)(c2(n, 3, 2)(x))
        cs1 = concatenate([AvgPool2D((2,2))(x), 
                           cx])
        cs2 = c2(n, 3)(c2(n, 3)(cs1))
        return concatenate([cs2, cs1])
    return block_func
def dec_block(m, n):
    def block_func(x):
        cx1 = c2(m//4, 1)(x)
        cx2 = d2(m//4, 3, 2)(cx1)
        return Dropout(0.1)(c2(n, 1)(cx2))
    return block_func


# In[ ]:


enc1 = enc_block(64, 64)(in_mp)
enc2 = enc_block(64, 128)(enc1)

dec2 = dec_block(64, 128)(enc2)
dec2_cat = _shortcut(enc1, dec2)
dec1 = dec_block(64, 64)(dec2_cat)

last_out = _shortcut(dec1, in_mp)


# In[ ]:


# post-processing
out_upconv = d2(32, 3, 2)(last_out)
out_conv = c2(32, 3)(out_upconv)
out = s_d2(1, 2, 2, activation = 'sigmoid')(out_conv)


# In[ ]:


from keras import backend as K
from keras.metrics import binary_crossentropy
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def bce_dice(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)-K.log(dice_coef(y_true, y_pred))


# In[ ]:


model = Model(inputs = [start_in], outputs = [out])    
model.compile(optimizer = 'adam', 
              loss = bce_dice, 
              metrics = ['binary_crossentropy', dice_coef, mean_iou])


# *Update: Changed to ELU units, added dropout.*
# 
# Next we fit the model on the training data, using a validation split of 0.1. We use a small batch size because we have so little data. I recommend using checkpointing and early stopping when training your model. I won't do it here to make things a bit more reproducible (although it's very likely that your results will be different anyway). I'll just train for 10 epochs, which takes around 10 minutes in the Kaggle kernel with the current parameters. 
# 
# *Update: Added early stopping and checkpointing and increased to 30 epochs.*

# In[ ]:


# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, 
                    validation_split=0.25, 
                    batch_size=16, epochs=10, 
                    callbacks=[earlystopper, checkpointer])


# All right, looks good! Loss seems to be a bit erratic, though. I'll leave it to you to improve the model architecture and parameters! 
# 
# # Make predictions
# 
# Let's make predictions both on the test set, the val set and the train set (as a sanity check). Remember to load the best saved model if you've used early stopping and checkpointing.

# In[ ]:


# Predict on train, val and test
model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou}, compile = False)
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))


# In[ ]:


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()


# The model is at least able to fit to the training data! Certainly a lot of room for improvement even here, but a decent start. How about the validation data?

# In[ ]:


# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()


# Not too shabby! Definitely needs some more training and tweaking.
# 
# # Encode and submit our results
# 
# Now it's time to submit our results. I've stolen [this](https://www.kaggle.com/rakhlin/fast-run-length-encoding-python) excellent implementation of run-length encoding.

# In[ ]:


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


# Let's iterate over the test IDs and generate run-length encodings for each seperate mask identified by skimage ...

# In[ ]:


new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))


# ... and then finally create our submission!

# In[ ]:


# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-1.csv', index=False)


# This scored 0.233 on the LB for me. That was with version 2 of this notebook; be aware that the results from the neural network are extremely erratic and vary greatly from run to run (version 3 is significantly worse, for example). Version 7 scores 0.277!
# 
# You should easily be able to stabilize and improve the results just by changing a few parameters, tweaking the architecture a little bit and training longer with early stopping.
# 
# **Have fun!**
# 
# LB score history:
# - Version 7: 0.277 LB

# In[ ]:




