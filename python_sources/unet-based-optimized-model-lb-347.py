#!/usr/bin/env python
# coding: utf-8

# # Intro
# # Unet based optimized model code LB .347
# # Approximatley 100 eoochs val_dice_coef - 91%
# 
# Most of the code is from the below kernels.. 
# https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
# https://www.kaggle.com/shenmbsw/data-augmentation-and-tensorflow-u-net

# In[ ]:


import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import h5py


import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from skimage import img_as_uint



from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, UpSampling2D
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import keras
from keras import optimizers

get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
import sklearn
import skimage
from skimage import transform
from os import environ

seed = 42

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Skimage      :', skimage.__version__)
print('Scikit-learn :', sklearn.__version__)
print('Keras        :', keras.__version__)
print('Tensorflow   :', tf.__version__)


# In[ ]:


IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = 'input/stage1_train/'
TEST_PATH = 'input/stage1_test/'


# In[ ]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[ ]:


train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]


# In[ ]:


def read_image_labels(image_id, aug_idx=-1, read_aug=False):
    # most of the content in this function is taken from 'Example Metric Implementation' kernel 
    # by 'William Cukierski'
    if read_aug == False:
        image_file = "{}{}/images/{}.png".format(TRAIN_PATH,image_id,image_id)
        mask_file = "{}{}/masks/*.png".format(TRAIN_PATH, image_id)
    else:
        image_file = "{}{}/augs/{}_{}.png".format(TRAIN_PATH,image_id,image_id,aug_idx)
        mask_file = "{}{}/augs_masks/{}_{}.png".format(TRAIN_PATH,image_id,image_id,aug_idx)
    
    image = skimage.io.imread(image_file)[:,:,:IMG_CHANNELS]
    if read_aug == False:
        masks = skimage.io.imread_collection(mask_file).concatenate()    
        height, width, _ = image.shape
        num_masks = masks.shape[0]
        label = np.zeros((height, width), np.bool)
        for index in range(0, num_masks):
            label[masks[index] > 0] = index + 1
    else:
        label = skimage.io.imread(mask_file)
        label = np.copy(label).astype('bool')

    return image, label


# In[ ]:


def data_aug(image,label,angel=30,resize_rate=0.9):
    
    flip = random.randint(0, 1)
    
    w = image.shape[0]
    h = image.shape[1]

    rwsize = random.randint(np.floor(random.uniform(0.5, 0.9)*w), w)
    rhsize = random.randint(np.floor(random.uniform(0.5, 0.9)*h), h)
    w_s = random.randint(0, w - rwsize)
    h_s = random.randint(0, h - rhsize)
        
    #print(image.shape, w_s, rwsize, h_s, rhsize)
    
    sh = random.random()/2-0.25
    rotate_angel = random.random()/180*np.pi*angel
    # Create Afine transform
    afine_tf = transform.AffineTransform(shear=sh,rotation=rotate_angel)
    # Apply transform to image data
    image = transform.warp(image, inverse_map=afine_tf,mode='edge')
    label = transform.warp(label, inverse_map=afine_tf,mode='edge')
    
    # Randomly corpping image frame
    image = image[w_s:w_s+rwsize, h_s:h_s+rhsize,:]
    label = label[w_s:w_s+rwsize, h_s:h_s+rhsize]
    
    # Ramdomly flip frame
    if flip:
        image = image[:,::-1,:]
        label = label[:,::-1]
    #print(image.dtype, label.dtype)
    image = resize(image, (IMG_HEIGHT, IMG_WIDTH, 3), mode='constant', preserve_range=True)
    image = skimage.img_as_ubyte(image)
    label = resize(label, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    label = np.copy(label).astype('bool')
    
    return image, label


# In[ ]:


def save_single_data_augmentation(image_id, idx, new_image, new_label):
    image,labels = read_image_labels(image_id)
    if not os.path.exists("{}{}/augs/".format(TRAIN_PATH,image_id)):
        os.makedirs("{}{}/augs/".format(TRAIN_PATH, image_id))
    if not os.path.exists("{}{}/augs_masks/".format(TRAIN_PATH,image_id)):
        os.makedirs("{}{}/augs_masks/".format(TRAIN_PATH, image_id))
            
    # also save the original image in augmented file 
    #plt.imsave(fname="../input/stage1_train/{}/augs/{}.png".format(image_id,image_id), arr = image)
    #plt.imsave(fname="../input/stage1_train/{}/augs_masks/{}.png".format(image_id,image_id),arr = labels)

    #for i in range(split_num):
    aug_img_dir = "{}{}/augs/{}_{}.png".format(TRAIN_PATH,image_id,image_id,idx)
    aug_mask_dir = "{}{}/augs_masks/{}_{}.png".format(TRAIN_PATH,image_id,image_id,idx)
    skimage.io.imsave(aug_img_dir, new_image)
    skimage.io.imsave(aug_mask_dir, img_as_uint(new_label))
    return aug_img_dir, aug_mask_dir


# In[ ]:


def make_data_augmentation(image_ids,split_num):
    for ax_index, image_id in tqdm(enumerate(image_ids),total=len(image_ids)):
        image, labels = read_image_labels(image_id)
        for i in range(split_num):
            new_image, new_label = data_aug(image,labels,angel=5,resize_rate=0.9)
            save_single_data_augmentation(image_id, i, new_image, new_label)


# In[ ]:


import shutil
def clean_data_augmentation(image_ids):
    for ax_index, image_id in tqdm(enumerate(image_ids),total=len(image_ids)):
        if os.path.exists("{}{}/augs/".format(TRAIN_PATH, image_id)):
            shutil.rmtree("{}{}/augs/".format(TRAIN_PATH, image_id))
        if os.path.exists("{}{}/augs_masks/".format(TRAIN_PATH, image_id)):
            shutil.rmtree("{}{}/augs_masks/".format(TRAIN_PATH, image_id))


# In[ ]:


plt.close('all')
for i in range(0, 10):
        image_id = train_ids[i]
        fig, axes = plt.subplots(1, 4, figsize=(18, 18))
        image, labels = read_image_labels(image_id)
        
        image = resize(image, (IMG_HEIGHT, IMG_WIDTH, 3), mode='constant', preserve_range=True)
        image = np.copy(image).astype('uint8')
        labels = resize(labels, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        labels = np.copy(labels).astype('bool')
    
        j=0
        axes[j].imshow(image)
        axes[j].axis('off')
        axes[j].set_title('{} th X'.format(image_id[:4]))
        
        j+=1
        axes[j].imshow(np.squeeze(labels))
        axes[j].axis('off')
        axes[j].set_title('{} th Y'.format(image_id[:4]))

        #fig, axes = plt.subplots(1, 2, figsize=(18, 18))

        new_image, new_label = data_aug(image, labels, angel=10, resize_rate=0.9)
        #print(new_image[:2,:2,1:2])
        w = random.randint(10,15)
        #print(new_image.dtype, new_image.shape, new_image[10:w,10:w])
        #print(new_label.dtype, new_label.shape, new_label[10:w,10:w])
        aug_img_file, aug_mask_file = save_single_data_augmentation(image_id, 0, new_image, new_label)
            
        new_image, new_label = read_image_labels(image_id, 0, True)
        new_label = np.expand_dims(new_label, axis=-1)
        #print(new_image.dtype, new_image.shape, new_image[10:w,10:w])
        #print(new_label.dtype, new_label.shape, new_label[10:w,10:w])

        j+=1
        axes[j].imshow(new_image)
        axes[j].axis('off')
        axes[j].set_title('{} th aug X'.format(image_id[:4]))
        
        j+=1
        axes[j].imshow(np.squeeze(new_label))
        axes[j].axis('off')
        axes[j].set_title('{} th aug Y'.format(image_id[:4]))


# In[ ]:


clean_data_augmentation(train_ids)


# In[ ]:


make_data_augmentation(train_ids, 3)


# # Get the data
# Get X_train Data and artificial data

# In[ ]:


# Get and resize train images and masks
X_train = np.zeros((len(train_ids) + len(train_ids)*3, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids) + len(train_ids)*3, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
count = 0
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    image, label = read_image_labels(id_)
    image = resize(image, (IMG_HEIGHT, IMG_WIDTH, 3), mode='constant', preserve_range=True)
    
    label = resize(label, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
    label = np.copy(label).astype('bool')
    
    idx = n + count * 3
    
    X_train[idx] = image
    Y_train[idx] = label

    for i in range(3):
        idx +=1
        new_image, new_label = read_image_labels(id_, i, True)
        new_label = np.expand_dims(new_label, axis=-1)
        X_train[idx] = new_image
        Y_train[idx] = new_label

    count +=1
        
        
print('Done!')


# In[ ]:


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


plt.close('all')
for i in range(0, 20):
        image_id = random.randint(0, (len(train_ids) + len(train_ids)*3-1))
        fig, axes = plt.subplots(1, 3, figsize=(18, 18))
    
        j=0
        axes[j].imshow(X_train[image_id])
        axes[j].axis('off')
        axes[j].set_title('{} th X'.format(image_id))
        
        j+=1
        axes[j].imshow(np.squeeze(Y_train[image_id]))
        axes[j].axis('off')
        axes[j].set_title('{} th Y'.format(image_id))
        
        image_id = random.randint(0, len(test_ids))

        j+=1
        axes[j].imshow(X_test[image_id])
        axes[j].axis('off')
        axes[j].set_title('{} th X_test'.format(image_id))


# In[ ]:


print(X_train.shape, Y_train.shape)
print(X_test.shape)


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


# In[ ]:


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# In[ ]:


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


# In[ ]:


def conv_layer(x, filters, k_size, act_func, dropout, batch_norm, name, k_init='he_normal', axis=3):
    conv = Conv2D(filters, k_size, kernel_initializer=k_init, name=name, padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation(act_func)(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)
    return conv


# In[ ]:


def keras_model_Unet_2D(img_width=256, img_height=256):
    '''
    Modified from https://keunwoochoi.wordpress.com/2017/10/11/u-net-on-keras-2-0/
    '''
    n_ch_exps = [4, 5, 6, 7, 8, 9]   #the n-th deep channel's exponent i.e. 2**n 16,32,64,128,256
    k_size = (3, 3)                  #size of filter kernel
    k_init = 'he_normal'             #kernel initializer

    if K.image_data_format() == 'channels_first':
        ch_axis = 1
        input_shape = (3, img_width, img_height)
    elif K.image_data_format() == 'channels_last':
        ch_axis = 3
        input_shape = (img_width, img_height, 3)

    inp = Input(shape=input_shape)
    s = Lambda(lambda x: x / 255) (inp)

    encodeds = []
    k_size_list = [(3,3), (3,3)]
    #k_size_list = [(3,3), (3,3)]

    # encoder
    enc = s
    print(n_ch_exps)
    for l_idx, n_ch in enumerate(n_ch_exps):
        for i in range(len(k_size_list)):
            k_size = k_size_list[i]
            do = 0.1
            enc = conv_layer(enc, 2**n_ch, k_size, 'relu', do, False, name='c{}_{}x{}_{}'.format((l_idx+1), 
                                                                                                        k_size[0], k_size[1], 
                                                                                                        (i+1)))
            #round(0.1*l_idx,1), 
        encodeds.append(enc)

        #print(l_idx, enc)
        if n_ch < n_ch_exps[-1]:  #do not run max pooling on the last encoding/downsampling step
            enc = MaxPooling2D(pool_size=(2,2))(enc)
    

    # decoder
    dec = enc
    print(n_ch_exps[::-1][1:])
    decoder_n_chs = n_ch_exps[::-1][1:]
    for l_idx, n_ch in enumerate(decoder_n_chs):
        l_idx_rev = len(n_ch_exps) - l_idx - 2  #
        
        k_size=(2,2)
        dec = UpSampling2D(size=k_size)(dec)
        dec = Conv2DTranspose(filters=2**n_ch, kernel_size=k_size, 
                              activation='relu', 
                              padding='same', 
                              kernel_initializer=k_init)(dec)
        
        dec = concatenate([dec, encodeds[l_idx_rev]], axis=ch_axis)
    
        #for i in range(len(k_size_list)):
        for i, k in reversed(list(enumerate(k_size_list))):    
            k_size = k_size_list[i]
            dec = conv_layer(dec, 2**n_ch, k_size, 'relu', do, False, 'dc{}_{}x{}_{}'.format((l_idx_rev+1), 
                                                                                                   k_size[0], k_size[1], (i+1)))
        
    #outp = Conv2DTranspose(filters=1, kernel_size=k_size, activation='sigmoid', padding='same', 
                           #kernel_initializer='glorot_normal')(dec)
    
    outp = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid', name='output')(dec)

    model = Model(inputs=[inp], outputs=[outp])
    
    return model


# *Update: Changed to ELU units, added dropout.*
# 
# Next we fit the model on the training data, using a validation split of 0.1. We use a small batch size because we have so little data. I recommend using checkpointing and early stopping when training your model. I won't do it here to make things a bit more reproducible (although it's very likely that your results will be different anyway). I'll just train for 10 epochs, which takes around 10 minutes in the Kaggle kernel with the current parameters. 
# 
# *Update: Added early stopping and checkpointing and increased to 30 epochs.*

# In[ ]:


model = keras_model_Unet_2D(img_width=IMG_WIDTH, img_height=IMG_HEIGHT)
model.summary()


# In[ ]:



# Perform a sanity check on some random training samples
for i in range(5):
    ix = random.randint(0, len(train_ids)-1)
    imshow(X_train_orig[ix])
    plt.show()

    imshow(np.squeeze(Y_train_orig[ix]))
    plt.show()


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])


# In[ ]:


#callbacks_list = []
modelName = 'UNet_2D_UCCC_33_aug_do0.1_orig.h5'
earlystopper = EarlyStopping(patience=10, verbose=1)
checkpointer = ModelCheckpoint(modelName, verbose=1, save_best_only=True)
results = model.fit(X_train_orig, Y_train_orig, validation_split=0.2, batch_size=32, epochs=1, #200,#50 
                    callbacks=[earlystopper, checkpointer])


# All right, looks good! Loss seems to be a bit erratic, though. I'll leave it to you to improve the model architecture and parameters! 
# 
# # Make predictions
# 
# Let's make predictions both on the test set, the val set and the train set (as a sanity check). Remember to load the best saved model if you've used early stopping and checkpointing.

# In[ ]:


print(modelName)


# In[ ]:


# Predict on train, val and test
#model = load_model(modelName, custom_objects={'dice_coef_loss':dice_coef_loss, 'dice_coef': dice_coef})
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
for i in range(5):
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
    lab_img = skimage.morphology.label(x > cutoff)
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
sub.to_csv('UNet_2D_UCCC_33_aug_do0.1_orig.csv', index=False)


# In[ ]:


sub.describe()


# This scored 0.347 on the LB for me. 
# 
# **Have fun!**
# 
# LB score history:
# - Version 10: 0.348 LB
