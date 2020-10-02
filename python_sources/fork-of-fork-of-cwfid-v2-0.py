#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
from sklearn import utils
from skimage.transform import rescale, resize
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import load_img
from keras.models import Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/dataset-master/dataset-master"))

# Any results you write to the current directory are saved as output.


# In[ ]:


def image_path(img_type, img_number):
    '''
    image_type: 'images' #original image, 'annotations' #crop-weed label, 'mask' #vegetation segmentation
    image_number: the number on the image name
    '''
    image_name = img_type[:-1]
    if img_number < 10:
        path = '../input/dataset-master/dataset-master/'+str(img_type)+'/00'+str(img_number)+'_'+str(image_name)+'.png'
    else:
        path = '../input/dataset-master/dataset-master/'+str(img_type)+'/0'+str(img_number)+'_'+str(image_name)+'.png'
    return path

def label_generator(number):
    annotation = cv2.imread(image_path('annotations', number))
    height = annotation.shape[0]
    width = annotation.shape[1]
   # channel = annotation.shape[2]
    labels = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            if np.all(annotation[i,j,:] == np.array([0,255,0])):
                labels[i,j,0] = 1
            elif np.all(annotation[i,j,:] == np.array([0,0,255])):
                labels[i,j,1] = 1
            elif np.all(annotation[i,j,:] == np.array([0,0,0])):
                labels[i,j,2] = 1
    return labels


# In[ ]:


image = cv2.imread(image_path('annotations',1))
image_rescaled = rescale(image, 1.0 / 6.0, anti_aliasing=True)
plt.imshow(image_rescaled)


# In[ ]:


# Load two images
img1 = cv2.imread(image_path('images',1))
img2 = cv2.imread(image_path('masks',1))
# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]
# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst
plt.imshow(img1)


# In[ ]:


label = label_generator(1)
plt.figure()
#plt.imshow(label)
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(label[:,:,i])


# In[ ]:


x_train = np.zeros((120, 161, 216, 3))
y_train = np.zeros((120, 161, 216, 3))
x_test = np.zeros((20, 161, 216, 3))
y_test = np.zeros((20, 161, 216, 3))

plt.figure(figsize=(8,5))

for i in range(40):
    image = cv2.imread(image_path('images',i+1))
    image_rescaled = rescale(image, 1.0 / 6.0, anti_aliasing=True)
    label = label_generator(i+1)
    label_rescaled = rescale(label, 1.0 / 6.0, anti_aliasing=True)
    x_train[i,:,:,:] = image_rescaled
    y_train[i,:,:,:] = label_rescaled
    x_train[40+i,:,:,:] = np.fliplr(image_rescaled)
    y_train[40+i,:,:,:] = np.fliplr(label_rescaled)
    x_train[80+1,:,:,:] = np.flipud(image_rescaled)
    y_train[80+i,:,:,:] = np.flipud(label_rescaled)
    plt.subplot(8,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i,:,:,:])
    
for i in range(20):
    image = cv2.imread(image_path('images',i+41))
    image_rescaled = rescale(image, 1.0 / 6.0, anti_aliasing=True)
    label = label_generator(i+41)
    label_rescaled = rescale(label, 1.0 / 6.0, anti_aliasing=True)
    x_test[i,:,:,:] = image_rescaled
    y_test[i,:,:,:] = label_rescaled
    


# In[ ]:


plt.figure()
#plt.imshow(y_train[i,:,:])
for i in range(40):
    plt.subplot(8,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y_train[i,:,:,2])


# In[ ]:


def depth_softmax(matrix):
    sigmoid = lambda x: 1 / (1 + K.exp(-x))
    sigmoided_matrix = sigmoid(matrix)
    softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
    return softmax_matrix

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

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

def weighted_binary_cross_entropy(y_true, y_pred):
    w = tf.reduce_sum(y_true)/tf.cast(tf.size(y_true), tf.float32)
    real_th = 0.5-(1.0/2.0)
    tf_th = tf.fill(tf.shape(y_pred), real_th) 
    tf_zeros = tf.fill(tf.shape(y_pred), 0.)
    return (1.0 - w) * y_true * - tf.log(tf.maximum(tf.zeros, tf.sigmoid(y_pred) + tf_th)) + (1- y_true) * w * -tf.log(1 - tf.maximum(tf_zeros, tf.sigmoid(y_pred) + tf_th))
#return weighted_binary_cross_entropy

def class_weighted_pixelwise_crossentropy(target, output):
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    weights = [0.8, 0.2]
    return -tf.reduce_sum(target * weights * tf.log(output))


# In[ ]:


def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x


# In[ ]:


def unet_res(n_classes = 1, start_neurons = 16, DropoutRatio = 0.5, img_height=161,img_width=216):
    # 101 -> 50
    input_layer = Input((img_height, img_width, 3))
    zero_pad = ZeroPadding2D(padding=((7,8),(4,4)))(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(zero_pad)
    conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = residual_block(conv1,start_neurons * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = residual_block(conv2,start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = residual_block(conv3,start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = residual_block(conv4,start_neurons * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 16)
    convm = residual_block(convm,start_neurons * 16, True)
    
    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)
    
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = residual_block(uconv4,start_neurons * 8, True)
    
    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(DropoutRatio)(uconv3)
    
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = residual_block(uconv3,start_neurons * 4, True)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = residual_block(uconv2,start_neurons * 2, True)
    
    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = residual_block(uconv1,start_neurons * 1, True)
    
    #uconv1 = Dropout(DropoutRatio/2)(uconv1)
    #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(n_classes, (1,1), padding="same", activation=None)(uconv1)
    output_layer = Cropping2D(cropping=((7,8),(4,4)))(output_layer_noActi)
    
    if n_classes == 1:
        output_layer = (Reshape((img_height, img_width),
                          input_shape=(img_height, img_width, 3)))(output_layer)
    else:
        output_layer = (Reshape((img_height, img_width, n_classes),
                          input_shape=(img_height, img_width, 3)))(output_layer)
    
    output_layer =  Activation('sigmoid')(output_layer)
    
    model = Model(input_layer, output_layer)
    
    return model


# In[ ]:


trial_model = unet_res(n_classes = 1, start_neurons=16)
trial_model.summary()


# In[ ]:


bg_model = unet_res(n_classes=1)
bg_model.compile(loss = 'binary_crossentropy',
             optimizer = 'Adam',
             metrics = [f1, mean_iou])

callbacks_bg = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(patience=3, verbose=1),
    ModelCheckpoint('model-bg-cwfid.h5', verbose=1, save_best_only=True)
]

model_bg_history = bg_model.fit(x = x_train,
                         y = y_train[:,:,:,2],
                         batch_size = 5,
                         epochs = 20,
                         verbose = 1,
                         validation_split = 0.3,
                          callbacks = callbacks_bg
                         )


# In[ ]:


weed_model = unet_res(n_classes=1)
weed_model.compile(loss = 'binary_crossentropy',
             optimizer = 'Adam',
             metrics = [f1, mean_iou])

callbacks_weed = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(patience=3, verbose=1),
    ModelCheckpoint('model-weed-cwfid.h5', verbose=1, save_best_only=True)
]

model_weed_history = weed_model.fit(x = x_train,
                         y = y_train[:,:,:,1],
                         batch_size = 5,
                         epochs = 20,
                         verbose = 1,
                         validation_split = 0.2,
                          callbacks = callbacks_weed
                         )


# In[ ]:


crop_model = unet_res(n_classes=1)
crop_model.compile(loss = 'binary_crossentropy',
             optimizer = 'Adam',
             metrics = [f1, mean_iou])

callbacks_crop = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(patience=3, verbose=1),
    ModelCheckpoint('model-crop-cwfid.h5', verbose=1, save_best_only=True)
]

model_crop_history = crop_model.fit(x = x_train,
                         y = y_train[:,:,:,0],
                         batch_size = 5,
                         epochs = 20,
                         verbose = 1,
                         validation_split = 0.3,
                          callbacks = callbacks_crop
                         )


# In[ ]:


pred_bg = bg_model.predict(x_test)
pred_weed = weed_model.predict(x_test)
pred_crop = crop_model.predict(x_test)


# In[ ]:


i = 15
plt.figure()
plt.subplot(2,3,1)
plt.imshow(pred_bg[i,:,:])
plt.subplot(2,3,4)
plt.imshow(y_test[i,:,:,2])
plt.subplot(2,3,2)
plt.imshow(pred_weed[i,:,:])
plt.subplot(2,3,5)
plt.imshow(y_test[i,:,:,1])
plt.subplot(2,3,3)
plt.imshow(pred_crop[i,:,:])
plt.subplot(2,3,6)
plt.imshow(y_test[i,:,:,0])


# In[ ]:


full_model = unet_res(n_classes=3)
full_model.compile(loss = 'binary_crossentropy',
             optimizer = 'Adam',
             metrics = [f1, mean_iou])

callbacks_full = [
    EarlyStopping(patience=8, verbose=1),
    ReduceLROnPlateau(patience=5, verbose=1),
    ModelCheckpoint('model-full-cwfid.h5', verbose=1, save_best_only=True)
]

model_full_history = full_model.fit(x = x_train,
                         y = y_train,
                         batch_size = 5,
                         epochs = 30,
                         verbose = 1,
                         validation_split = 0.2,
                          callbacks = callbacks_full
                         )


# In[ ]:


pred_full = full_model.predict(x_test)


# In[ ]:


i = 15
plt.figure()
plt.subplot(2,3,1)
plt.imshow(pred_full[i,:,:,2])
plt.subplot(2,3,4)
plt.imshow(y_test[i,:,:,2])
plt.subplot(2,3,2)
plt.imshow(pred_full[i,:,:,1])
plt.subplot(2,3,5)
plt.imshow(y_test[i,:,:,1])
plt.subplot(2,3,3)
plt.imshow(pred_full[i,:,:,0])
plt.subplot(2,3,6)
plt.imshow(y_test[i,:,:,0])


# In[ ]:





# In[ ]:




