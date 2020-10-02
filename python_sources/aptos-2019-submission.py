#!/usr/bin/env python
# coding: utf-8

# <h1>APTOS Diabetic Retinopathy Severity Prediction</h1>

# In[ ]:


get_ipython().system('pip install ../input/efficientnet/efficientnet-master/efficientnet-master')


# In[ ]:


import cv2
import numpy as np # linear algebra

import pandas as pd
import tensorflow as tf

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras import Input
from keras.models import Model
from keras.utils import *
from keras.layers import *

from skimage.transform import resize

from tensorflow import set_random_seed
import matplotlib.pyplot as plt

set_random_seed(2)
np.random.seed(0)

import os
import gc


# In[ ]:


from efficientnet.keras import *


# In[ ]:


print(os.listdir('../input/pretrained-weights'))


# In[ ]:


'''
    Preprocessing using Ben Graham's method (Last competition's winner) 
    https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping
'''
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): 
            return img 
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img
    
def load_ben_color(image, IMG_SIZE, sigmaX=10):
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image


# In[ ]:


'''
    Define model
'''
def output_relu(x):
    return K.relu(x, max_value=4)

def get_model(version, IMG_SIZE): 
    base_model = 0
    if version == 0:
        base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    elif version == 1:
        base_model = EfficientNetB1(weights=None, include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    elif version == 2:
        base_model = EfficientNetB2(weights=None, include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    elif version == 3:
        base_model = EfficientNetB3(weights=None, include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    elif version == 4:
        base_model = EfficientNetB4(weights=None, include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    else:
        return None 
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation=output_relu, kernel_initializer='he_normal')(x)
    model = Model(inputs=base_model.input, outputs=x)

    return model


# In[ ]:


model_b0_256 = get_model(0, 300)
model_b0_256.load_weights('../input/pretrained-weights/model_b0_finetuned_20_epochs.h5')

model_b1_240 = get_model(1, 300)
model_b1_240.load_weights('../input/pretrained-weights/model_b1.h5')

model_b3_300 = get_model(3, 300)
model_b3_300.load_weights('../input/pretrained-weights/model_b3_epochs_30_300.h5')


# In[ ]:


test_csv = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
id_code = test_csv['id_code']
prediction = np.empty(len(id_code), dtype='uint8')
for i in range(len(id_code)):
    img = cv2.imread('../input/aptos2019-blindness-detection/test_images/{}.png'.format(id_code[i]))
#     img240 = load_ben_color(img, 240).astype('float32') / 255.
#     img256 = load_ben_color(img, 256).astype('float32') / 255.
    img300 = load_ben_color(img, 300).astype('float32') / 255.
        
#     X1 = np.array([img256, cv2.flip(img256, 0), cv2.flip(img256, 1)])
#     X2 = np.array([img240, cv2.flip(img240, 0), cv2.flip(img240, 1)])
    X3 = np.array([img300])

    pred1 = model_b0_256.predict(X3)
    pred2 = model_b1_240.predict(X3)
    pred3 = model_b3_300.predict(X3)
    
    pred1 = np.mean(pred1)
    pred2 = np.mean(pred2)
    pred3 = np.mean(pred3)
    
    pred = (pred1 + pred2 + pred3) / 3
    pred = np.rint(pred).astype('uint8')
    print(pred)
    prediction[i] = pred


# In[ ]:


test_csv['diagnosis'] = prediction
test_csv.to_csv("submission.csv", index=False)
# print(test_csv)
unique, counts = np.unique(prediction, return_counts=True)
tmp = dict(zip(unique, counts))
print(tmp)

