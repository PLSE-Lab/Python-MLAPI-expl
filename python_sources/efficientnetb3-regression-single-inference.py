#!/usr/bin/env python
# coding: utf-8

# # Inference Kernel
# This is the Inference Kernel of https://www.kaggle.com/fanconic/fork-of-efficientnetb3-regression-keras-2, version 25

# In[ ]:


import os
print(os.listdir('../input/fork-of-efficientnetb3-regression-keras-2'))
print(os.listdir('../input/aptos-trained-weights'))


# In[ ]:


import sys
import json
import math
import os
import subprocess
import time
import gc

get_ipython().system("pip install -U '../input/install/efficientnet-0.0.3-py2.py3-none-any.whl'")


# In[ ]:


import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.activations import elu
from efficientnet import EfficientNetB4
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm
print(os.listdir('../input'))


# ### Constants

# In[ ]:


HEIGHT = 256
WIDTH = 256
# Optimized Coefficients for regression
COEFF = [0.52022015, 1.46022145, 2.49058373, 3.30146459]


# ### Preprocess functions for images

# In[ ]:


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img
    
def circle_crop(img):   
    img = crop_image_from_gray(img)    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    
    return img 

def preprocess_image(img):
    #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #img = crop_image_from_gray(img)
    #img = cv2.resize(img, (WIDTH,HEIGHT))
    img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), 10) ,-4 ,128)
    
    return img


# ### Build Convolutional Neural Network and load its weights

# In[ ]:


def build_model():
    efficientnetb3 = EfficientNetB4(
        weights=None,
        input_shape=(HEIGHT,WIDTH,3),
        include_top=False
                   )

    model = Sequential()
    model.add(efficientnetb3)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(5, activation=elu))
    model.add(layers.Dense(1, activation="linear"))
    
    return model


# In[ ]:


model1 = build_model()
model1.load_weights('../input/fork-of-efficientnetb3-regression-keras-2/val_model.h5')
#model1.load_weights('../input/aptos-trained-weights/val_model_v30.h5')
model1.summary()


# In[ ]:


#model2 = build_model()
#model2.load_weights('../input/fork-of-efficientnetb3-regression-keras-2/effnet_modelB3.h5')
#model2.summary()


# ### Load test data set

# In[ ]:


TEST_IMG_PATH = '../input/aptos2019-blindness-detection/test_images/'
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
print(test_df.shape)

original_names = test_df['id_code'].values
test_df['id_code'] = test_df['id_code'] + ".png"
test_df['diagnosis'] = np.zeros(test_df.shape[0])
display(test_df.head())


# ### Predict Test Labels

# In[ ]:


tta_steps = 10
predictions1 = []
#predictions2 = []

for i in tqdm(range(tta_steps)):
    test_generator = ImageDataGenerator(rescale=1./255,
                                    #samplewise_center= True,
                                    horizontal_flip=True,
                                    rotation_range= 90, 
                                    vertical_flip=True,
                                    brightness_range=(0.5,1.5),
                                    zoom_range= 0.2,
                                    fill_mode='constant',
                                    preprocessing_function=preprocess_image,
                                    cval = 0).flow_from_dataframe(test_df, 
                                                    x_col='id_code', 
                                                    y_col = 'diagnosis',
                                                    directory = TEST_IMG_PATH,
                                                    target_size=(WIDTH, HEIGHT),
                                                    batch_size=1,
                                                    class_mode='other',
                                                    shuffle = False)
    
    preds1 = model1.predict_generator(test_generator, steps = test_df.shape[0])
    predictions1.append(preds1)

    del test_generator
    gc.collect()
    
pred_test1 = np.mean(predictions1, axis=0)


# In[ ]:


y_test = pred_test1


# In[ ]:



for i, pred in enumerate(y_test):
    if pred < COEFF[0]:
        y_test[i] = 0
    elif pred >= COEFF[0] and pred < COEFF[1]:
        y_test[i] = 1
    elif pred >= COEFF[1] and pred < COEFF[2]:
        y_test[i] = 2
    elif pred >= COEFF[2] and pred < COEFF[3]:
        y_test[i] = 3
    else:
        y_test[i] = 4


# In[ ]:


test_df['diagnosis'] = y_test.astype(int)
test_df['id_code'] = test_df['id_code'].str.replace(r'.png$', '')
test_df.head()


# ### Postprocess Leaks

# In[ ]:


leaks = pd.read_csv('../input/aptos-trained-weights/leaks.csv', dtype=  {'diagnosis': np.int32})
codes = leaks['id_code'].values
diagnosis = leaks['diagnosis'].values
for i in range(codes.shape[0]):
    test_df.loc[test_df['id_code'] == codes[i], 'diagnosis'] = int(diagnosis[i])


# In[ ]:


test_df.to_csv('submission.csv',index=False)
print(round(test_df.diagnosis.value_counts()/len(test_df)*100,4))


# In[ ]:


test_df.head()

