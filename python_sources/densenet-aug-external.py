#!/usr/bin/env python
# coding: utf-8

# reference: [APTOS 2019: DenseNet Keras Starter](https://www.kaggle.com/xhlulu/aptos-2019-densenet-keras-starter);
# reference : [Fast cropping, preprocessing and augmentation](https://www.kaggle.com/joorarkesteijn/fast-cropping-preprocessing-and-augmentation)

# In[ ]:


import json
import math
import os

import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import tensorflow as tf
from tqdm import tqdm
print(os.listdir("../input"))
print(os.listdir("../input/moredata/external/External"))
get_ipython().run_line_magic('matplotlib', 'inline')


# Set random seed for reproducibility.

# In[ ]:


np.random.seed(2019)
tf.set_random_seed(2019)


# # Loading & Exploration

# In[ ]:


train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
train_df_aug = pd.read_csv('../input/moredata/external/External/Label.csv')


print(train_df.shape)
print(test_df.shape)
print(train_df_aug.shape)
train_df.head()
train_df_aug.head()


# ### Displaying some Sample Images

# In[ ]:


def display_samples(df, columns=4, rows=3):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'id_code']
        image_id = df.loc[i,'diagnosis']
        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(train_df)


# # Resize Images & Data Augmentation
# 
# We will resize the images to 224x224, then create a single numpy array to hold the data.

# In[ ]:


def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        pad_width = ((t,b), (l,r))
    return pad_width

def preprocess_image(image_path, desired_size=224):
    im = Image.open(image_path)
    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    
    return im


# In[ ]:


IMAGE_SIZE = 224
def info_image(im):
    # Compute the center (cx, cy) and radius of the eye
    cy = im.shape[0]//2
    midline = im[cy,:]
    midline = np.where(midline>midline.mean()/3)[0]
    if len(midline)>im.shape[1]//2:
        x_start, x_end = np.min(midline), np.max(midline)
    else: # This actually rarely happens p~1/10000
        x_start, x_end = im.shape[1]//10, 9*im.shape[1]//10
    cx = (x_start + x_end)/2
    r = (x_end - x_start)/2
    return cx, cy, r
def resize_image(im, augmentation=True):
    # Crops, resizes and potentially augments the image to IMAGE_SIZE
    cx, cy, r = info_image(im)
    scaling = IMAGE_SIZE/(2*r)
    rotation = 0
    if augmentation:
        scaling *= 1 + 0.3 * (np.random.rand()-0.5)
        rotation = 360 * np.random.rand()
    M = cv2.getRotationMatrix2D((cx,cy), rotation, scaling)
    M[0,2] -= cx - IMAGE_SIZE/2
    M[1,2] -= cy - IMAGE_SIZE/2
    return cv2.warpAffine(im,M,(IMAGE_SIZE,IMAGE_SIZE)) # This is the most important line

[green channel](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/102613#latest-591316)
# In[ ]:


def toCLAHEgreen(img):  
    clipLimit=2.0 
    tileGridSize=(8, 8)  
    img = np.array(img)     
    green_channel = img[:, :, 1]    
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cla = clahe.apply(green_channel) 
    cla=clahe.apply(cla)
    return cla


# In[ ]:


N = train_df.shape[0]
M = train_df_aug.shape[0]

x_train = np.empty((2*N+2*M, 224, 224, 1), dtype=np.uint8)

for i, image_id in enumerate(tqdm(train_df['id_code'])):
    image_path=f'../input/aptos2019-blindness-detection/train_images/{image_id}.png'
    im = Image.open(image_path)
    im = toCLAHEgreen(im)
    #r,g,b = im.split()
    #im = cv2.cvtColor(im, im,dstCn = 1)
    #im = np.expand_dims(im, axis=2)
    x_train[i, :, :, :] = resize_image(im, augmentation=False)
    
    image_path=f'../input/aptos2019-blindness-detection/train_images/{image_id}.png'
    im = cv2.imread(image_path)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    x_train[N+i, :, :] = resize_image(toCLAHEgreen(im), augmentation=True)
    
new1_df = pd.concat([train_df,train_df],ignore_index=True)

for i, image_id in enumerate(tqdm(train_df_aug['id_code'])):
    image_path=f'../input/moredata/external/External/images/{image_id}.jpg'
    im = Image.open(image_path)
    im = im.resize((224,)*2, resample=Image.LANCZOS)
    im = toCLAHEgreen(im)
    x_train[2*N+i, :, :] = im.copy()
    
    
    image_path=f'../input/moredata/external/External/images/{image_id}.jpg'
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    x_train[2*N+M+i, :, :] = resize_image(toCLAHEgreen(im), augmentation=True)
    
new2_df = pd.concat([train_df_aug,train_df_aug],ignore_index=True)

new_train_df = pd.concat([new1_df,new2_df],ignore_index=True)
new_train_df.tail()


# In[ ]:


N = test_df.shape[0]
x_test = np.empty((N, 224, 224), dtype=np.uint8)

for i, image_id in enumerate(tqdm(test_df['id_code'])):
    image_path=f'../input/aptos2019-blindness-detection/test_images/{image_id}.png'
    im = Image.open(image_path)
    im = im.resize((224,)*2, resample=Image.LANCZOS)
    im = toCLAHEgreen(im)
    x_test[i, :, :] = im.copy()


# In[ ]:


y_train = pd.get_dummies(new_train_df['diagnosis']).values

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)


# In[ ]:


y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
y_train_multi[:, 4] = y_train[:, 4]

for i in range(3, -1, -1):
    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])

print("Original y_train:", y_train.sum(axis=0))
print("Multilabel version:", y_train_multi.sum(axis=0))


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train_multi, 
    test_size=0.15, 
    random_state=2019
)


# # Mixup & Data Generator
# 
# Please Note: Although I show how to construct Mixup, **it is currently unused**. Please see notice at the top of the kernel.

# In[ ]:


BATCH_SIZE = 32
x_train = 
data_generator = ImageDataGenerator().flow(x_train, y_train, batch_size=BATCH_SIZE)


# ### Creating keras callback for QWK

# In[ ]:


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return


# # Model: DenseNet-121

# In[ ]:


densenet = DenseNet121(
    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(224,224,3)
)


# In[ ]:


def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )
    
    return model


# In[ ]:


model = build_model()
model.summary()


# # Training & Evaluation

# In[ ]:


kappa_metrics = Metrics()

history = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=15,
    validation_data=(x_val, y_val),
    callbacks=[kappa_metrics]
)


# ## Find best threshold
# 
# Please Note: Although I show how to construct a threshold optimizer, **it is currently unused**. Please see notice at the top of the kernel.

# In[ ]:


model.load_weights('model.h5')
y_val_pred = model.predict(x_val)

def compute_score_inv(threshold):
    y1 = y_val_pred > threshold
    y1 = y1.astype(int).sum(axis=1) - 1
    y2 = y_val.sum(axis=1) - 1
    score = cohen_kappa_score(y1, y2, weights='quadratic')
    
    return 1 - score

simplex = scipy.optimize.minimize(
    compute_score_inv, 0.5, method='nelder-mead'
)

best_threshold = simplex['x'][0]


# ## Submit

# In[ ]:


y_test = model.predict(x_test) > 0.5
y_test = y_test.astype(int).sum(axis=1) - 1

test_df['diagnosis'] = y_test
test_df.to_csv('submission.csv',index=False)

