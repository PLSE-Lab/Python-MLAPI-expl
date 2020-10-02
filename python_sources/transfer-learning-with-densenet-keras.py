#!/usr/bin/env python
# coding: utf-8

# # CNN Starter with DenseNet in Keras
# 
# This a fork of the preprocessing phase, described here: https://www.kaggle.com/xhlulu/exploration-and-preprocessing-for-keras-224x224
# 
# The resulting files are:
# * `X_train`: 25361x224x224x3
# * `X_test`: 7960x224x224x3
# * `y_train`: 25361x5005
# 
# Using those files, I will show how to perform data augmentation and transfer learning.

# In[ ]:


import os
import cv2
from PIL import Image, ImageOps
import math

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import time
import gc
from sklearn.model_selection import train_test_split


# In[ ]:


print(os.listdir("../input"))


# ## Exploration

# In[ ]:


label_df = pd.read_csv('../input/humpback-whale-identification/train.csv')
submission_df = pd.read_csv('../input/humpback-whale-identification/sample_submission.csv')
label_df.head()


# In[ ]:


label_df['Id'].describe()


# In[ ]:


# Display the most frequent ID (without counting new_whale)
label_df['Id'].value_counts()[1:16].plot(kind='bar')


# In[ ]:


def display_samples(df, columns=4, rows=3):
    fig=plt.figure(figsize=(5*columns, 3*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'Image']
        image_id = df.loc[i,'Id']
        img = cv2.imread(f'../input/humpback-whale-identification/train/{image_path}')
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)

display_samples(label_df)


# The width of the image seem to be bigger than the height. We will have to pad the images, then resize them to 224x224x3

# ## Preprocessing

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


def pad_and_resize(image_path, dataset):
    img = cv2.imread(f'../input/humpback-whale-identification/{dataset}/{image_path}')
    pad_width = get_pad_width(img, max(img.shape))
    padded = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)
    resized = cv2.resize(padded, (224,224)).astype('uint8')
    
    return resized


# ### Pad and resize with cv2

# In[ ]:


img = cv2.imread(f'../input/humpback-whale-identification/train/{label_df.loc[0,"Image"]}')

pad_width = get_pad_width(img, max(img.shape))
padded = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)
resized = cv2.resize(padded, (224,224))
plt.imshow(resized)


# ### Resizing

# In[ ]:


train_resized_imgs = []
test_resized_imgs = []

for image_path in label_df['Image']:
    train_resized_imgs.append(pad_and_resize(image_path, 'train'))

for image_path in submission_df['Image']:
    test_resized_imgs.append(pad_and_resize(image_path, 'test'))


# In[ ]:


X_train = np.stack(train_resized_imgs)
X_test = np.stack(test_resized_imgs)


# In[ ]:


target_dummies = pd.get_dummies(label_df['Id'])
train_label = target_dummies.columns.values
y_train = target_dummies.values


# In[ ]:


del train_resized_imgs, test_resized_imgs

gc.collect()
time.sleep(10)


# ### Preprocess input the same way keras does

# In[ ]:


from tensorflow.keras.applications.densenet import preprocess_input


# In[ ]:


X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)


# ## Prepare Data For Keras

# In[ ]:


import json

from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input, DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint


# In[ ]:


densenet = DenseNet121(
    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(224,224,3)
)

model = Sequential()
model.add(densenet)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(5005, activation='softmax'))
model.summary()


# In[ ]:


model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    'model.h5', 
    monitor='val_acc', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=False,
    mode='auto'
)

history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=64,
    epochs=10,
    callbacks=[checkpoint],
    verbose=2,
    validation_split=0.1
)


# In[ ]:


with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()


# In[ ]:


model.load_weights('model.h5')
submission_df['Id'] = model.predict(X_test)
submission_df.to_csv('submission.csv', index=None)

