#!/usr/bin/env python
# coding: utf-8

# # quickdraw-doodle-recognition
# 
# Hello, everyone. This competition is a level3 image classification. 
# 
# **Dataset's name:description(size, count of train data, count of test data)**  
# Level 1 : **MNIST** : pictures of 0~9, grey(28\*28, 60000, 10000)  
# Level 2 : **CIFAR-10** : pictures of 10 objects, rgb(32\*32, 50000, 10000)  
# Level 3 : **CIFAR-100** : pictures of 100 objects, rgb(32\*32, 50000, 10000)  
# Level 4 : **ImageNet** : pictures of 1000 objects, rgb(224\*224)
# (I set this level by my criteria.)
# 
# In image classification, generally, people go through the following steps:  
# 1. Check data  
#     1-1. Check data's size  
#     1-2. Draw image(when data is not an image)  
#     1-3. Check image's info  
# 2. Construct Model  
#     2-1. VGG, ResNet, GoogleNet, your own model etc...  
#     2-2. Set parameters  
#     2-3. optimizers, annealing  
# 3. Training  
# 4. Predict test data  
#     4-1. If you don't satisfy, go to step2.  
# 5. Make submissions  
# 
# Then let's start.

# In[ ]:


import os
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import cv2
import matplotlib.pyplot as plt
import datetime as dt
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input


# ## Hyper Parameter

# In[ ]:


DP_DIR = '../input/shuffle-csvs/'
INPUT_DIR = '../input/quickdraw-doodle-recognition/'
NCSVS = 100
NCATS = 340
BASE_SIZE = 256
size = 64
epochs = 30
batch_size = 100
start = dt.datetime.now()


# # 1. Check data
# 
# ## 1-1. Check data's size

# It's too bigger than memory given to us(140GB >> 13GB). So, we will use other data.  
# <https://www.kaggle.com/gaborfodor/shuffle-csvs>  
# In this kernel, gaborfodor makes shuffle-csvs. Each of csv files includes all kinds of pictures. A file is fully available within the memory given to us.  

# ## 1-2. Draw Image

# Draw image with line function in cv2 module.

# In[ ]:


def draw_img(lines):
    img = np.zeros((BASE_SIZE, BASE_SIZE))
    for line in lines:
        for i in range(len(line[0]) - 1):
            _ = cv2.line(img, (line[0][i], line[1][i]), (line[0][i + 1], line[1][i + 1]), 255, 6)
    return cv2.resize(img, (size, size))


# Make image randomly.

# In[ ]:


def image_gen(batchsize, cnt):
    while True:
        for k in np.random.permutation(cnt):
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(json.loads)
                x = np.zeros((len(df), size, size, 1))
                for i, lines in enumerate(df.drawing.values):
                    x[i, :, :, 0] = draw_img(lines)
                    
                x = preprocess_input(x).astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                
                yield x, y


# In[ ]:


def df_to_image(df):
    df['drawing'] = df['drawing'].apply(json.loads)
    x = np.zeros((len(df), size, size, 1))
    for i, lines in enumerate(df.drawing.values):
        x[i, :, :, 0] = draw_img(lines)
    x = preprocess_input(x).astype(np.float32)
    return x


# This code is based on [this kernel](https://www.kaggle.com/gaborfodor/greyscale-mobilenet-lb-0-892).

# In[ ]:


train_datagen = image_gen(batch_size, range(NCSVS - 1))


# ## Check image's info

# Make map with number and category's name.

# In[ ]:


files = sorted(os.listdir('../input/quickdraw-doodle-recognition/train_simplified/'), reverse=False, key=str.lower)
class_dict = {file[:-4].replace(" ", "_"): i for i, file in enumerate(files)}
classreverse_dict = {v: k for k, v in class_dict.items()}


# # 2. Construct Model

# In[ ]:


def CNN_model():
    model = MobileNet(input_shape=(size, size, 1), alpha=1., weights=None, classes=NCATS)
    
#     My Own Model
#     model = Sequential()

#     model.add(Conv2D(32,kernel_size=3,activation='relu',padding='same',input_shape=(size,size,1)))
#     model.add(BatchNormalization())
#     model.add(Conv2D(32,kernel_size=3,activation='relu', padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.4))

#     model.add(Conv2D(64,kernel_size=3,activation='relu', padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(64,kernel_size=3,activation='relu', padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.4))

#     model.add(Flatten())
#     model.add(Dense(2 * NCATS, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.4))
#     model.add(Dense(NCATS, activation='softmax'))

    model.summary()
    
    return model


# 

# In[ ]:


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


# Use Adam optimizers.

# In[ ]:


model = CNN_model()

model.compile(optimizer=Adam(lr=0.0024), loss='categorical_crossentropy', metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])


# ## annealing

# In[ ]:


callbacks = [ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)]


# # 3. Training

# Make validation data.

# In[ ]:


valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=34000)
x_valid = df_to_image(valid_df)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
print(x_valid.shape, y_valid.shape)
print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))


# In[ ]:


history = model.fit_generator(train_datagen, epochs = epochs, verbose = 1, 
                              validation_data=(x_valid, y_valid),
                              steps_per_epoch=x_valid.shape[0] // batch_size, callbacks=callbacks)


# # 4. Predict test data

# In[ ]:


test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))
test.head()


# In[ ]:


x_test = df_to_image(test)
print(test.shape, x_test.shape)
print('Test array memory {:.2f} GB'.format(x_test.nbytes / 1024.**3 ))


# In[ ]:


test_predictions = model.predict(x_test, batch_size=batch_size)


# # 5. Make submissions

# Select top3 category.

# In[ ]:


top3 = pd.DataFrame(np.argsort(-test_predictions, axis=1)[:, :3])
top3.head()


# Change number to category's name, and submit submissions.

# In[ ]:


word = top3.replace(classreverse_dict)
test['word'] = word[0] + ' ' + word[1] + ' ' + word[2]
submission = test[['key_id', 'word']]
submission.to_csv('submission.csv', index=False)
submission.head()


# In[ ]:


end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))

