#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import os, cv2, random, csv
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils


# In[ ]:


TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'

ROWS = 32
COLS = 32
CHANNELS = 1

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i%250 == 0: print('Processed {} of {}'.format(i, count))
    
    return data

train = prep_data(train_images)
test = prep_data(test_images)

print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))

labels = []
for i in train_images:
    if 'dog' in i:
        labels.append(1)
    else:
        labels.append(0)
        
train = train.reshape(-1, 32,32,1)
test = test.reshape(-1, 32,32,1)
X_train = train.astype('float32')
X_test = test.astype('float32')
X_train /= 255
X_test /= 255
Y_train=labels

X_valid = X_train[:5000,:,:,:]
Y_valid =   Y_train[:5000]
X_train = X_train[5001:25000,:,:,:]
Y_train  = Y_train[5001:25000]

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)


# In[ ]:


def show_cats_and_dogs(idx):
    cat = read_image(train_cats[idx])
    dog = read_image(train_dogs[idx])
    pair = np.concatenate((cat, dog), axis=1)
    plt.figure(figsize=(10,5))
    plt.imshow(pair)
    plt.show()
    
for idx in range(0,5):
    show_cats_and_dogs(idx)


# In[ ]:


optimizer = 'adam'
objective = 'binary_crossentropy'


def catdog():
    
    model = Sequential()

    model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=(ROWS, COLS, CHANNELS), activation='relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    #model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(1, 1)))
    
    #model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    #model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(1, 1)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return model


model = catdog()


# In[ ]:


nb_epoch = 8
batch_size = 128      
        
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_valid, Y_valid))
    
submission = model.predict_proba(X_test, verbose=1)
test_id = range(1,12501)
predictions_df = pd.DataFrame({'id': test_id, 'label': submission[:,0]})

predictions_df.to_csv("submission.csv", index=False)

