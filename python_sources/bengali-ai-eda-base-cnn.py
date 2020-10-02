#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm.auto import tqdm
from glob import glob
import time, gc
import cv2

from tensorflow import keras
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import clone_model
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
from matplotlib import pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')
test_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')
class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')
sample_sub_df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')


# Image processing

# In[ ]:


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(df, size=64, pad=16):
    resized = {}
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    for i in tqdm(range(df.shape[0])):
        #somehow the original input is inverted
        img0 = 255 - df.loc[df.index[i]].values.reshape(137, 236).astype(np.uint8)
        #normalize each image by its max val
        img0 = (img0*(255.0/img0.max())).astype(np.uint8)
        
        _, thresh = cv2.threshold(img0,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ymin,ymax,xmin,xmax = bbox(thresh[5:-5,5:-5] > 80)
        #cropping may cut too much, so we need to add it back
        xmin = xmin - 13 if (xmin > 13) else 0
        ymin = ymin - 10 if (ymin > 10) else 0
        xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
        ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
        img = img0[ymin:ymax,xmin:xmax]
        #remove lo intensity pixels as noise
        img[img < 28] = 0
        lx, ly = xmax-xmin,ymax-ymin
        l = max(lx,ly) + pad
        #make sure that the aspect ratio is kept in rescaling
        img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
        resized_roi = cv2.resize(img, (size, size),interpolation=cv2.INTER_AREA)
        resized[df.index[i]] = resized_roi.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized


def get_dummies(df):
    cols = []
    for col in df:
        cols.append(pd.get_dummies(df[col].astype(str)))
    return pd.concat(cols, axis=1)

class MultiOutputDataGenerator(keras.preprocessing.image.ImageDataGenerator):

    def flow(self,
             x,
             y=None,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             subset=None):

        targets = None
        target_lengths = {}
        ordered_outputs = []
        for output, target in y.items():
            if targets is None:
                targets = target
            else:
                targets = np.concatenate((targets, target), axis=1)
            target_lengths[output] = target.shape[1]
            ordered_outputs.append(output)


        for flowx, flowy in super().flow(x, targets, batch_size=batch_size,
                                         shuffle=shuffle):
            target_dict = {}
            i = 0
            for output in ordered_outputs:
                target_length = target_lengths[output]
                target_dict[output] = flowy[:, i: i + target_length]
                i += target_length

            yield flowx, target_dict
            


# Build Model

# In[ ]:


def Basic_Model():
    inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 1))

    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1), kernel_initializer='glorot_normal')(inputs)
    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu',  kernel_initializer='glorot_normal',)(model)
#    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu',  kernel_initializer='glorot_normal',)(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu',  kernel_initializer='glorot_normal',)(model)
    model = Dropout(rate=0.3)(model)
    
    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu',  kernel_initializer='glorot_normal',)(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu',  kernel_initializer='glorot_normal',)(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu',  kernel_initializer='glorot_normal',)(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = Dropout(rate=0.3)(model)
    
#    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_initializer='glorot_normal',)(model)
#    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_initializer='glorot_normal',)(model)
#    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_initializer='glorot_normal',)(model)
#    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_initializer='glorot_normal',)(model)
#    model = BatchNormalization(momentum=0.15)(model)
#    model = MaxPool2D(pool_size=(2, 2))(model)
#    model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu', kernel_initializer='glorot_normal',)(model)
#    model = BatchNormalization(momentum=0.15)(model)
#    model = Dropout(rate=0.3)(model)
    
    model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu',  kernel_initializer='glorot_normal',)(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_initializer='glorot_normal', )(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=256, kernel_size=(5, 5), padding='SAME', activation='relu', kernel_initializer='glorot_normal',)(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = Dropout(rate=0.3)(model)
    
    #root model
    root_model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='elu',  kernel_initializer='glorot_normal',)(model)
    root_model = MaxPool2D(pool_size=(2, 2))(root_model)
    root_model = Dropout(rate=0.2)(root_model)

    root_model = Flatten()(root_model) 
    root_model = Dense(1024, activation = "elu")(root_model)
    root_model = BatchNormalization(momentum=0.15)(root_model)
    root_model = Dropout(rate=0.2)(root_model)
    root_model = Dense(512, activation = "elu")(root_model)
    root_model = BatchNormalization(momentum=0.15)(root_model)
    root_model = Dropout(rate=0.1)(root_model)
    root_dense = Dense(256, activation = "elu")(root_model)
        
    #others 
    model = Flatten()(model)
    model = Dense(1024, activation = "elu")(model)
    model = Dropout(rate=0.3)(model)
    dense = Dense(512, activation = "elu")(model)
    
    
    head_root = Dense(168, activation = 'softmax',name='dense_23')(root_dense)
    head_vowel = Dense(11, activation = 'softmax',name='dense_24')(dense)
    head_consonant = Dense(7, activation = 'softmax',name='dense_25')(dense)
    
    model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


# In[ ]:


# Set a learning rate annealer. Learning rate will be half after 3 epochs if accuracy is not increased
learning_rate_reduction_root = ReduceLROnPlateau(monitor='dense_23_accuracy', 
                                            patience=3, 
                                            verbose=1,
                                            factor=0.5, 
                                            min_lr=0.00001)
learning_rate_reduction_vowel = ReduceLROnPlateau(monitor='dense_24_accuracy', 
                                            patience=3, 
                                            verbose=1,
                                            factor=0.5, 
                                            min_lr=0.00001)
learning_rate_reduction_consonant = ReduceLROnPlateau(monitor='dense_25_accuracy', 
                                            patience=3, 
                                            verbose=1,
                                            factor=0.5, 
                                            min_lr=0.00001)


# Training loop

# In[ ]:


import datetime
IMG_SIZE=64
N_CHANNELS=1
HEIGHT = 137
WIDTH = 236
batch_size = 256
epochs = 28

histories = []
model = Basic_Model()
for i in range(4):
    start = datetime.datetime.now()
    train_df = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
    X_train = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic','grapheme'], axis=1)
    X_train = crop_resize(X_train)/255
    
    X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)
    
    Y_train_root = pd.get_dummies(train_df['grapheme_root']).values
    Y_train_vowel = pd.get_dummies(train_df['vowel_diacritic']).values
    Y_train_consonant = pd.get_dummies(train_df['consonant_diacritic']).values

    print(f'Training images: {X_train.shape}')
    print(f'Training labels root: {Y_train_root.shape}')
    print(f'Training labels vowel: {Y_train_vowel.shape}')
    print(f'Training labels consonants: {Y_train_consonant.shape}')

    # Divide the data into training and validation set
    x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(X_train, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.08, random_state=666)
    del train_df
    del X_train
    del Y_train_root, Y_train_vowel, Y_train_consonant

    # Data augmentation for creating more training data
    datagen = MultiOutputDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.15, # Randomly zoom image 
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


    # This will just calculate parameters required to augment the given data. This won't perform any augmentations
    datagen.fit(x_train)
    
    # Fit the model
    history = model.fit_generator(datagen.flow(x_train, {'dense_23': y_train_root, 'dense_24': y_train_vowel, 'dense_25': y_train_consonant}, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]), 
                              steps_per_epoch=x_train.shape[0] // batch_size, 
                              callbacks=[learning_rate_reduction_root, learning_rate_reduction_vowel, learning_rate_reduction_consonant])

    histories.append(history)
    end = datetime.datetime.now()
    print ('use time'+str(end-start))
    # Delete to reduce memory usage
    del x_train
    del x_test
    del y_train_root
    del y_test_root
    del y_train_vowel
    del y_test_vowel
    del y_train_consonant
    del y_test_consonant
    gc.collect()


# In[ ]:


preds_dict = {
    'grapheme_root': [],
    'vowel_diacritic': [],
    'consonant_diacritic': []
}

components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
target=[] # model predictions placeholder
row_id=[] # row_id place holder
for i in range(4):
    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 
    df_test_img.set_index('image_id', inplace=True)

    X_test = crop_resize(df_test_img)/255
    X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)
    
    preds = model.predict(X_test)

    for i, p in enumerate(preds_dict):
        preds_dict[p] = np.argmax(preds[i], axis=1)

    for k,id in enumerate(df_test_img.index.values):  
        for i,comp in enumerate(components):
            id_sample=id+'_'+comp
            row_id.append(id_sample)
            target.append(preds_dict[comp][k])
    del df_test_img
    del X_test
    gc.collect()

df_sample = pd.DataFrame(
    {
        'row_id': row_id,
        'target':target
    },
    columns = ['row_id','target'] 
)
df_sample.to_csv('submission.csv',index=False)


# In[ ]:




