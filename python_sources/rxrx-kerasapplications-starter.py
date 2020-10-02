#!/usr/bin/env python
# coding: utf-8

# # Starter Code for Training a Model using Keras Application Pretrained Models.
# 
# Note: This does not currently produce good results for RCIC data. However it does work well for MNIST.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirs in os.listdir('../input'):
    print(dirs)
if 'recursion-cellular-image-classification' in os.listdir('../input'):
    DATA_DIR = '../input/recursion-cellular-image-classification'
else:
    DATA_DIR = '../input'

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
import sys

try:
    get_ipython().system('git clone https://github.com/recursionpharma/rxrx1-utils')
except:
    pass

sys.path.append('rxrx1-utils')
import rxrx.io as rio

from matplotlib import pyplot as plt
import keras


# # Load in data

# In[ ]:


# Load train data
train_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
# I want a complete list of images that we have - so I will duplicate evert entry in train_data for two sites
site_list = []
for id_code in train_data.id_code.unique():
    site_list.append([id_code, 1])
    site_list.append([id_code, 2])
# Create dataframe of id codes and the two sites
site_df = pd.DataFrame(site_list, columns=['id_code', 'site'])    
# Merge that site df into the train_data df.
train_data = pd.merge(train_data, site_df, how='left', on='id_code')

# Create a column that has the relative path to an image
train_data['rel_path'] = train_data.apply(lambda x: os.path.join(
            x.experiment, 
            'Plate{}'.format(x.plate),
            '{}_s{}_w'.format(x.well, x.site))+'{channel}.png', axis=1)

display(train_data.head())


# # Configs

# In[ ]:


IMAGE_SHAPE = (512, 512, 3)
NUM_CLASSES = 1108


# # CV Split
# I will just arbitrarily split on the first of each experiment

# In[ ]:


val_experiments = ["HEPG2-01", "HUVEC-01", "RPE-01", "U2OS-01"]
cv_partition = {}
cv_partition['train'] = []
cv_partition['validate'] = []
cv_partition['train'].append(np.array(train_data[~train_data.experiment.isin(val_experiments)].index))
cv_partition['validate'].append(np.array(train_data[train_data.experiment.isin(val_experiments)].index))


# # Data Generator

# In[ ]:


class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, img_paths, labels, img_rootdir, batch_size, mode='train', img_shappe=(512, 512, 3), n_classes=1108 ):
        self.img_paths = img_paths
        self.labels = labels
        self.img_rootdir = img_rootdir
        self.batch_size = 8
        self.mode = mode
        self.img_shape = img_shappe
        self.n_classes = n_classes
        
        # Create the indicies into the ID's and labels
        self.indexes = np.arange(len(self.img_paths))
        
    def __len__(self):
        'return the length of teh data generator (ie now many batches will be returned)'
        return int(np.floor(len(self.img_paths) / self.batch_size))
    
    def __getitem__(self, batch_idx):
        ' Generate one batch of data and return it'

        # Get the indexes of this batch
        batch_indexes = self.indexes[batch_idx*self.batch_size : (batch_idx+1)*self.batch_size]
        
        # Pre allocate
        X = np.zeros((self.batch_size, *self.img_shape))
        if self.mode == 'train':
            #y = np.zeros((self.batch_size, self.config.NUM_CLASSES), dtype=np.float32)
            y = np.zeros(self.batch_size, dtype=int)
        
        # Populate data
        for batch_idx, img_idx in enumerate(batch_indexes):
            X[batch_idx] = self.load_image(self.img_paths[img_idx])
            if self.mode == 'train':
                y[batch_idx] = self.labels[img_idx]
        
        # One-hot encode the labels
        if self.mode == 'train':
            y_onehot = keras.utils.to_categorical(
                    y,
                    num_classes=self.n_classes,
                    dtype='float32'
                )
            
        if self.mode == 'train':
            return X, y_onehot
        else:
            return X
    
    def load_image(self, img_path):
        channel_paths = []
        for channel in range(6):
            channel_paths.append(os.path.join(self.img_rootdir, img_path.format(channel=channel+1)))
        img = rio.load_images_as_tensor(channel_paths)
        img = rio.convert_tensor_to_rgb(img).astype(np.float32)
        img -= np.mean(img, axis=(0, 1))
        img /= 127.5

        return img

train_datagen = DataGenerator(
            img_paths = train_data.rel_path.values[cv_partition['train'][0]],
            labels = train_data.sirna.values[cv_partition['train'][0]],
            img_rootdir = TRAIN_DIR,
            batch_size = 8)
val_datagen = DataGenerator(
            img_paths = train_data.rel_path.values[cv_partition['validate'][0]],
            labels = train_data.sirna.values[cv_partition['validate'][0]],
            img_rootdir = TRAIN_DIR,
            batch_size = 8)

#X, y = train_datagen[0]
#img = X[0] #load_image(train_data.at[1, 'rel_path'])
#plt.imshow(img)


# # Create a Model

# In[ ]:


from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, Dropout, GlobalMaxPooling2D
from keras.models import Model

base_model = ResNet50(
                input_tensor=Input(shape=IMAGE_SHAPE, name='Input_Tensor'),
                pooling=None,
                weights='imagenet',
                include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(4096, activation='relu', name='head2')(x)
x = Dropout(0.25)(x)
x = Dense(2048, activation='relu', name='head3')(x)
x = Dropout(0.25)(x)
head = Dense(NUM_CLASSES, activation='softmax', name='output')(x)
model = Model(inputs=base_model.input, outputs=head)
#model.summary()


# In[ ]:


model.compile(  loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(lr=0.001),
                metrics=['categorical_accuracy'])


# In[ ]:


import multiprocessing as mp
print('workeres = {}'.format(mp.cpu_count()))
model_history = model.fit_generator(generator=train_datagen,
                                   validation_data = val_datagen,
                                   validation_steps=None,
                                   epochs=10,
                                   steps_per_epoch=100,
                                   use_multiprocessing=True,
                                   max_queue_size=10,
                                   workers=mp.cpu_count())


# In[ ]:


fig, ax = plt.subplots(2, 1, figsize=(10, 20))
print(model_history.history.keys())
# Plot model loss
ax[0].plot(model_history.history['loss'])
ax[0].plot(model_history.history['val_loss'])
ax[0].set_title('model loss')
ax[0].set_ylabel('loss')
ax[0].set_xlabel('epoch')
ax[0].legend(['train', 'test'], loc='upper left')

# Plot model accuracy
ax[1].plot(model_history.history['categorical_accuracy'])
ax[1].plot(model_history.history['val_categorical_accuracy'])
ax[1].set_title('model accuracy')
ax[1].set_ylabel('accuracy')
ax[1].set_xlabel('epoch')
ax[1].legend(['train', 'test'], loc='upper left')


# In[ ]:


get_ipython().system('rm -r rxrx1-utils')


# In[ ]:




