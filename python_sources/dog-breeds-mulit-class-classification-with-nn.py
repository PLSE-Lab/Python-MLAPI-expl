#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Input, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Constants and Helper Functions

# In[ ]:


data_dir = '../input/dog-breed-identification'


# In[ ]:


def plot_results(h):
    plt.plot(h['loss'], 'blue', label='Train Loss')
    plt.plot(h['val_loss'], 'orange', label='Val. Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


# In[ ]:


df = pd.read_csv(data_dir + '/labels.csv')
df['filename'] = df.id + '.jpg'

# top_dogs = df['breed'].value_counts()
# top_dogs = top_dogs[:16].index # top N
# df = df[df['breed'].isin(top_dogs)].reset_index(drop=True)
print(df.shape)
print(df.columns)


# In[ ]:


# Constants
input_shape = (224,224,3)
num_classes = len(set(df.breed.values))


# In[ ]:


def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_dataframe(
        df,
        directory=data_dir + '/train/train',
        x_col='filename',
        y_col='breed',
        target_size=(input_shape[0], input_shape[1]),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        seed=42,
        subset='training'
    )
    valid_generator = train_datagen.flow_from_dataframe(
        df,
        directory=data_dir + '/train/train',
        x_col='filename',
        y_col='breed',
        target_size=(input_shape[0], input_shape[1]),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        seed=42,
        subset='validation'
    )
    
    return train_generator, valid_generator


# In[ ]:


train_generator, valid_generator = get_data_generators()

for s in sorted(train_generator.class_indices):
    if train_generator.class_indices[s] != valid_generator.class_indices[s]:
        raise Exception(f'MisMatch: {train_generator.class_indices[s]} {valid_generator.class_indices[s]}')

unique_train_classes = set(train_generator.classes)
unique_valid_classes = set(valid_generator.classes)

if len(unique_train_classes) != num_classes or len(unique_valid_classes) != num_classes:
    raise Exception('Train and Valid do not contain all classes.')


# In[ ]:


def get_model():
    pretrained_model = Xception(weights='../input/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                include_top=False,
                                input_shape=input_shape)
    for l in pretrained_model.layers:
        l.trainable = False
    
    X_input = Input(shape=input_shape)
    X = pretrained_model(X_input)
    X = Flatten()(X)
    X = Dense(4005, activation='relu')(X)
#     X = Dense(2048, activation='relu')(X)
#     X = Dropout(0.5)(X)
    out = Dense(num_classes, activation='softmax')(X)
    return Model(X_input, [out])


# In[ ]:


model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop_callback = EarlyStopping(monitor='val_loss',
                                    verbose=1,
                                    patience=7,
                                    min_delta=1e-3)
checkpoint_callback = ModelCheckpoint('./best-model.h5', save_best_only=True)


# In[ ]:


history = model.fit_generator(
    train_generator,
    epochs=100,
    callbacks=[
        early_stop_callback,
        checkpoint_callback
    ],
    validation_data=valid_generator,
    verbose=1,
    shuffle=True,
)


# In[ ]:


all_history = {
    'loss': [],
    'val_loss': []
}
all_history['loss'] = all_history['loss'] + history.history['loss']
all_history['val_loss'] = all_history['val_loss'] + history.history['val_loss']


# In[ ]:


plot_results(all_history)


# # FastAI Version - this is 10x better

# In[ ]:


from fastai import *
from fastai.vision import *

# Move trained resnet so pytorch can find it
# Copy training images to a new folder for fastai
get_ipython().system('cp ../input/resnet-from-fastai/resnet34-333f7ec4.pth /tmp/.torch/models/resnet34-333f7ec4.pth')
get_ipython().system('cp -r ../input/dog-breed-identification/train ../train-jpg')


# In[ ]:


bs = 64


# In[ ]:


data = ImageDataBunch.from_df(path='../train-jpg',
                              df=df, 
                              fn_col='filename',
                              label_col='breed',
                              folder='train',
                              ds_tfms=get_transforms(), 
                              size=224, 
                              bs=bs, 
                              num_workers=0)
data = data.normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=3)


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


learn.load('stage-1')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-5, 1e-4))


# In[ ]:




