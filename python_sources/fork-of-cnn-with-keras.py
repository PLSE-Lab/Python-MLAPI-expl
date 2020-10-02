#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


# ### Load dataset info

# In[ ]:


path_to_train = '../input/train/'
data = pd.read_csv('../input/train.csv')

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)


# ### Create datagenerator

# In[ ]:


class data_generator:
    
    def create_train(dataset_info, batch_size, shape, augument=True):
        assert shape[2] == 3
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, idx in enumerate(random_indexes):
                image = data_generator.load_image(
                    dataset_info[idx]['path'], shape)   
                if augument:
                    image = data_generator.augment(image)
                batch_images[i] = image
                batch_labels[i][dataset_info[idx]['labels']] = 1
            yield batch_images, batch_labels
            
    
    def load_image(path, shape):
        image_red_ch = skimage.io.imread(path+'_red.png')
        image_yellow_ch = skimage.io.imread(path+'_yellow.png')
        image_green_ch = skimage.io.imread(path+'_green.png')
        image_blue_ch = skimage.io.imread(path+'_blue.png')

        image_red_ch += (image_yellow_ch/2).astype(np.uint8) 
        image_green_ch += (image_yellow_ch/2).astype(np.uint8)

        image = np.stack((
            image_red_ch, 
            image_green_ch, 
            image_blue_ch), -1)
        image = resize(image, (shape[0], shape[1]), mode='reflect')
        return image
                
            
    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug


# 
# ### Show data

# In[ ]:


# create train datagen
train_datagen = data_generator.create_train(
    train_dataset_info, 5, (299,299,3), augument=True)


# In[ ]:


images, labels = next(train_datagen)

fig, ax = plt.subplots(1,5,figsize=(25,5))
for i in range(5):
    ax[i].imshow(images[i])
print('min: {0}, max: {1}'.format(images.min(), images.max()))


# ### Create model

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, Conv3D, MaxPooling3D, Reshape, BatchNormalization, MaxPooling2D
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import keras

def create_model(input_shape, n_out):
    
    '''pretrain_model = InceptionResNetV2(
        include_top=False, 
        weights='imagenet', 
        input_shape=input_shape)'''
    
    model = Sequential()
    model.add(Reshape((299,299,3,1), input_shape=input_shape))
    model.add(Conv3D(4, kernel_size=(4,4,2), strides=(1,1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2,2,1), strides=(2,2,1)))
    model.add(Dropout(0.5))
    model.add(Conv3D(8, kernel_size=(4,4,2), strides=(1,1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2,2,1), strides=(2,2,1)))
    model.add(Dropout(0.5))
    model.add(Conv3D(16, kernel_size=(4,4,1), strides=(1,1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2,2,1), strides=(2,2,1)))
    model.add(Dropout(0.5))
    model.add(Reshape((34,34,16)))
    model.add(Conv2D(32, kernel_size=(4,4), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, kernel_size=(4,4), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_out))
    model.add(Activation('sigmoid'))
    return model
#input_shape=(299,299,3)


# In[ ]:


from keras.callbacks import EarlyStopping, ModelCheckpoint
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


# In[ ]:


keras.backend.clear_session()

model = create_model(
    input_shape=(299,299,3), 
    n_out=28)

model.compile(
    loss='categorical_crossentropy', 
    optimizer=Adam(),
    metrics=['acc', f1])

model.summary()


# ### Train model

# In[ ]:


epochs = 100; batch_size = 32
checkpointer = ModelCheckpoint(
    './model.hdf5', 
    verbose=1, 
    save_best_only=True, monitor='val_f1', mode='max')

# split and suffle data 
np.random.seed(2018)
indexes = np.arange(train_dataset_info.shape[0])
np.random.shuffle(indexes)
train_indexes = indexes[:27500]
valid_indexes = indexes[27500:]

# create train and valid datagens
train_generator = data_generator.create_train(
    train_dataset_info[train_indexes], batch_size, (299,299,3), augument=True)
validation_generator = data_generator.create_train(
    train_dataset_info[valid_indexes], 100, (299,299,3), augument=False)

# train model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    validation_data=next(validation_generator),
    epochs=epochs, 
    verbose=1,
    callbacks=[checkpointer])


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('loss')
ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax[1].set_title('acc')
ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
ax[0].legend()
ax[1].legend()


# ### Create submit

# In[ ]:


submit = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


get_ipython().run_cell_magic('time', '', "predicted = []\nfor name in tqdm(submit['Id']):\n    path = os.path.join('../input/test/', name)\n    image = data_generator.load_image(path, (299,299,3))\n    score_predict = model.predict(image[np.newaxis])[0]\n    label_predict = np.arange(28)[score_predict>=0.5]\n    str_predict_label = ' '.join(str(l) for l in label_predict)\n    predicted.append(str_predict_label)")


# In[ ]:


submit['Predicted'] = predicted
submit.to_csv('submission.csv', index=False)

