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


from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import keras


def conv_block(x, nb_filter, nb_row, nb_col, padding = "same", strides = (1, 1), use_bias = False):
    '''Defining a Convolution block that will be used throughout the network.'''
    
    x = Conv2D(nb_filter, (nb_row, nb_col), strides = strides, padding = padding, use_bias = use_bias)(x)
    x = BatchNormalization(axis = -1, momentum = 0.9997, scale = False)(x)
    x = Activation("relu")(x)
    
    return x

def stem(input):
    '''The stem of the pure Inception-v4 and Inception-ResNet-v2 networks. This is input part of those networks.'''
    
    # Input shape is 299 * 299 * 3 (Tensorflow dimension ordering)
    x = conv_block(input, 32, 3, 3, strides = (2, 2), padding = "same") # 149 * 149 * 32
    x = conv_block(x, 32, 3, 3, padding = "same") # 147 * 147 * 32
    x = conv_block(x, 64, 3, 3) # 147 * 147 * 64

    x1 = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(x)
    x2 = conv_block(x, 96, 3, 3, strides = (2, 2), padding = "same")

    x = concatenate([x1, x2], axis = -1) # 73 * 73 * 160

    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, padding = "same")

    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 96, 3, 3, padding = "same")

    x = concatenate([x1, x2], axis = -1) # 71 * 71 * 192

    x1 = conv_block(x, 192, 3, 3, strides = (2, 2), padding = "same")
    
    x2 = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(x)

    x = concatenate([x1, x2], axis = -1) # 35 * 35 * 384
    
    return x

def inception_A(input):
    '''Architecture of Inception_A block which is a 35 * 35 grid module.'''
    
    a1 = AveragePooling2D((3, 3), strides = (1, 1), padding = "same")(input)
    a1 = conv_block(a1, 96, 1, 1)
    
    a2 = conv_block(input, 96, 1, 1)
    
    a3 = conv_block(input, 64, 1, 1)
    a3 = conv_block(a3, 96, 3, 3)
    
    a4 = conv_block(input, 64, 1, 1)
    a4 = conv_block(a4, 96, 3, 3)
    a4 = conv_block(a4, 96, 3, 3)
    
    merged = concatenate([a1, a2, a3, a4], axis = -1)
    
    return merged

def inception_B(input):
    '''Architecture of Inception_B block which is a 17 * 17 grid module.'''
    
    b1 = AveragePooling2D((3, 3), strides = (1, 1), padding = "same")(input)
    b1 = conv_block(b1, 128, 1, 1)
    
    b2 = conv_block(input, 384, 1, 1)
    
    b3 = conv_block(input, 192, 1, 1)
    b3 = conv_block(b3, 224, 1, 7)
    b3 = conv_block(b3, 256, 7, 1)
    
    b4 = conv_block(input, 192, 1, 1)
    b4 = conv_block(b4, 192, 7, 1)
    b4 = conv_block(b4, 224, 1, 7)
    b4 = conv_block(b4, 224, 7, 1)
    b4 = conv_block(b4, 256, 1, 7)
    
    merged = concatenate([b1, b2, b3, b4], axis = -1)
    
    return merged

def inception_C(input):
    '''Architecture of Inception_C block which is a 8 * 8 grid module.'''
    
    c1 = AveragePooling2D((3, 3), strides = (1, 1), padding = "same")(input)
    c1 = conv_block(c1, 256, 1, 1)
    
    c2 = conv_block(input, 256, 1, 1)

    c3 = conv_block(input, 384, 1, 1)
    c31 = conv_block(c2, 256, 1, 3)
    c32 = conv_block(c2, 256, 3, 1)
    c3 = concatenate([c31, c32], axis = -1)

    c4 = conv_block(input, 384, 1, 1)
    c4 = conv_block(c3, 448, 3, 1)
    c4 = conv_block(c3, 512, 1, 3)
    c41 = conv_block(c3, 256, 1, 3)
    c42 = conv_block(c3, 256, 3, 1)
    c4 = concatenate([c41, c42], axis = -1)
  
    merged = concatenate([c1, c2, c3, c4], axis = -1)
    
    return merged

def reduction_A(input, k = 192, l = 224, m = 256, n = 384):
    '''Architecture of a 35 * 35 to 17 * 17 Reduction_A block.'''

    ra1 = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(input)
    
    ra2 = conv_block(input, n, 3, 3, strides = (2, 2), padding = "same")

    ra3 = conv_block(input, k, 1, 1)
    ra3 = conv_block(ra3, l, 3, 3)
    ra3 = conv_block(ra3, m, 3, 3, strides = (2, 2), padding = "same")

    merged = concatenate([ra1, ra2, ra3], axis = -1)
    
    return merged

def reduction_B(input):
    '''Architecture of a 17 * 17 to 8 * 8 Reduction_B block.'''
    
    rb1 = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(input)
    
    rb2 = conv_block(input, 192, 1, 1)
    rb2 = conv_block(rb2, 192, 3, 3, strides = (2, 2), padding = "same")
    
    rb3 = conv_block(input, 256, 1, 1)
    rb3 = conv_block(rb3, 256, 1, 7)
    rb3 = conv_block(rb3, 320, 7, 1)
    rb3 = conv_block(rb3, 320, 3, 3, strides = (2, 2), padding = "same")
    
    merged = concatenate([rb1, rb2, rb3], axis = -1)
    
    return merged

def inception_v4(nb_classes = 1001, load_weights = True):
    '''Creates the Inception_v4 network.'''
    
    init = Input((299, 299, 3)) # Channels last, as using Tensorflow backend with Tensorflow image dimension ordering
    
    # Input shape is 299 * 299 * 3
    x = stem(init) # Output: 35 * 35 * 384
    
    # 4 x Inception A
    for i in range(4):
        x = inception_A(x)
        # Output: 35 * 35 * 384
        
    # Reduction A
    x = reduction_A(x, k = 192, l = 224, m = 256, n = 384) # Output: 17 * 17 * 1024

    # 7 x Inception B
    for i in range(7):
        x = inception_B(x)
        # Output: 17 * 17 * 1024
        
    # Reduction B
    x = reduction_B(x) # Output: 8 * 8 * 1536

    # 3 x Inception C
    for i in range(3):
        x = inception_C(x) 
        # Output: 8 * 8 * 1536
        
    # Average Pooling
    x = AveragePooling2D((8, 8))(x) # Output: 1536

    # Dropout
    x = Dropout(0.2)(x) # Keep dropout 0.2 as mentioned in the paper
    x = Flatten()(x) # Output: 1536

    # Output layer
    output = Dense(units = nb_classes, activation = "softmax")(x) # Output: 1000

    model = Model(init, output, name = "Inception-v4")   
        
    return model


# In[ ]:


# keras.backend.clear_session()
model=inception_v4(nb_classes = 28, load_weights = True)
#model = create_model(input_shape=(299,299,3), n_out=28)
model.compile(loss='binary_crossentropy', optimizer=Adam(1e-04),metrics=['acc'])
model.summary()


# ### Train model

# In[ ]:


epochs = 100; batch_size = 16
checkpointer = ModelCheckpoint(
    '../working/InceptionV4.h5', 
    verbose=2, 
    save_best_only=True)

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

