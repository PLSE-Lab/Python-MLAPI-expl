#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import Image
Image("../input/Dance_Robots_Comic.jpg")


# (This is part 2 of 3 of my How to Teach an AI to Dance. I originally made 3 separate notebooks for this task before compiling them into one later. The complete assembled notebook of all 3 parts can be found here: https://www.kaggle.com/valkling/how-to-teach-an-ai-to-dance)
# 
# # AI Dance Part 2: Autoencoder Compression
# 
# Now that we have the preprocessed frames from the shadow dancer video, we will still need to compress them much further to fit them into our RNN model. Among the many uses of autoencoders is making specialized compression models. In this section, we will train an autoencoder on our dance images and use it to compress the images into a much smaller numpy array, saving the model so that we can decode the images later.

# In[ ]:


import os
import sys
import random
import warnings
from pylab import imshow, show, get_cmap

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
import skimage
from PIL import Image
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.util import crop, pad
from skimage.morphology import label

from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, UpSampling2D, Flatten, Reshape
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras import backend as K
import tensorflow as tf

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed


# ## Read in Images

# In[ ]:


# Set some parameters
IMG_WIDTH = 96
IMG_HEIGHT = 64
IMG_CHANNELS = 1
INPUT_SHAPE=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
TRAIN_PATH = '../input/dancer_images/data/'


# In[ ]:


train_ids = next(os.walk(TRAIN_PATH))[2]
train_ids[:10]


# In[ ]:


Image.open(TRAIN_PATH + 'frame5.jpg')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nX_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=\'float32\')\nfinal_train_ids = []\nmissing_count = 0\nprint(\'Getting train images ... \')\nsys.stdout.flush()\nfor n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):\n    path = TRAIN_PATH +\'frame\'+ str(n+1) + \'.jpg\'\n    try:\n        img = imread(path)\n        img = img.astype(\'float32\') / 255.\n        img = resize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode=\'constant\', preserve_range=True)\n        X_train[n-missing_count] = img\n    except:\n        print(" Problem with: "+path)\n        missing_count += 1\n        \nprint("Total missing: "+ str(missing_count))')


# In[ ]:


for n in range(10,20):
    imshow(X_train[n].reshape(64,96))
    plt.show()


# ## Create the Model
# 
# In addition to the Autoencoder model, we will also prepare an encoder and decoder for later. It is important to give the layers the same unique names and shapes in all 3 as we will be using the keras load_weights by_name option to copy our trained Autoencoder weights to each respective layer later.

# In[ ]:


def Encoder():
    inp = Input(shape=INPUT_SHAPE)
    x = Conv2D(128, (4, 4), activation='elu', padding='same',name='encode1')(inp)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',name='encode2')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',name='encode3')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same',name='encode4')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',name='encode5')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same',name='encode6')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',name='encode7')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same',name='encode8')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same',name='encode9')(x)
    x = Flatten()(x)
    x = Dense(256, activation='elu',name='encode10')(x)
    encoded = Dense(128, activation='sigmoid',name='encode11')(x)
    return Model(inp, encoded)

encoder = Encoder()
encoder.summary()


# In[ ]:


D_INPUT_SHAPE=[128]
def Decoder():
    inp = Input(shape=D_INPUT_SHAPE, name='decoder')
    x = Dense(256, activation='elu', name='decode1')(inp)
    x = Dense(768, activation='elu', name='decode2')(x)
    x = Reshape((4, 6, 32))(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same', name='decode3')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='decode4')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same', name='decode5')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='decode6')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (2, 2), activation='elu', padding='same', name='decode7')(x)
    x = Conv2D(128, (3, 3), activation='elu', padding='same', name='decode8')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (2, 2), activation='elu', padding='same', name='decode9')(x)
    x = Conv2D(64, (4, 4), activation='elu', padding='same', name='decode10')(x)
    x = Conv2D(128, (3, 3), activation='elu', padding='same', name='decode11')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (4, 4), activation='elu', padding='same', name='decode12')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', name='decode13')(x)
    x = Conv2D(16, (2, 2), activation='elu', padding='same', name='decode14')(x)
    decoded = Conv2D(1, (2, 2), activation='sigmoid', padding='same', name='decode15')(x)
    return Model(inp, decoded)

decoder = Decoder()
decoder.summary()


# In[ ]:


def Autoencoder():
    inp = Input(shape=INPUT_SHAPE)
    x = Conv2D(128, (4, 4), activation='elu', padding='same',name='encode1')(inp)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',name='encode2')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',name='encode3')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same',name='encode4')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',name='encode5')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same',name='encode6')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same',name='encode7')(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same',name='encode8')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same',name='encode9')(x)
    x = Flatten()(x)
    x = Dense(256, activation='elu',name='encode10')(x)
    encoded = Dense(128, activation='sigmoid',name='encode11')(x)
    x = Dense(256, activation='elu', name='decode1')(encoded)
    x = Dense(768, activation='elu', name='decode2')(x)
    x = Reshape((4, 6, 32))(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same', name='decode3')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='decode4')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (2, 2), activation='elu', padding='same', name='decode5')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same', name='decode6')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (2, 2), activation='elu', padding='same', name='decode7')(x)
    x = Conv2D(128, (3, 3), activation='elu', padding='same', name='decode8')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (2, 2), activation='elu', padding='same', name='decode9')(x)
    x = Conv2D(64, (4, 4), activation='elu', padding='same', name='decode10')(x)
    x = Conv2D(128, (3, 3), activation='elu', padding='same', name='decode11')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (4, 4), activation='elu', padding='same', name='decode12')(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same', name='decode13')(x)
    x = Conv2D(16, (2, 2), activation='elu', padding='same', name='decode14')(x)
    decoded = Conv2D(1, (2, 2), activation='sigmoid', padding='same', name='decode15')(x)
    return Model(inp, decoded)

model = Autoencoder()
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


# ## Callbacks

# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=4, 
                                            verbose=1, 
                                            factor=0.5,
                                            min_lr=0.00001)
filepath = "Dancer_Auto_Model.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             save_best_only=True,
                             monitor='val_loss',
                             mode='min')

early_stopping = EarlyStopping(monitor='val_loss',
                              patience=8,
                              verbose=1,
                              mode='min',
                              restore_best_weights=True)


# ### Custom Image Sample Callback
# 
# Here is a custom callback I made named ImgSample. It tests the result of the autoencoder after every epoch by desplaying an sample image. The goal is to have the dancer come into focus as clearly as possible.

# In[ ]:


class ImgSample(Callback):

    def __init__(self):
       super(Callback, self).__init__() 

    def on_epoch_end(self, epoch, logs={}):
        sample_img = X_train[50]
        sample_img = sample_img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
        sample_img = self.model.predict(sample_img)[0]
        imshow(sample_img.reshape(64,96))
        plt.show()

imgsample = ImgSample()


# In[ ]:


imshow(X_train[50].reshape(64,96))


# ## Train the Autoencoder

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.fit(X_train, X_train,\n          epochs=30, \n          batch_size=32,\n          verbose=2,\n          validation_split=0.05,\n        callbacks=[learning_rate_reduction, checkpoint, early_stopping, imgsample])')


# ## Sample the Autoencoder Results
# 
# The reconstructions look pretty close to the originals, then the autoencoder works.

# In[ ]:


decoded_imgs = model.predict(X_train)


# In[ ]:


plt.figure(figsize=(20, 4))
for i in range(10):
    # original
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_train[i].reshape(IMG_HEIGHT, IMG_WIDTH))
    plt.axis('off')
 
    # reconstruction
    plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(decoded_imgs[i].reshape(IMG_HEIGHT, IMG_WIDTH))
    plt.axis('off')
 
plt.tight_layout()
plt.show()


# ## Save Models and Create Encoded Dataset

# In[ ]:


model.save_weights("Dancer_Auto_Weights.hdf5")


# In[ ]:


encoder = Encoder()
decoder = Decoder()

encoder.load_weights("Dancer_Auto_Weights.hdf5", by_name=True)
decoder.load_weights("Dancer_Auto_Weights.hdf5", by_name=True)

model.save('Dancer_Auto_Model.hdf5') 
decoder.save('Dancer_Decoder_Model.hdf5') 
encoder.save('Dancer_Encoder_Model.hdf5')
model.save_weights("Dancer_Auto_Weights.hdf5")
decoder.save_weights("Dancer_Decoder_Weights.hdf5")
encoder.save_weights("Dancer_Encoder_Weights.hdf5")


# In[ ]:


Encoder_imgs = encoder.predict(X_train)
Encoder_imgs.shape
np.save('Encoded_Dancer.npy',Encoder_imgs)


# ## Decode a Sample to Double Check Results
# 
# If the encoder and decoder models are working correctly, the dancer should appear like in the reconstruction of the autoencoder above.

# In[ ]:


decoded_imgs = decoder.predict(Encoder_imgs[0:20])

plt.figure(figsize=(20, 4))
for i in range(10):
    # reconstruction
    plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(decoded_imgs[i].reshape(IMG_HEIGHT, IMG_WIDTH))
    plt.axis('off')
 
plt.tight_layout()
plt.show()


# ## Part 2 Results
# The results are really good, there is only maybe a touch of blurriness around the hands after decoding. The image actually looks better than the binary image we started with. We can confidently proceed knowing that we can encode and decode the images without issue. I am quite happy with these results
# 
# ### Possible Improvements
# - The autoencoder works so well that we could do more without much issue, either compress further or use more detailed images.
# 
# - The Autoencoder could be used to make a much much larger training set. Even if the uncompressed images get to big for the memory limit, it is possible to just train the autoencoder on a subset of the images then compress the whole set after. A 128 array is not that big, I don't foresee resource exhaustion errors being an major issue, even for much larger datasets.
