#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


DATA_DIR = '/kaggle/input/anime-faces/data/data/'
len(os.listdir(DATA_DIR))


# In[ ]:


import tensorflow as tf
from scipy.misc import imread, imshow
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import UpSampling2D, Conv2D, Dense, BatchNormalization, LeakyReLU, Input,                          Reshape, MaxPooling2D, Flatten, AveragePooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam

import glob


# In[ ]:


def generator():
    model = Sequential([
            Input(shape=(100, )),
            Dense(2048),
            LeakyReLU(0.2),
            
            Dense(8 * 8 * 256),
            BatchNormalization(),
            LeakyReLU(0.2),
        
            Reshape((8, 8, 256)),
            Conv2D(128, (5, 5), padding='same'),
            BatchNormalization(),
            LeakyReLU(0.2),
            
            Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'),
            LeakyReLU(0.2),
        
            Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'),
            LeakyReLU(0.2),
        
            Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh')
            
    ])
    return model


# In[ ]:


G = generator()
G.summary()


# In[ ]:


def discriminator():
    model = Sequential([
            Input(shape=(64, 64, 3)),
            Conv2D(64, (5, 5), padding='valid'),
            BatchNormalization(),
            LeakyReLU(0.2),
            AveragePooling2D(pool_size=(2, 2)),
        
            Conv2D(128, (3, 3), padding='valid'),
            BatchNormalization(),
            LeakyReLU(0.2),
            AveragePooling2D(pool_size=(2, 2)),
            
            Conv2D(256, (3, 3), padding='valid'),
            BatchNormalization(),
            LeakyReLU(0.2),
            AveragePooling2D(pool_size=(2, 2)),
        
            Flatten(),
            Dense(1024),
            BatchNormalization(),
            LeakyReLU(0.2),
            
            Dense(1, activation='sigmoid')
            
    ])
    model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return model


# In[ ]:


D = discriminator()
D.summary()


# In[ ]:


def sample_noise(batch_size, noise_dim=100):
    return np.random.normal(size=(batch_size, noise_dim))

def smooth_pos_labels(y):
    return y - 0.3 + (np.random.random(y.shape) * 0.5) # Smooth real label to [0.7, 1.2]

def smooth_neg_labels(y):
    return y + np.random.random(y.shape) * 0.3 # Smooth fake label to [0.0, 0.3]


# In[ ]:


def load_images(data_dir):
    all_images = []
    for i, file_name in enumerate(glob.glob(data_dir + '*.png')):
        image = imread(file_name)
        all_images.append(image)
    X = np.array(all_images)
    X = (X - 127.5) / 127.5 #Normalize to -1 and 1 range
    np.random.shuffle(X)
    return X
X = load_images(DATA_DIR)


# In[ ]:


def show_5images(images):
    plt.figure()
    for i, image in enumerate(images):
        ax = plt.subplot(1, 5, i+1)
        plt.axis('off')
        plt.imshow(image)
show_5images(X[20: 25])


# In[ ]:


def compile_gan_model(D, G, noise_dim=100):
    D.trainable = False
    gan_input = Input(shape=(noise_dim))
    gan_output = D(G(gan_input))
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy',
                optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return gan
gan = compile_gan_model(D, G)


# In[ ]:


gan.summary()


# In[ ]:


def load_batch(X, batch_size):
    n = X.shape[0]
    n_batches = int(n / batch_size)
    for i in range(n_batches):
        yield X[i*batch_size: (i+1)*batch_size]


# In[ ]:


def train(X, D, G, gan, epochs=100, batch_size=64):
    discriminator_loss = []
    generator_loss = []
    for i in range(1, epochs+1):
        for x in load_batch(X, batch_size):
            real_images = x
            noise = sample_noise(batch_size)
            generated_images = G.predict(noise)
            
            y_real = np.ones(batch_size)
            y_real = smooth_pos_labels(y_real)
            y_fake = np.zeros(batch_size)
            y_fake = smooth_neg_labels(y_fake)
            
            d_loss_real = D.train_on_batch(x, y_real)
            d_loss_fake = D.train_on_batch(generated_images, y_fake)
            d_loss = d_loss_real + d_loss_fake
            discriminator_loss.append(d_loss)
            
            y_real = np.ones(batch_size)
            g_loss = gan.train_on_batch(noise, y_real)
            generator_loss.append(g_loss)
        
        print('[Epoch {0}]. Discriminator loss : {1}. Generator_loss: {2}.'.format(i, discriminator_loss[i], generator_loss[i]))
#         plt.figure(figsize=(10, 10))
        test_noise = sample_noise(5)
        test_images = G.predict(test_noise)
        show_5images(test_images)
        plt.show()
#         for j, img in enumerate(test_images):
#             ax = plt.subplot(1, 3, j+1)
#             plt.imshow(img)
#             plt.axis('off')
#             plt.show()


# In[ ]:


train(X, D, G, gan, 150)

