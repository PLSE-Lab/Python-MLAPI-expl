#!/usr/bin/env python
# coding: utf-8

# # Generating Anime Faces with Deep Convolutional Generative Adversarial Networks With Keras
# This is the code for training a DCGAN to generate anime faces. Although I didn't train the model on here, I trained it locally. There is a link to a Jupyter Notebook with the loss values at the bottom of the page. The results of the model are shown there as well

# In[ ]:


from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

import os

from PIL import Image, ImageSequence

from matplotlib.pyplot import imshow


# In[ ]:


os.listdir('../input/')


# Adding every image into an array and getting rid of that pesky 4th alpha channel that pngs have

# In[ ]:


def load_data():
        data = []
        paths = []
        for r, d, f in os.walk(r'../input/anime-faces/data/data'):
            for file in f:
                if '.png' in file:
                    paths.append(os.path.join(r, file))

        for path in paths:
            img = Image.open(path)
            x = np.array(img)
            x = x[...,:3]
            if(x.shape == (64,64,3)):
                data.append(x)

        x_train = np.array(data)
        x_train = x_train.reshape(len(data),64,64,3)
        return x_train


# In[ ]:


class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 50

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        generator = self.generator

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()
        model.add(Dense(64 * 4 * 4, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((4, 4, 64)))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=(3,3), padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(3, kernel_size=(2,2), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)
    

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        X_train = load_data()

        #normalize pixel values
        X_train = X_train / 256

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))


# Trained for 15000 epochs locally. Results shown below

# In[ ]:


dcgan = DCGAN()
#dcgan.train(epochs=15000, batch_size=60, save_interval=100)


# # Sampling Through Latent Space of Generator
# ![sample1](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/perfectloop1.gif)
# ![sample2](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/perfectloop2.gif)
# ![sample3](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/perfectloop3.gif)
# ![sample4](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/perfectloop4.gif)
# # Generator Model
# ![gen](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/generator.JPG)
# 
# # Discriminator Model
# ![disc](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/discriminator.JPG)
# 
# # Training
# ![training](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/training.gif)
# 
# # Results
# ![pic](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/image_101037.png)
# ![pic](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/image_108400.png)
# ![pic](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/image_136257.png)
# ![pic](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/image_140609.png)
# ![pic](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/image_147776.png)
# ![pic](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/image_162085.png)
# ![pic](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/image_201763.png)
# ![pic](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/image_205310.png)
# ![pic](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/image_207384.png)
# ![pic](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/image_242376.png)
# ![pic](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/image_24908.png)
# ![pic](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/image_34030.png)
# ![pic](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/image_54458.png)
# ![pic](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/image_66887.png)
# ![pic](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/image_80884.png)
# ![pic](https://raw.githubusercontent.com/vee-upatising/Anime-DCGAN/master/results/image_9288.png)
# 
# 
# 
# # [View Notebook](https://nbviewer.jupyter.org/github/vee-upatising/Anime-DCGAN/blob/master/Anime%20DCGAN.ipynb)
