#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('wget "https://storage.googleapis.com/kaggle-competitions-data/kaggle/3364/all.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1560541212&Signature=iFwmlgQvQVaszdiLOqDd9I2pTHSSd2V42BZf2Y%2BhCYJqrI8egfhQ4Xih3yFsKw9G7o1yiXnybWgQQ8ZaeOb%2B1PKQJvognkN6Sb2Sm6fG2289R5mnAwUSKz5i1%2BYJmUqYb3wyUEICbSXC%2BFODa6AyZA403beQBjI3mVNLufA6Kh2ynM%2FbzyuEazw8C7NOWWuEYXJda9yNcG2vSIldtjauxOAe4Y5DewwEzHhhOV%2FCt6LLGK0KLRwIEDuESHkZBc1OpcQpBTNQRQVCNI6C5E%2BzFbPFVSeC%2FkLWx5isI%2BezEKD0QSRtmz0SbpgxXlaOPEVosISUJipPCYRPyD2mfiQP7Q%3D%3D&response-content-disposition=attachment%3B+filename%3Dchallenges-in-representation-learning-facial-expression-recognition-challenge.zip"')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('unzip all.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1560541212&Signature=iFwmlgQvQVaszdiLOqDd9I2pTHSSd2V42BZf2Y%2BhCYJqrI8egfhQ4Xih3yFsKw9G7o1yiXnybWgQQ8ZaeOb%2B1PKQJvognkN6Sb2Sm6fG2289R5mnAwUSKz5i1%2BYJmUqYb3w')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('mkdir "jj"')
import os
os.chdir("jj")


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('wget "https://storage.googleapis.com/kaggle-competitions-data/kaggle/3364/all.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1560541212&Signature=iFwmlgQvQVaszdiLOqDd9I2pTHSSd2V42BZf2Y%2BhCYJqrI8egfhQ4Xih3yFsKw9G7o1yiXnybWgQQ8ZaeOb%2B1PKQJvognkN6Sb2Sm6fG2289R5mnAwUSKz5i1%2BYJmUqYb3wyUEICbSXC%2BFODa6AyZA403beQBjI3mVNLufA6Kh2ynM%2FbzyuEazw8C7NOWWuEYXJda9yNcG2vSIldtjauxOAe4Y5DewwEzHhhOV%2FCt6LLGK0KLRwIEDuESHkZBc1OpcQpBTNQRQVCNI6C5E%2BzFbPFVSeC%2FkLWx5isI%2BezEKD0QSRtmz0SbpgxXlaOPEVosISUJipPCYRPyD2mfiQP7Q%3D%3D&response-content-disposition=attachment%3B+filename%3Dchallenges-in-representation-learning-facial-expression-recognition-challenge.zip"')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('mv "all.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1560541212&Signature=iFwmlgQvQVaszdiLOqDd9I2pTHSSd2V42BZf2Y%2BhCYJqrI8egfhQ4Xih3yFsKw9G7o1yiXnybWgQQ8ZaeOb%2B1PKQJvognkN6Sb2Sm6fG2289R5mnAwUSKz5i1%2BYJmUqYb3w" "test.zip"')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('unzip "test.zip"')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('tar -xvzf "fer2013.tar.gz"')


# In[ ]:


os.listdir("fer2013")


# In[ ]:


data = pd.read_csv("fer2013/fer2013.csv")


# In[ ]:


get_ipython().system('mkdir "processed"')


# In[ ]:


import pandas as pd
import cv2
import numpy as np
import random
import os
import sys

def unique_name(pardir,prefix,suffix='jpg'):
    filename = '{0}_{1}.{2}'.format(prefix,random.randint(1,10**8),suffix)
    filepath = os.path.join(pardir,filename)
    if not os.path.exists(filepath):
        return filepath
    unique_name(pardir,prefix,suffix)



#Get csv path from the commandline argument
dataset_path = "./fer2013/fer2013.csv"

#defining image size
image_size=(48,48)

#read csv
data = pd.read_csv(dataset_path,delimiter=',',dtype='a')

#separating label indexes from the dataframe
labels = np.array(data['emotion'],np.float)

#named labels to convert from index
named_labels = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

#separating row major pixels from dataframe
imagebuffer = np.array(data['pixels'])

#creating np array of image pixels
images = np.array([np.fromstring(image,np.uint8,sep=' ') for image in imagebuffer])


# num_shape = int(np.sqrt(images.shape[-1]))
# images.shape = (images.shape[0],num_shape,num_shape)


# In[ ]:


images_nn = images.reshape(images.shape[0],48,48)


# In[ ]:


print(images_nn[0])


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

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 48
        self.img_cols = 48
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

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

        model.add(Dense(128 * 12 * 12, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((12, 12, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

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
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
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

#         (X_train,Y_train),(X_test,Y_test)=mnist.load_data()
        X_train = images_nn
        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)
        
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

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("mnist_%d.png" % epoch)
        plt.close()




# In[ ]:



dcgan = DCGAN()
dcgan.train(epochs=20000, batch_size=32, save_interval=500)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import matplotlib.pyplot as plt
img = cv2.imread("mnist_19500.png")
plt.figure(figsize=(20,20))
plt.imshow(img)
plt.show()


# In[ ]:


dcgan.generator.save("generator.h6")
dcgan.discriminator.save("discriminator.h5")


# In[ ]:


get_ipython().system('ls')


# In[ ]:




