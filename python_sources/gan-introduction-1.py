#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
#print(os.listdir("../input"))

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam


# # Introduction
# GANs are powerful tools, and this tutorial will help you start using them, as well as introduce you to some important concepts.
# 
# ## What will you learn?
# 
# By time you finish this tutorial, you will:
# 
# - Understand basic concepts and building blocks of GANs
#   - e.g. Network Structure, input data, labels to use, and Cost Function
# - Build GANs model to generate hand-written digits
# 
# ## What do I need to know before getting started?
# 
# This tutorial will assume that you're familiar with some basic MLP/deep learning concepts, along with basics of numpy and Keras. If you're brand new to deep learning, I recommend you work through Course 1 and Couse 2 of this series (https://www.deeplearning.ai/deep-learning-specialization/), which doesn't assume deep learning background, before getting started.
# 
# # Basic Concept
# ![GAN Concept](https://sthalles.github.io/assets/dcgan/GANs.png)
# image from https://sthalles.github.io/intro-to-gans/
# 
# Here is the basic structure of GANs.
# 
# Generator receives input called Latent Variables. For mnist work, this is a set of random values with around 100 dimension. This is going to be a seed for generating image.
# 
# Discriminator receives an image as input, and try to classify if that image is real or genarated one. Output is a continuous value from 0 to 1, 0 representing a generated image, 1 representing a real image.
# 
# ## Training Discriminator
# ![Discriminator](https://i.imgur.com/7e8Ga74.png)
# 
# Training set will be a set of images and labels. If the image is real one, the label is going to be 1. Otherwise, the label is goint to be 0.
# 
# In programming code, we use noise as input for Generator, and use predict method to generate images.
# 
# Then we are going to use that image as input for Discriminator.
# 
# While you are training Discriminator, you are not touching Generator. We only use Generator to generate images here.
# 
# ## Training Generator
# ![Generater](https://i.imgur.com/2mNi2DN.png)
# 
# As the output of GANs is a value represents probability of the input image came from real image set, we are going to use Combined network of [Generator + Discriminator] in training Generator.
# 
# However, when you trian Generator, you don't want to update weights in Discriminator. This can be done by setting [trainable = False].
# 
# As usual, input for Generator is a set of generated noise.
# 
# ### Labeling in training Generator
# Tricky part of GANs is labeling in training Generator. Since the image we use here is obviously a fake one, the label should be 0.
# 
# However, we label them 1 in training Generator. This is because the purpose of Generator is to deceive Discriminator.
# 
# So for Generator, loss function should be lower when the Discriminator classifies fake image as real.
# 
# Since both training is binary classification task, we can use binary crossentropy for loss function.
# 
# # Let's Get Started
# 
# ## Basic Network Structure
# First, we define GAN class and build a basic GANs structure in constructor.

# In[ ]:


class GAN():
    def __init__(self):
        #mnist input size
        self.img_rows = 28 
        self.img_cols = 28
        self.channels = 1 # black and white
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        # size of Latent Variable (input to generator)
        self.z_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # discriminator model
        self.discriminator = self.build_discriminator() # take mnist data shape as input, and output sigmoid - [0,1]
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Generator model (no need to compile, as we don't train generator by itself)
        self.generator = self.build_generator()

        # Combined model to use in training Generator
        self.combined = self.build_combined1()
        #self.combined = self.build_combined2()
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


# Later in this notebook, I will introduce two ways of building Generator-Discriminator combined network for training Generator. (build_combined1 and build_combined2 methods)
# 
# Before that, let's write build_generator and build_discriminator methods.

# In[ ]:


def build_generator(self):
    # The Generator takes noise as input and generate images
    noise_shape = (self.z_dim,)
    
    # Use Sequential for starting
    model = Sequential()

    # Add Dense Layer
    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    # Normalize (mean=0, std=1, momentum: moving average momentum)
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(self.img_shape), activation='tanh'))
    model.add(Reshape(self.img_shape))
    model.summary()
    return model

def build_discriminator(self):
    img_shape = (self.img_rows, self.img_cols, self.channels)
    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model


# BatchNorm doesn't affect much in this case, but it can make learning more efficient in some cases.
# 
# In this code I use BatchNorm only in Generator, because I don't want to normalize mixture of real image and noisy fake image especially on early learning stage.

# In[ ]:


def build_combined1(self): #simpler to write
    self.discriminator.trainable = False
    model = Sequential([self.generator, self.discriminator]) # connect models by using a list of model names
    return model

def build_combined2(self): #suitable for building more complex network structure in the future
    z = Input(shape=(self.z_dim,)) #define latent variable
    img = self.generator(z) #use it to generate image
    self.discriminator.trainable = False
    valid = self.discriminator(img) #sigmoid output [0-1]
    model = Model(z, valid) # use Model method using the first input and the last output
    model.summary()
    return model


# Then there are two ways of writing combined networks.
# 
# build_combined1 uses Sequential method and simpler to write. You can connect models by just using a list of model names.
# 
# On the other hand, build_combined2 uses Model method using the first input and the last output.
# 
# It looks more complicated compared to build_combined1, but suitable for building more complex network structure in the future.
# 
# ## Training
# Let's move on to write training code.
# 
# The loss function introduced in GANs original paper https://arxiv.org/abs/1406.2661 is below.
# 
# ![loss func](https://i.imgur.com/VvtT1Vu.jpg)
# 
# G is for Generator, and D is for Discriminator.
# 
# In the right formula, the first term is used when using real image, and the second term is used when using fake image came from Generator.
# 
# ### Loss function for Discriminator
# 
# D(x) represents the probability that x came from real image set. We train D to maximize the probability of assigning the correct label to both real images and fake images from Generator.
# 
# That means:
# 
# - when using real image > training procedure is to maximize log(D(x)), thus to output D(x) = 1.
# - when using fake image > training procedure is to maximize log(1-D(G(z))), thus to output D(x) = 0.
# 
# ### Loss function for Generator
# 
# The training procedure for G is to maximize the probability of D making a mistake, thus to output D(G(z)) = 1.
# 
# See below

# In[ ]:


x = np.linspace(0.01, 1, 100);
plt.figure(0)
plt.plot(x, np.log(x))
plt.xlabel("D(x)")
plt.ylabel("log(D(x)")
plt.show()


# In[ ]:


x = np.linspace(0, 0.99, 100);
plt.figure(0)
plt.plot(x, np.log(1-x))
plt.xlabel("D(G(z))")
plt.ylabel("log(1-D(G(z)))")
plt.show()


# Having that in mind, let's write training method!

# In[ ]:


def train(self, epochs, batch_size=128, save_interval=50):
    # load the dataset
    (X_train, _), (_, _) = mnist.load_data()

    # rescale to -1 - 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    # use later to make half real - half fake set of images
    half_batch = int(batch_size / 2)

    num_batches = int(X_train.shape[0] / half_batch)
    print('Number of batches:', num_batches)
    
    for epoch in range(epochs):

        for iteration in range(num_batches):

            # ---------------------
            #  train Discriminator
            # ---------------------

            # Generate image for half of the training set to use
            noise = np.random.normal(0, 1, (half_batch, self.z_dim))
            gen_imgs = self.generator.predict(noise)

            # Select a random batch of images (for the other half of the training set)
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # train discriminator / use real images and fake images separetely
            #np.ones > ground truth for real images
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            #np.zeros > ground truth for fake images
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            # Take average of both loss function
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.z_dim))

            # Labels for generated images (for training generator) are 1 
            valid_y = np.array([1] * batch_size)

            # train generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # print progress
            print ("epoch:%d, iter:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, iteration, d_loss[0], 100*d_loss[1], g_loss))


# Here it goes!
# 
# Now we implemented a complete set of GANs program.
# 
# Let's combine those methods to complete GAN() class.

# In[ ]:


class GAN():
    def __init__(self): # Build basic structure of GANs in constructor
        #mnist input size
        self.img_rows = 28 
        self.img_cols = 28
        self.channels = 1 # black and white
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        # size of Latent Variable (input to generator)
        self.z_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # discriminator model
        self.discriminator = self.build_discriminator() # take mnist data shape as input, and output sigmoid - [0,1]
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Generator model (no need to compile, as we don't train generator by itself)
        self.generator = self.build_generator()

        # Combined model to use in training generator
        self.combined = self.build_combined1()
        #self.combined = self.build_combined2()
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        # The Generator takes noise as input and generate images
        noise_shape = (self.z_dim,)
        
        # Use Sequential for starting
        model = Sequential()

        # Add Dense Layer
        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        # Normalize (mean=0, std=1, momentum: moving average momentum)
        # this doesn't affect much for this case, but BatchNorm can lead to efficient learning
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        model.summary()
        return model

    def build_discriminator(self):
        img_shape = (self.img_rows, self.img_cols, self.channels)
        model = Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model
    
    def build_combined1(self): #simpler to write
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator]) # connect models by using a list of model names
        return model

    def build_combined2(self): #suitable for building more complex network structure in the future
        z = Input(shape=(self.z_dim,)) #define latent variable
        img = self.generator(z) #use it to generate image
        self.discriminator.trainable = False
        valid = self.discriminator(img) #sigmoid output [0-1]
        model = Model(z, valid) # use Model method using the first input and the last output
        model.summary()
        return model

    def train(self, epochs, batch_size=128, save_interval=50):
        # load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # rescale to -1 - 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # use later to make half real - half fake set of images
        half_batch = int(batch_size / 2)

        num_batches = int(X_train.shape[0] / half_batch)
        print('Number of batches:', num_batches)
        
        for epoch in range(epochs):

            for iteration in range(num_batches):

                # ---------------------
                #  train Discriminator
                # ---------------------

                # Generate image for half of the training set to use
                noise = np.random.normal(0, 1, (half_batch, self.z_dim))
                gen_imgs = self.generator.predict(noise)

                # Select a random batch of images (for the other half of the training set)
                idx = np.random.randint(0, X_train.shape[0], half_batch)
                imgs = X_train[idx]

                # train discriminator / use real images and fake images separetely
                #np.ones > ground truth for real images
                d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
                #np.zeros > ground truth for fake images
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
                # Take average of both loss function
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  train Generator
                # ---------------------

                noise = np.random.normal(0, 1, (batch_size, self.z_dim))

                # Labels for generated images (for training generator) are 1 
                valid_y = np.array([1] * batch_size)

                # train generator
                g_loss = self.combined.train_on_batch(noise, valid_y)

                # print progress
                print ("epoch:%d, iter:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, iteration, d_loss[0], 100*d_loss[1], g_loss))

                # if at save interval > save generated image samples
                if epoch % save_interval == 0:
                    self.save_imgs(epoch)

    def save_imgs(self, epoch):
        # number of rows and columns for displaying generated images
        r, c = 5, 5

        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images to 0 - 1
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


# Now we are all good to go executing the program.

# In[ ]:


gan = GAN()


# In[ ]:


gan.train(epochs=1, batch_size=1200, save_interval=1)


# # Generated Image
# After 400 iterations
# ![400](https://i.imgur.com/4Js02jt.png)
# 
# After 1000 iterations
# ![1000](https://i.imgur.com/iZa9jKx.png)
# 
# After 5000 iterations
# ![5000](https://i.imgur.com/3EzEc0A.png)
# 
# After 15000 iterations
# ![15000](https://i.imgur.com/WKeqIFK.png)
# 
# # Conclusion
# Congratulations! Now you understand the basic concept and building blocks of GANs.
# 
# But there is still much more journey ahead of you. How can you improve this program? Maybe you can try different network structures or different loss functions?
# 
# You may notice white pixels in random places in generated images. This is because we used full connected layers. Simple MLP doesn't really take the relations between pixels into account. Maybe we can use a network that does consider them to generate better images.
# 
# Also you may notice that you cannot specify which digit to generate. How do we make our program to generate, say 2 or 7 specifically?
# 
# In the following tutorial, we will learn how to do those things! Stay tuned;)
# 
# ### References
# - https://arxiv.org/pdf/1406.2661.pdf Generative Adversarial Nets paper, Goodfellow et al, 2014
# - http://www.deeplearningbook.org/ by Goodfellow, part3 20 > Deep Generative Models
# - https://www.youtube.com/watch?v=9JpdAg6uMXs Introduction to GANs, NIPS 2016
# - https://github.com/soumith/ganhacks How to Train a GAN? Tips and tricks to make GANs work
# - https://github.com/eriklindernoren/Keras-GAN/tree/master/gan Reference code that I used (and edit as I see fit) for this notebook, thanks!

# In[ ]:




