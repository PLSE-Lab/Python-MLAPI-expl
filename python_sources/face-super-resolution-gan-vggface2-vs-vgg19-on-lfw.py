#!/usr/bin/env python
# coding: utf-8

# # Experiments on Super-Resolution with Generative Adversarial Networks 
# 
# 
# 
# ************
# 
# 
# 
# ### Notebooks
# 
# 
# https://www.kaggle.com/sgladysh/face-super-resolution-gan-vggface2-vs-vgg19-on-lfw
# 
# https://www.kaggle.com/sgladysh/face-super-resolution-gan-vggface2-vs-vgg19-celeba
# 
# https://www.kaggle.com/sgladysh/super-resolution-gan-on-simpsons
# 
# 
# 

# 
# 
# 
# 
# 
# ![](https://github.com/s-gladysh/Super-Resolution-GAN-Experiments/raw/master/w2.jpg)
# 
# 
# 
# 
# *********************
# 
# 
# 

# 
# 
# 
# 
# 
# The goal of this project is to research and improve SOTA (State of the Art) in Super-Resolution Generative Adversarial Networks (GANs) through extending and modifying various existent architectures, to try new ideas, as well as to evaluate the results on different datasets (LFW, CelebA, Simpsons, etc).
# 
# 
# More concretetly, in this notebook I am focused on comparison between VGG19 and VGGFace2 used as pre-trained backbones for feature extraction within Super-Resolution GAN. 
# 
# 
# 
# 
# 
# #### Experiment in the given notebook is performed on Labeled Faces in the Wild (LFW) dataset.
# 
# 
# 
# 
# 
# ***************************
# 
# 
# Thanks to https://kaggle.com/mandanach for collaboration! 
# 
# 

# ***************************
# 
# 
# 
# # Research Hypothesis
# 
# 
# *Substituting VGG19 by VGGFace2 in SRGAN, if applied on human faces dataset, can improve quantitative metrics (PSNR, SSIM) and improve the obsevable visual quality of generated images*
# 
# 
# 
# *******************************

# 
# # Idea of Experiment
# 
# ********************
# 
# 
# 
# 
# 1 - Super-Resolution GAN with VGG19 - approximately same as in the original SRGAN article [2] and to some extent relies on codebase [3]
# 
# 
# *******************
# 
# 
# 
# 
# ![](https://github.com/s-gladysh/Super-Resolution-GAN-Experiments/raw/master/12s.jpg)
# 
# 
# ***************************************
# 
# 
# 
# 
# 2 - Super-Resolution GAN with VGGFace2 instead of VGG19
# 
# 
# ************************
# 
# 
# 
# 
# 
# ![](https://github.com/s-gladysh/Super-Resolution-GAN-Experiments/raw/master/12is.jpeg)
# 
# 
# **************************************
# 
# 
# 
# 
# PSNR and SSIM metrics are used in order to evaluate and compare the models.
# 
# 
# **********************
# 
# 
# 

# 
# ***********************
# 
# 
# ## Super-Resolution - Introduction
# 
# 
# Super-resolution is the process of recovering a high-resolution (HR) image from a low-resolution (LR) image. A recovered HR image then is referred as a super-resolution image or SR image. 
# 
# 
# 
# Super-resolution is still considered a challenging research problem in computer vision.
# 
# 
# *****************
# 
# 
# ### General Challenges of Super-Resolution
# 
# 
# #### Ill-posed inverse problem
# 
# Instead of a single unique solution, there exist multiple solutions for the same low-resolution image. To constrain the solution-space, reliable prior information is typically required.
# 
# 
# #### Complexity growth when up-scaling factor increases
# 
# The complexity of the problem increases as the up-scaling factor increases. At higher factors, the recovery of missing scene details becomes even more complex, and consequently it often leads to reproduction of wrong information.
# 
# 
# #### Complexity of assessment of the quality of output
# 
# Assessment of the quality of output is not straightforward and loosely correlate to human perception.

# ## Super-Resolution: Methods Classification
# 
# Super-resolution methods can be categorized into the following taxonomy according to the authors of the survey [1] based on their features: 
# 
# 
# 
# ***********************************
# 
# 
# ![](https://github.com/s-gladysh/Super-Resolution-GAN-Experiments/raw/master/SR_Taxonomy.png)
# 
# 
# 
# *******************************************
# 
# In this Notebook we will be focusing on Super-Resolution methods based on GAN models.
# 
# **************************************

# ## Generative Adversarial Networks 
# 
# Generative Adversarial Networks (GAN) [4] is a Deep Neural Networks architecture based on a game-theoretic approach, where two components of the model, namely a generator and discriminator, try to compete with each other. 
# 
# The Generator is trying to fool the Discriminator by creating faked images. Whereas the Discriminator is trying not to be fooled and learns how to detect faked ones better. In this way the Generator learns to generate better more realistic images.
# 
# 
# 
# *******************************
# 
# ![](https://github.com/s-gladysh/Super-Resolution-GAN-Experiments/raw/master/11.png)
# 
# 
# 
# 
# ***************************
# 
# 
# 
# ### GAN Applied to the problem of Super-Resolution: 
# 
# The Generator creates SR images that a Discriminator cannot distinguish as a real HR image or an artificially super-resolved output. In this manner, HR images with better perceptual quality are generated.
# 
# 
# 
# ********************************

# ### SRGAN - Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
# 
# The authors of SRGAN [2] proposed to use an adversarial objective function that promotes super-resolved (SR) outputs that lie close to the manifold of natural images.
# 
# The main highlight of their work is a multi-task loss formulation that consists of three main parts: 
# (1) a MSE loss that encodes pixel-wise similarity, 
# (2) a perceptual similarity metric in terms of a distance metric defined over high-level image representation (e.g., deep network features), and 
# (3) an adversarial loss that balances a min-max game between a generator and a discriminator (standard GAN objective [6]). 
# 
# 

# ## Loss Functions
# 
# 
# **********************
# 
# 
# **MSE = Pixel-by-Pixel Mean Squared Error** 
# 
# MSE works if the goal is to generate a picture having the best pixel colors conformity with the ground truth picture. However, in real-life scenarios it might be necessary to concentrate on the ***structure*** or ***relief*** of the picture.
# 
# 
# 
# 
# ************************
# 
# 
# 
# $${\large MSE = { 1 \over {mn} }  \sum\limits_{i=0}^{m-1} \sum\limits_{j=0}^{n-1} |I(i, j) - K(i, j)|^2}$$
# 
# 
# 
# 
# ***********************
# 
# 
# 
# **Perceptual Loss**  is a weighted sum of the content loss and adversarial loss
# 
# ***************************
# 
# 
# $${\large l^{SR} = l_X^{SR} + 10^{-3}l_{Gen}^{SR}}$$
# 
# 
# ***************************
# 
# 
# $l^{SR}$ - perceptual loss 
# 
# $l_X^{SR}$ - content loss 
# 
# $l_{Gen}^{SR}$ - adversarial loss 
# 
# ***************************
# 
# 
# 
# ****************************
# 
# **Content Loss** can be of two types:
# 
#          
# **Pixel-wise MSE** loss mean squared error between each pixel in real image and a pixel in generated image
# 
# **************************
# 
# 
# $${\large l_{MSE}^{SR} = {1 \over {r^2WH}} \sum\limits_{x=1}^{rW} \sum\limits_{y=1}^{rH}  (I_{x,y}^{HR} - G_{{\theta}_G} (I^{LR})_{x,y})^2}$$
# 
# 
# 
# ************************
# 
# 
# $ l_{MSE}^{SR} $ - pixel-wise mean squared error  
# 
# $ r $  -  down-sampling factor 
# 
# $ W $  -  width of a tensor, representing low-resolution image 
# 
# $ H $  -  heighth of a tensor, representing low-resolution image 
# 
# $ {x, y} $  -  pixel coordinates 
# 
# $ I^{HR} $  -  high-resolution image
# 
# $ I^{LR} $  -  low-resolution image 
# 
# $ G $  -  generating function 
# 
# $ G_{{\theta}_G} $  -  generating function parametrized with $ {\theta}_G $ 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# ***************************
# 
# **VGG loss** is the Euclidean distance between the feature maps of the generated image and the real image
# 
# 
# ***********************
# 
# 
# 
# $${\large l_{{VGG}/{i,j}}^{SR} = {1 \over {W_{i,j}H_{i,j}}} \sum\limits_{x=1}^{W_{i,j}} \sum\limits_{y=1}^{H_{i,j}}  ({\phi}_{i,j}(I^{HR})_{x,y} - {\phi}_{i,j} (G_{{\theta}_G} (I^{LR}))_{x,y})^2}$$
# 
# 
# **************************
# 
# 
# 
# $ l_{{VGG}/{i,j}}^{SR} $  -  VGG loss
# 
# 
# 
# ***************************
# 
# 
# 
# $ {\phi}_{i,j} $  -   the feature map obtained by the j-th convolution (after activation) before the i-th maxpooling layer within the VGG19 network
# 
# 
# 
# 
# 
# ***********************
# 
# 
# **Adversarial Loss** is calculated based on probabilities provided by Discriminator
# 
# *************************
# 
# 
# $${\large l_{Gen}^{SR} = \sum\limits_{n=1}^{N} - \log{D_{{\theta}_D}} (G_{{\theta}_G} (I^{LR}))}$$
# 
# 
# ************************
# 
# 
# $ l_{Gen}^{SR} $  -  generative loss
# 
# 
# ***********************
# 
# 
# $ D $  -  discriminator function 
# 
# $ D_{{\theta}_D} $  -  discriminator function parametrized with $ {\theta}_D $ 
# 
# $ {D_{{\theta}_D}} (G_{{\theta}_G} (I^{LR})) $   -  probability that the reconstructed image $ G_{{\theta}_G} (I^{LR}) $  is a natural HR image
# 
# 
# **********************
# 
# **Discriminator** is trained to solve maximization:
# 
# ************************
# 
# $${\large \min\limits_{{\theta}_G} \max\limits_{{\theta}_D}  {\mathbb E}_{I^{HR} \sim p_{train} (I^{HR})} [\log{D_{{\theta}_D}} (I^{HR})]  +  {\mathbb E}_{I^{LR} \sim p_{G} (I^{HR})} [\log{1 - D_{{\theta}_D}} ({G_{{\theta}_G}} (I^{LR}))]}$$
# 
# 
# ***********************
# 
# **Generator** is trained to solve minimization:
# 
# ************************
# 
# 
# $$ { \large \hat{\theta}_G = \arg \min\limits_{{\theta}_G} {1 \over {N}} \sum\limits_{n=1}^{N} l^{SR} (G_{{\theta}_G} (I_{n}^{LR}),  I_{n}^{HR}) } $$
# 
# ************************
# 
# 
# 

# ***********************************************
# 
# # Implementation: initial steps
# 
# 
# 
# ### Import libraries 
# 

# In[ ]:


#!pip install tf-nightly


# In[ ]:


import tensorflow


# In[ ]:


#!pip install cloud-tpu-client
#from cloud_tpu_client import Client
#print(tensorflow.__version__)
#Client().configure_tpu_version(tensorflow.__version__, restart_type='ifNeeded')


# In[ ]:


from keras import Input
from keras.applications import VGG19
from keras.callbacks import TensorBoard
from keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam

import glob
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from imageio import imread
from skimage.transform import resize as imresize

from PIL import Image


# ### Use fixed seeds for random number generators to guarantee reproducible results

# In[ ]:


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
tensorflow.random.set_seed(SEED)


#  ### Define some  hyper-parameters
# 

# In[ ]:


# let's start with 200 epochs for now, bit keep in mind, that it might be necessary to train more afterwards

epochs1 = 200
epochs2 = 200


# let's start from batch size equals to 8 (due to RAM limits)
batch_size = 8

# define the shape of low resolution image (LR) 
low_resolution_shape = (64, 64, 3)

# define the shape of high resolution image (HR) 
high_resolution_shape = (256, 256, 3)

# for simplicity let's start with Adam
common_optimizer1 = Adam(0.0002, 0.5)
common_optimizer2 = Adam(0.0002, 0.5)


# ## Load and Transform the Data
# 
# 

# In[ ]:


data_dir = "/kaggle/input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/*/*.*"


# LFW dataset contains images (photos) of people provided with attribute labels [6]
# 

# 
# *****************************
# 
# 

# # Implementation of Super-Resolution GAN

# ## Generator
# 
# 

# 
# 
# 
# ![](https://github.com/s-gladysh/Super-Resolution-GAN-Experiments/raw/master/Generator.png)
# 
# 
# 
# 

# The code is based on SRGAN implementation [3] with several experimental changes. 
# 

# In[ ]:


def residual_block(x):

    filters = [64, 64]
    kernel_size = 3
    strides = 1
    padding = "same"
    momentum = 0.8
    activation = "relu"

    res = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=strides, padding=padding)(x)
    res = Activation(activation=activation)(res)
    res = BatchNormalization(momentum=momentum)(res)

    res = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=strides, padding=padding)(res)
    res = BatchNormalization(momentum=momentum)(res)

    res = Add()([res, x])
    return res


# In[ ]:


def build_generator1():
    
    # use 16 residual blocks in generator
    residual_blocks = 16
    momentum = 0.8
    
    # dimension equals to LR - Low Resolution
    input_shape = (64, 64, 3)
    
    # input layer for the generator network
    input_layer = Input(shape=input_shape)
    
    # pre-residual block: convolutional layer before residual blocks 
    gen1 = Conv2D(filters=64, kernel_size=9, strides=1, padding='same', activation='relu')(input_layer)
    
    # add 16 residual blocks
    res = residual_block(gen1)
    for i in range(residual_blocks - 1):
        res = residual_block(res)
    
    # post-residual block: convolutional layer and batch-norm layer after residual blocks
    gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
    gen2 = BatchNormalization(momentum=momentum)(gen2)
    
    # take the sum of pre-residual block(gen1) and post-residual block(gen2)
    gen3 = Add()([gen2, gen1])
    
    # UpSampling: learning to increase dimensionality
    gen4 = UpSampling2D(size=2)(gen3)
    gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
    gen4 = Activation('relu')(gen4)
    
    # UpSampling: learning to increase dimensionality
    gen5 = UpSampling2D(size=2)(gen4)
    gen5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen5)
    gen5 = Activation('relu')(gen5)
    
    # convolution layer at the output
    gen6 = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen5)
    output = Activation('tanh')(gen6)
    
    # model 
    model = Model(inputs=[input_layer], outputs=[output], name='generator')
    return model


# In[ ]:


def build_generator2():
    
    # use 16 residual blocks in generator
    residual_blocks = 16
    momentum = 0.8
    
    # dimension equals to LR - Low Resolution
    input_shape = (64, 64, 3)
    
    # input layer for the generator network
    input_layer = Input(shape=input_shape)
    
    # pre-residual block: convolutional layer before residual blocks 
    gen1 = Conv2D(filters=64, kernel_size=9, strides=1, padding='same', activation='relu')(input_layer)
    
    # add 16 residual blocks
    res = residual_block(gen1)
    for i in range(residual_blocks - 1):
        res = residual_block(res)
    
    # post-residual block: convolutional layer and batch-norm layer after residual blocks
    gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
    gen2 = BatchNormalization(momentum=momentum)(gen2)
    
    # take the sum of pre-residual block(gen1) and post-residual block(gen2)
    gen3 = Add()([gen2, gen1])
    
    # UpSampling: learning to increase dimensionality
    gen4 = UpSampling2D(size=2)(gen3)
    gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
    gen4 = Activation('relu')(gen4)
    
    # UpSampling: learning to increase dimensionality
    gen5 = UpSampling2D(size=2)(gen4)
    gen5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen5)
    gen5 = Activation('relu')(gen5)
    
    # convolution layer at the output
    gen6 = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen5)
    output = Activation('tanh')(gen6)
    
    # model 
    model = Model(inputs=[input_layer], outputs=[output], name='generator')
    return model


# # Discriminator
# 

# ![](https://github.com/s-gladysh/Super-Resolution-GAN-Experiments/raw/master/Discriminator.png)
# 

# In[ ]:


def build_discriminator1():
    
    # define hyper-parameters
    leakyrelu_alpha = 0.2
    momentum = 0.8
    
    # dimentions correspond to HR - High Resolution
    input_shape = (256, 256, 3)
    
    # input layer for discriminator
    input_layer = Input(shape=input_shape)
    
    # 8 convolutional layers with batch normalization  
    dis1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
    dis1 = LeakyReLU(alpha=leakyrelu_alpha)(dis1)

    dis2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dis1)
    dis2 = LeakyReLU(alpha=leakyrelu_alpha)(dis2)
    dis2 = BatchNormalization(momentum=momentum)(dis2)

    dis3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
    dis3 = LeakyReLU(alpha=leakyrelu_alpha)(dis3)
    dis3 = BatchNormalization(momentum=momentum)(dis3)

    dis4 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dis3)
    dis4 = LeakyReLU(alpha=leakyrelu_alpha)(dis4)
    dis4 = BatchNormalization(momentum=0.8)(dis4)

    dis5 = Conv2D(256, kernel_size=3, strides=1, padding='same')(dis4)
    dis5 = LeakyReLU(alpha=leakyrelu_alpha)(dis5)
    dis5 = BatchNormalization(momentum=momentum)(dis5)

    dis6 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(dis5)
    dis6 = LeakyReLU(alpha=leakyrelu_alpha)(dis6)
    dis6 = BatchNormalization(momentum=momentum)(dis6)

    dis7 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis6)
    dis7 = LeakyReLU(alpha=leakyrelu_alpha)(dis7)
    dis7 = BatchNormalization(momentum=momentum)(dis7)

    dis8 = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(dis7)
    dis8 = LeakyReLU(alpha=leakyrelu_alpha)(dis8)
    dis8 = BatchNormalization(momentum=momentum)(dis8)
    
    # fully-connected layer 
    dis9 = Dense(units=1024)(dis8)
    dis9 = LeakyReLU(alpha=0.2)(dis9)
    
    # last fully-connected layer - for classification 
    output = Dense(units=1, activation='sigmoid')(dis9)
    
    
    model = Model(inputs=[input_layer], outputs=[output], name='discriminator')
    return model


# In[ ]:


def build_discriminator2():
    
    # define hyper-parameters
    leakyrelu_alpha = 0.2
    momentum = 0.8
    
    # dimentions correspond to HR - High Resolution
    input_shape = (256, 256, 3)
    
    # input layer for discriminator
    input_layer = Input(shape=input_shape)
    
    # 8 convolutional layers with batch normalization  
    dis1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
    dis1 = LeakyReLU(alpha=leakyrelu_alpha)(dis1)

    dis2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dis1)
    dis2 = LeakyReLU(alpha=leakyrelu_alpha)(dis2)
    dis2 = BatchNormalization(momentum=momentum)(dis2)

    dis3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
    dis3 = LeakyReLU(alpha=leakyrelu_alpha)(dis3)
    dis3 = BatchNormalization(momentum=momentum)(dis3)

    dis4 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dis3)
    dis4 = LeakyReLU(alpha=leakyrelu_alpha)(dis4)
    dis4 = BatchNormalization(momentum=0.8)(dis4)

    dis5 = Conv2D(256, kernel_size=3, strides=1, padding='same')(dis4)
    dis5 = LeakyReLU(alpha=leakyrelu_alpha)(dis5)
    dis5 = BatchNormalization(momentum=momentum)(dis5)

    dis6 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(dis5)
    dis6 = LeakyReLU(alpha=leakyrelu_alpha)(dis6)
    dis6 = BatchNormalization(momentum=momentum)(dis6)

    dis7 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis6)
    dis7 = LeakyReLU(alpha=leakyrelu_alpha)(dis7)
    dis7 = BatchNormalization(momentum=momentum)(dis7)

    dis8 = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(dis7)
    dis8 = LeakyReLU(alpha=leakyrelu_alpha)(dis8)
    dis8 = BatchNormalization(momentum=momentum)(dis8)
    
    # fully-connected layer 
    dis9 = Dense(units=1024)(dis8)
    dis9 = LeakyReLU(alpha=0.2)(dis9)
    
    # last fully-connected layer - for classification 
    output = Dense(units=1, activation='sigmoid')(dis9)
    
    
    model = Model(inputs=[input_layer], outputs=[output], name='discriminator')
    return model


# *************************

# # Feature Extractors based on VGG19

# In[ ]:


VGG19_base = VGG19(weights="imagenet")


# In[ ]:


from tensorflow.keras.utils import plot_model
from IPython.display import Image


# In[ ]:


plot_model(VGG19_base, to_file='vgg19_base.png', show_shapes=True)
Image(filename='vgg19_base.png') 


# In[ ]:


"""
def build_VGG19_b3c1():
    input_shape = (256, 256, 3)
    VGG19_base.outputs = [VGG19_base.get_layer('block3_conv1').output]
    input_layer = Input(shape=input_shape)
    features = VGG19_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGG19_b3c1 = build_VGG19_b3c1()
VGG19_b3c1.trainable = False
VGG19_b3c1.compile(loss='mse', optimizer=common_optimizer2, metrics=['accuracy'])
"""


# In[ ]:


"""
def build_VGG19_b3c2(): 
    input_shape = (256, 256, 3)
    VGG19_base.outputs = [VGG19_base.get_layer('block3_conv1').output]
    input_layer = Input(shape=input_shape)
    features = VGG19_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGG19_b3c2 = build_VGG19_b3c2()
VGG19_b3c2.trainable = False
VGG19_b3c2.compile(loss='mse', optimizer=common_optimizer2, metrics=['accuracy'])
"""


# In[ ]:


def build_VGG19_b3c3(): 
    input_shape = (256, 256, 3)
    VGG19_base.outputs = [VGG19_base.get_layer('block3_conv3').output]
    input_layer = Input(shape=input_shape)
    features = VGG19_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGG19_b3c3 = build_VGG19_b3c3()
VGG19_b3c3.trainable = False
VGG19_b3c3.compile(loss='mse', optimizer=common_optimizer2, metrics=['accuracy'])


# In[ ]:


"""
def build_VGG19_b3c4():
    input_shape = (256, 256, 3)
    VGG19_base.outputs = [VGG19_base.get_layer('block3_conv4').output]
    input_layer = Input(shape=input_shape)
    features = VGG19_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGG19_b3c4 = build_VGG19_b3c4()
VGG19_b3c4.trainable = False
VGG19_b3c4.compile(loss='mse', optimizer=common_optimizer2, metrics=['accuracy'])
"""


# In[ ]:


"""
def build_VGG19_b4c1():  
    input_shape = (256, 256, 3)
    VGG19_base.outputs = [VGG19_base.get_layer('block4_conv1').output]
    input_layer = Input(shape=input_shape)
    features = VGG19_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGG19_b4c1 = build_VGG19_b4c1()
VGG19_b4c1.trainable = False
VGG19_b4c1.compile(loss='mse', optimizer=common_optimizer2, metrics=['accuracy'])
"""


# In[ ]:


"""
def build_VGG19_b4c2(): 
    input_shape = (256, 256, 3)
    VGG19_base.outputs = [VGG19_base.get_layer('block4_conv2').output]
    input_layer = Input(shape=input_shape)
    features = VGG19_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGG19_b4c2 = build_VGG19_b4c2()
VGG19_b4c2.trainable = False
VGG19_b4c2.compile(loss='mse', optimizer=common_optimizer2, metrics=['accuracy'])
"""


# In[ ]:


"""
def build_VGG19_b4c3():
    input_shape = (256, 256, 3)
    VGG19_base.outputs = [VGG19_base.get_layer('block4_conv3').output]
    input_layer = Input(shape=input_shape)
    features = VGG19_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGG19_b4c3 = build_VGG19_b4c3()
VGG19_b4c3.trainable = False
VGG19_b4c3.compile(loss='mse', optimizer=common_optimizer2, metrics=['accuracy'])
"""


# In[ ]:


"""
def build_VGG19_b4c4():
    input_shape = (256, 256, 3)
    VGG19_base.outputs = [VGG19_base.get_layer('block4_conv4').output]
    input_layer = Input(shape=input_shape)
    features = VGG19_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGG19_b4c4 = build_VGG19_b4c4()
VGG19_b4c4.trainable = False
VGG19_b4c4.compile(loss='mse', optimizer=common_optimizer2, metrics=['accuracy'])
"""


# In[ ]:


"""
def build_VGG19_b5c1():
    input_shape = (256, 256, 3)
    VGG19_base.outputs = [VGG19_base.get_layer('block5_conv1').output]
    input_layer = Input(shape=input_shape)
    features = VGG19_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGG19_b5c1 = build_VGG19_b5c1()
VGG19_b5c1.trainable = False
VGG19_b5c1.compile(loss='mse', optimizer=common_optimizer2, metrics=['accuracy'])
"""


# In[ ]:


"""
def build_VGG19_b5c2():
    input_shape = (256, 256, 3)
    VGG19_base.outputs = [VGG19_base.get_layer('block5_conv2').output]
    input_layer = Input(shape=input_shape)
    features = VGG19_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGG19_b5c2 = build_VGG19_b5c2()
VGG19_b5c2.trainable = False
VGG19_b5c2.compile(loss='mse', optimizer=common_optimizer2, metrics=['accuracy'])
"""


# In[ ]:


"""
def build_VGG19_b5c3():
    input_shape = (256, 256, 3)
    VGG19_base.outputs = [VGG19_base.get_layer('block5_conv3').output]
    input_layer = Input(shape=input_shape)
    features = VGG19_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGG19_b5c3 = build_VGG19_b5c3()
VGG19_b5c3.trainable = False
VGG19_b5c3.compile(loss='mse', optimizer=common_optimizer2, metrics=['accuracy'])
"""


# In[ ]:


"""
def build_VGG19_b5c4():
    input_shape = (256, 256, 3)
    VGG19_base.outputs = [VGG19_base.get_layer('block5_conv4').output]
    input_layer = Input(shape=input_shape)
    features = VGG19_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGG19_b5c4 = build_VGG19_b5c4()
VGG19_b5c4.trainable = False
VGG19_b5c4.compile(loss='mse', optimizer=common_optimizer2, metrics=['accuracy'])
"""


# **********************************************

# ## Build & compile Discriminator 1
# 
# 

# In[ ]:


discriminator1 = build_discriminator1()
discriminator1.trainable = True
discriminator1.compile(loss='mse', optimizer=common_optimizer1, metrics=['accuracy'])


# ## Build Generator 1
# 
# 

# In[ ]:


generator1 = build_generator1()


# ********************************

# # Build & compile SRGAN-VGG19 
# 

# ### SRGAN-VGG19
# 
# 
# 
# ********************************************
# 
# 
# ![](https://github.com/s-gladysh/Super-Resolution-GAN-Experiments/raw/master/12.jpeg)
# 
# 
# 
# 
# *********************************************
# 
# 
# HR - High Resolution image
# 
# LR - Low Resolution image
# 
# SR - Super Resolution image
# 
# Generator - estimates for a LR its corresponding HR which is a SR
# 
# Discriminator - is trained to distinguish SR and real images
# 
# *****************************

# In[ ]:



def build_adversarial_model_vgg19(generator1, discriminator1, vgg19):
    
    # input layer for high-resolution images
    input_high_resolution1 = Input(shape=high_resolution_shape)

    # input layer for low-resolution images
    input_low_resolution1 = Input(shape=low_resolution_shape)

    # generate high-resolution images from low-resolution images
    generated_high_resolution_images1 = generator1(input_low_resolution1)

    # extract feature maps from generated images
    features1 = vgg19(generated_high_resolution_images1)
    
    # make a discriminator non-trainable 
    discriminator1.trainable = False
    discriminator1.compile(loss='mse', optimizer=common_optimizer1, metrics=['accuracy'])

    # discriminator will give us a probability estimation for the generated high-resolution images
    probs1 = discriminator1(generated_high_resolution_images1)

    # create and compile 
    adversarial_model1 = Model([input_low_resolution1, input_high_resolution1], [probs1, features1])
    adversarial_model1.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=common_optimizer1)
    return adversarial_model1
    


# In[ ]:


adversarial_model_vgg19 = build_adversarial_model_vgg19(generator1, discriminator1, VGG19_b3c3)


# **********************************

# # PSNR - Peak Signal-to-Noise Ratio

# ***********************
# 
# 
# 
# PSNR is the ratio between maximum possible power of signal and power of corrupting noise 
# 
# 
# ************************
# 
# 
# 
# $${\large PSNR = 10  \log_{10}  \left( {MAX_I^2 \over MSE} \right) }$$
# 
# 
# ************************
# 
# 
# 
# $ MAX_I $  -  maximum possible power of a signal of image I
# 
# $ MSE $  -  mean squared error pixel by pixel 
# 
# 
# 
# *************************
# 

# In[ ]:


def calc_psnr2(original_image, generated_image):
  original_image = tensorflow.convert_to_tensor(original_image, dtype=tensorflow.float32)
  generated_image = tensorflow.convert_to_tensor(generated_image, dtype=tensorflow.float32)
  psnr2 = tensorflow.image.psnr(original_image, generated_image, max_val=1.0)

  return tensorflow.math.reduce_mean(psnr2, axis=None, keepdims=False, name=None)


# In[ ]:


def plot_psnr1(psnr2_1):
    psnr2_1_means = psnr2_1['psnr2_quality']
    plt.figure(figsize=(10,8))
    plt.plot(psnr2_1_means, label="PSNR_1 quality")
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.legend()
    plt.show()


# In[ ]:


def plot_psnr2(psnr2_2):
    psnr2_2_means = psnr2_2['psnr2_quality']
    plt.figure(figsize=(10,8))
    plt.plot(psnr2_2_means, label="PSNR_2 quality")
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.legend()
    plt.show()


# In[ ]:


def plot_psnr12(psnr2_1, psnr2_2):
    psnr2_1_means = psnr2_1['psnr2_quality']
    psnr2_2_means = psnr2_2['psnr2_quality']
    plt.figure(figsize=(10,8))
    plt.plot(psnr2_1_means, label="PSNR_1 quality")
    plt.plot(psnr2_2_means, label="PSNR_2 quality")
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.legend()
    plt.show()


# 
# ***************
# 

# # SSIM - Structural Similarity Index

# *************************
# 
# 
# SSIM measures the perceptual difference between two similar images 
# 
# 
# **************************
# 
# 
# $${\large SSIM(x, y) = {(2 \mu_x \mu_y + c_1) (2 \sigma_{xy} + c_2) \over (\mu_x^2 + \mu_y^2 + c_1) ( \sigma_x^2 + \sigma_y^2 +c_2)}  }$$
# 
# 
# ************************
# 
# 
# 
# $ \mu_x $ - average value for the first image 
# 
# $ \mu_y $ - average value for the second image 
# 
# 
# 
# $ \sigma_x $            - standard deviation for the first image 
# 
# $ \sigma_y $            - standard deviation for the second image 
# 
# $ \sigma_{xy} = \mu_{xy} - \mu_x  \mu_y $       - covariation  
# 
# $ c_1, c_2 $            - coefficients 
# 
# 
# 
# **********************************************

# In[ ]:


def calc_ssim2(original_image, generated_image):
  original_image = tensorflow.convert_to_tensor(original_image, dtype=tensorflow.float32)
  generated_image = tensorflow.convert_to_tensor(generated_image, dtype=tensorflow.float32)
  ssim2 = tensorflow.image.ssim(original_image, generated_image, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)

  return tensorflow.math.reduce_mean(ssim2, axis=None, keepdims=False, name=None)


# In[ ]:


def plot_ssim1(ssim2_1):
    ssim2_1_means = ssim2_1['ssim2_quality']

    plt.figure(figsize=(10,8))
    plt.plot(ssim2_1_means, label="SSIM_1 quality")
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.legend()
    plt.show()


# In[ ]:


def plot_ssim2(ssim2_2):
    ssim2_2_means = ssim2_2['ssim2_quality']

    plt.figure(figsize=(10,8))
    plt.plot(ssim2_2_means, label="SSIM_2 quality")
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.legend()
    plt.show()


# In[ ]:


def plot_ssim12(ssim2_1, ssim2_2):
    ssim2_1_means = ssim2_1['ssim2_quality']
    ssim2_2_means = ssim2_2['ssim2_quality']

    plt.figure(figsize=(10,8))
    plt.plot(ssim2_1_means, label="SSIM_1 quality")
    plt.plot(ssim2_2_means, label="SSIM_2 quality")
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.legend()
    plt.show()


# In[ ]:


psnr2_1 = {'psnr2_quality': []}
ssim2_1 = {'ssim2_quality': []}

psnr2_2 = {'psnr2_quality': []}
ssim2_2 = {'ssim2_quality': []}

losses1 = {"d_history":[], "g_history":[]}
losses2 = {"d_history":[], "g_history":[]}


# ## Losses plotting

# In[ ]:


def plot_loss1(losses1):
    """
    @losses.keys():
        0: loss
        1: accuracy
    """

    d_loss1 = losses1['d_history']
    g_loss1 = losses1['g_history']
    
   
    plt.figure(figsize=(10,8))
    plt.plot(d_loss1, label="Discriminator1 loss")
    plt.plot(g_loss1, label="Generator1 loss")
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
 


# In[ ]:


def plot_loss2(losses2):
    """
    @losses.keys():
        0: loss
        1: accuracy
    """
    
    d_loss2 = losses2['d_history']
    g_loss2 = losses2['g_history']
    
    plt.figure(figsize=(10,8))
    
    plt.plot(d_loss2, label="Discriminator2 loss")
    plt.plot(g_loss2, label="Generator2 loss")
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
 


# In[ ]:


def plot_loss12(losses1, losses2):
    """
    @losses.keys():
        0: loss
        1: accuracy
    """

    d_loss1 = losses1['d_history']
    g_loss1 = losses1['g_history']
    
    d_loss2 = losses2['d_history']
    g_loss2 = losses2['g_history']
    
    plt.figure(figsize=(10,8))
    plt.plot(d_loss1, label="Discriminator1 loss")
    plt.plot(g_loss1, label="Generator1 loss")
    
    plt.plot(d_loss2, label="Discriminator2 loss")
    plt.plot(g_loss2, label="Generator2 loss")
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# ## Sampling images
# 

# In[ ]:


def sample_images(data_dir, batch_size, high_resolution_shape, low_resolution_shape):
    
    # create the list of all images, which are inside of data_dir catalogue
    all_images = glob.glob(data_dir)
    
    # select a random batch with images
    images_batch = np.random.choice(all_images, size=batch_size)

    low_resolution_images = []
    high_resolution_images = []

    for img in images_batch:
        # take the numpy ndarray from the current image
        img1 = imread(img, as_gray=False, pilmode='RGB')
        img1 = img1.astype(np.float32)
        
        # change the size
        img1_high_resolution = imresize(img1, high_resolution_shape)
        img1_low_resolution = imresize(img1, low_resolution_shape)
        
        # apply the augmentation: random horizontal flip
        if np.random.random() < 0.5:
            img1_high_resolution = np.fliplr(img1_high_resolution)
            img1_low_resolution = np.fliplr(img1_low_resolution)

        high_resolution_images.append(img1_high_resolution)
        low_resolution_images.append(img1_low_resolution)
    
    # convert lists into numpy ndarrays
    return np.array(high_resolution_images), np.array(low_resolution_images)


# ## Saving images
# 

# In[ ]:


def save_images(low_resolution_image, original_image, generated_image, path):

    # save low-resolution, high-resolution(original) and generated high-resolution images into one picture

    fig = plt.figure()
    
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(original_image)
    ax.axis("off")
    ax.set_title("ORIGINAL")
    
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(low_resolution_image)
    ax.axis("off")
    ax.set_title("LOW_RESOLUTION")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(generated_image)
    ax.axis("off")
    ax.set_title("GENERATED")

    plt.savefig(path)


# 
# ***************************
# 

# # Training loop with SRGAN-VGG19
# 

# In[ ]:


for epoch in range(epochs1):

    d_history1 = []
    g_history1 = []
    #print("Epoch:{}".format(epoch))
    
    high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                                          low_resolution_shape=low_resolution_shape,
                                                                          high_resolution_shape=high_resolution_shape)
    
    # normalize the images
    high_resolution_images = high_resolution_images / 127.5 - 1.
    low_resolution_images = low_resolution_images / 127.5 - 1.
    
    # generate high-resolution images from low-resolution images
    generated_high_resolution_images1 = generator1.predict(low_resolution_images)

    
    # generate a batch of true and fake labels 
    real_labels = np.ones((batch_size, 16, 16, 1))
    fake_labels = np.zeros((batch_size, 16, 16, 1))
    
 
    d_loss_real1 = discriminator1.train_on_batch(high_resolution_images, real_labels)
    d_loss_real1 =  np.mean(d_loss_real1)
    d_loss_fake1 = discriminator1.train_on_batch(generated_high_resolution_images1, fake_labels)
    d_loss_fake1 =  np.mean(d_loss_fake1)
    
    # calculate total loss of discriminator as average loss on true and fake labels
    d_loss1 = 0.5 * np.add(d_loss_real1, d_loss_fake1)
    losses1['d_history'].append(d_loss1)
    #print("D_loss_1:", d_loss1)
    
    
    # train the generator
    
    # sample the batch with images
    high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                                    low_resolution_shape=low_resolution_shape,
                                                                    high_resolution_shape=high_resolution_shape)
    
    #  normalize the images
    high_resolution_images = high_resolution_images / 127.5 - 1.
    low_resolution_images = low_resolution_images / 127.5 - 1.
    
    # extract feature maps for true high-resolution images
    image_features1 = VGG19_b3c3.predict(high_resolution_images)

    
    # train the generator

    
    g_loss1 = adversarial_model_vgg19.train_on_batch([low_resolution_images, high_resolution_images],
                                             [real_labels, image_features1])
    
    losses1['g_history'].append(0.5 * (g_loss1[1]))
    #print( "G_loss_1:", 0.5 * (g_loss1[1]) )
    
    
    #calculate the psnr per generated and original image and add it to the list
    
   
    ps2_1 = calc_psnr2(high_resolution_images, generated_high_resolution_images1)
    #print("PSNR_2:", np.mean(ps2_1.numpy()))
    psnr2_1['psnr2_quality'].append(ps2_1)
    
    #calculate the ssim per generated and original image and add it to the list

  
    ss2_1 = calc_ssim2(high_resolution_images, generated_high_resolution_images1)
    #print("SSIM_2:", np.mean(ss2_1.numpy()))
    ssim2_1['ssim2_quality'].append(ss2_1)
    
    
    # save and print image samples
    if (epoch == 50) or (epoch == 100) or (epoch == 150):
        high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                                        low_resolution_shape=low_resolution_shape,
                                                                        high_resolution_shape=high_resolution_shape)
        
        # normalize the images
        high_resolution_images = high_resolution_images / 127.5 - 1.
        low_resolution_images = low_resolution_images / 127.5 - 1.

        generated_images1 = generator1.predict_on_batch(low_resolution_images)

        for index, img in enumerate(generated_images1):
            save_images(low_resolution_images[index], high_resolution_images[index], img,
                        path="/kaggle/working/img2_{}_{}".format(epoch, index))
        

plot_loss1(losses1)
plot_psnr1(psnr2_1)
plot_ssim1(ssim2_1)


# 
# ******************************
# 

# ## Build & compile Discriminator 2
# 

# In[ ]:


discriminator2 = build_discriminator2()
discriminator2.trainable = True
discriminator2.compile(loss='mse', optimizer=common_optimizer2, metrics=['accuracy'])


# ## Build Generator 2
# 

# In[ ]:


generator2 = build_generator2()


# # VGGFace2
# 
# VGGFace refers to a series of models developed for face recognition by members of the Visual Geometry Group at the University of Oxford.
# 
# Their two models for face recognition are: VGGFace (2015) and VGGFace2 (2017).
# 
# Qiong Cao, et al. about VGGFace2 [11]:
# > *In this paper, we introduce a new large-scale face dataset named VGGFace2. The dataset contains 3.31 million images of 9131 subjects, with an average of 362.6 images for each subject. Images are downloaded from Google Image Search and have large variations in pose, age, illumination, ethnicity and profession (e.g. actors, athletes, politicians).*

# # Feature Extractors based on VGGFace2
# 
# 

# In[ ]:


get_ipython().system('pip install git+https://github.com/rcmalli/keras-vggface.git')


# In[ ]:


from keras_vggface.vggface import VGGFace


# In[ ]:


VGGFace2_VGG16_base = VGGFace(model='vgg16')


# In[ ]:


plot_model(VGGFace2_VGG16_base, to_file='VGGFace2_VGG16_base.png', show_shapes=True)
Image(filename='VGGFace2_VGG16_base.png') 


# In[ ]:


"""
def build_VGGFace2_VGG16_b3c1():
    input_shape = (256, 256, 3)
    VGGFace2_VGG16_base.outputs = [VGGFace2_VGG16_base.get_layer('conv3_1').output]
    input_layer = Input(shape=input_shape)
    features = VGGFace2_VGG16_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGGFace2_VGG16_b3c1 = build_VGGFace2_VGG16_b3c1()
VGGFace2_VGG16_b3c1.trainable = False
VGGFace2_VGG16_b3c1.compile(loss='mse', optimizer=common_optimizer1, metrics=['accuracy'])
"""


# In[ ]:


"""
def build_VGGFace2_VGG16_b3c2():
    input_shape = (256, 256, 3)
    VGGFace2_VGG16_base.outputs = [VGGFace2_VGG16_base.get_layer('conv3_2').output]
    input_layer = Input(shape=input_shape)
    features = VGGFace2_VGG16_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGGFace2_VGG16_b3c2 = build_VGGFace2_VGG16_b3c2()
VGGFace2_VGG16_b3c2.trainable = False
VGGFace2_VGG16_b3c2.compile(loss='mse', optimizer=common_optimizer1, metrics=['accuracy'])
"""


# In[ ]:


def build_VGGFace2_VGG16_b3c3():
    input_shape = (256, 256, 3)
    VGGFace2_VGG16_base.outputs = [VGGFace2_VGG16_base.get_layer('conv3_3').output]
    input_layer = Input(shape=input_shape)
    features = VGGFace2_VGG16_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGGFace2_VGG16_b3c3 = build_VGGFace2_VGG16_b3c3()
VGGFace2_VGG16_b3c3.trainable = False
VGGFace2_VGG16_b3c3.compile(loss='mse', optimizer=common_optimizer1, metrics=['accuracy'])


# In[ ]:


"""
def build_VGGFace2_VGG16_b4c1():
    input_shape = (256, 256, 3)
    VGGFace2_VGG16_base.outputs = [VGGFace2_VGG16_base.get_layer('conv4_1').output]
    input_layer = Input(shape=input_shape)
    features = VGGFace2_VGG16_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGGFace2_VGG16_b4c1 = build_VGGFace2_VGG16_b4c1()
VGGFace2_VGG16_b4c1.trainable = False
VGGFace2_VGG16_b4c1.compile(loss='mse', optimizer=common_optimizer1, metrics=['accuracy'])
"""


# In[ ]:


"""
def build_VGGFace2_VGG16_b4c2():
    input_shape = (256, 256, 3)
    VGGFace2_VGG16_base.outputs = [VGGFace2_VGG16_base.get_layer('conv4_2').output]
    input_layer = Input(shape=input_shape)
    features = VGGFace2_VGG16_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGGFace2_VGG16_b4c2 = build_VGGFace2_VGG16_b4c2()
VGGFace2_VGG16_b4c2.trainable = False
VGGFace2_VGG16_b4c2.compile(loss='mse', optimizer=common_optimizer1, metrics=['accuracy'])
"""


# In[ ]:


"""
def build_VGGFace2_VGG16_b4c3():
    input_shape = (256, 256, 3)
    VGGFace2_VGG16_base.outputs = [VGGFace2_VGG16_base.get_layer('conv4_3').output]
    input_layer = Input(shape=input_shape)
    features = VGGFace2_VGG16_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGGFace2_VGG16_b4c3 = build_VGGFace2_VGG16_b4c3()
VGGFace2_VGG16_b4c3.trainable = False
VGGFace2_VGG16_b4c3.compile(loss='mse', optimizer=common_optimizer1, metrics=['accuracy'])
"""


# In[ ]:


"""
def build_VGGFace2_VGG16_b5c1():
    input_shape = (256, 256, 3)
    VGGFace2_VGG16_base.outputs = [VGGFace2_VGG16_base.get_layer('conv5_1').output]
    input_layer = Input(shape=input_shape)
    features = VGGFace2_VGG16_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGGFace2_VGG16_b5c1 = build_VGGFace2_VGG16_b5c1()
VGGFace2_VGG16_b5c1.trainable = False
VGGFace2_VGG16_b5c1.compile(loss='mse', optimizer=common_optimizer1, metrics=['accuracy'])
"""


# In[ ]:


"""
def build_VGGFace2_VGG16_b5c2():
    input_shape = (256, 256, 3)
    VGGFace2_VGG16_base.outputs = [VGGFace2_VGG16_base.get_layer('conv5_2').output]
    input_layer = Input(shape=input_shape)
    features = VGGFace2_VGG16_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGGFace2_VGG16_b5c2 = build_VGGFace2_VGG16_b5c2()
VGGFace2_VGG16_b5c2.trainable = False
VGGFace2_VGG16_b5c2.compile(loss='mse', optimizer=common_optimizer1, metrics=['accuracy'])
"""


# In[ ]:


"""
def build_VGGFace2_VGG16_b5c3():
    input_shape = (256, 256, 3)
    VGGFace2_VGG16_base.outputs = [VGGFace2_VGG16_base.get_layer('conv5_3').output]
    input_layer = Input(shape=input_shape)
    features = VGGFace2_VGG16_base(input_layer)
    model = Model(inputs=[input_layer], outputs=[features])
    return model

VGGFace2_VGG16_b5c3 = build_VGGFace2_VGG16_b5c3()
VGGFace2_VGG16_b5c3.trainable = False
VGGFace2_VGG16_b5c3.compile(loss='mse', optimizer=common_optimizer1, metrics=['accuracy'])
"""


# 
# 
# *************************
# 
# 

# # Experiment - SRGAN-VGGFace2
# 
# 
# 
# 
# Inspired by the fact that we experiment with **digital images of human faces**, let's substitute VGG19 with **VGGFace2** in the Feature Extractor:
# 
# 
# 
# 
# ***************************
# 
# 
# 
# ***************************
# 
# 
# ![](https://github.com/s-gladysh/Super-Resolution-GAN-Experiments/raw/master/12i.jpeg)
# 
# 
# ***************************
# 

# ### Build & compile SRGAN-VGGFace2
# 

# In[ ]:


def build_adversarial_model_vggface_vgg16(generator2, discriminator2, vggface_vgg16):
    
    # input layer for high-resolution images
    input_high_resolution2 = Input(shape=high_resolution_shape)

    # input layer for low-resolution images
    input_low_resolution2 = Input(shape=low_resolution_shape)

    # generate high-resolution images from low-resolution images
    generated_high_resolution_images2 = generator2(input_low_resolution2)

    # extract feature maps from generated images
    features2 = vggface_vgg16(generated_high_resolution_images2)
    
    # make a discriminator non-trainable 
    discriminator2.trainable = False
    discriminator2.compile(loss='mse', optimizer=common_optimizer2, metrics=['accuracy'])

    # discriminator will give us a probability estimation for the generated high-resolution images
    probs2 = discriminator2(generated_high_resolution_images2)

    # create and compile 
    adversarial_model2 = Model([input_low_resolution2, input_high_resolution2], [probs2, features2])
    adversarial_model2.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=common_optimizer2)
    return adversarial_model2


# In[ ]:


adversarial_model_vggface_vgg16 = build_adversarial_model_vggface_vgg16(generator2, discriminator2, VGGFace2_VGG16_b3c3)


# 
# 
# *******************
# 

# # Training loop with SRGAN-VGGFace2
#  

# In[ ]:


for epoch in range(epochs2):

    d_history2 = []
    g_history2 = []

    #print("Epoch:{}".format(epoch))
    
    high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                                          low_resolution_shape=low_resolution_shape,
                                                                          high_resolution_shape=high_resolution_shape)
    
    # normalize the images
    high_resolution_images = high_resolution_images / 127.5 - 1.
    low_resolution_images = low_resolution_images / 127.5 - 1.
    
    # generate high-resolution images from low-resolution images
    generated_high_resolution_images2 = generator2.predict(low_resolution_images)

    
    # generate a batch of true and fake labels 
    real_labels = np.ones((batch_size, 16, 16, 1))
    fake_labels = np.zeros((batch_size, 16, 16, 1))
    
    # train the discriminator on true and fake labels 
    d_loss_real2 = discriminator2.train_on_batch(high_resolution_images, real_labels)
    d_loss_real2 =  np.mean(d_loss_real2)
    d_loss_fake2 = discriminator2.train_on_batch(generated_high_resolution_images2, fake_labels)
    d_loss_fake2 =  np.mean(d_loss_fake2)
    
    # calculate total loss of discriminator as average loss on true and fake labels
    d_loss2 = 0.5 * np.add(d_loss_real2, d_loss_fake2)
    losses2['d_history'].append(d_loss2)
    #print("D_loss_2:", d_loss2)
    

    # train the generator
    
    # sample the batch with images
    high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                                    low_resolution_shape=low_resolution_shape,
                                                                    high_resolution_shape=high_resolution_shape)
    
    #  normalize the images
    high_resolution_images = high_resolution_images / 127.5 - 1.
    low_resolution_images = low_resolution_images / 127.5 - 1.
    
    # extract feature maps for true high-resolution images
    image_features2 = VGGFace2_VGG16_b3c3.predict(high_resolution_images)
    
    gfaces_loss2 = adversarial_model_vggface_vgg16.train_on_batch([low_resolution_images, high_resolution_images],
                                             [real_labels, image_features2])
    
    losses2['g_history'].append(0.5 * (gfaces_loss2[1]))
    #print( "G_loss_2:", 0.5 * (gfaces_loss2[1]) )
    
   
    #calculate the psnr per generated and original image and add it to the list
    
    ps2_2 = calc_psnr2(high_resolution_images, generated_high_resolution_images2)
    #print("PSNR_2:", np.mean(ps2_2.numpy()))
    psnr2_2['psnr2_quality'].append(ps2_2)
    
   
    #calculate the ssim per generated and original image and add it to the list

    ss2_2 = calc_ssim2(high_resolution_images, generated_high_resolution_images2)
    #print("SSIM_1:", np.mean(ss2_1.numpy()))
    ssim2_2['ssim2_quality'].append(ss2_2)
    
   
    # save and print image samples 
    if (epoch == 50) or (epoch == 100) or (epoch == 150):
        high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                                        low_resolution_shape=low_resolution_shape,
                                                                        high_resolution_shape=high_resolution_shape)
        
        # normalize the images
        high_resolution_images = high_resolution_images / 127.5 - 1.
        low_resolution_images = low_resolution_images / 127.5 - 1.

        generated_images2 = generator2.predict_on_batch(low_resolution_images)

        for index, img in enumerate(generated_images2):
            save_images(low_resolution_images[index], high_resolution_images[index], img,
                        path="/kaggle/working/img2_{}_{}".format(epoch, index))
            

plot_loss2(losses2)
plot_psnr2(psnr2_2)
plot_ssim2(ssim2_2)


# 
# ********************************
# 

# ## Save models weights
# 

# In[ ]:


generator1.save_weights("/kaggle/working/generator1.h5")
discriminator1.save_weights("/kaggle/working/discriminator1.h5")


# In[ ]:


generator2.save_weights("/kaggle/working/generator2.h5")
discriminator2.save_weights("/kaggle/working/discriminator2.h5")


# 
# 
# **************************
# 
# 
# ## SRGAN-VGG19 + SRGAN-VGGFace2 plots

# In[ ]:



plot_loss12(losses1, losses2)
plot_psnr12(psnr2_1, psnr2_2)
plot_ssim12(ssim2_1, ssim2_2)


# 
# ***************
# 

# # References:
# 
# *************
# 
# 
# [1] A Deep Journey into Super-resolution: A Survey
# 
# https://arxiv.org/pdf/1904.07523.pdf
# 
# 
# ***************
# 
# 
# [2] Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
# Network 
# 
# https://arxiv.org/pdf/1609.04802.pdf
# 
# 
# ************
# 
# 
# [3] Generative Adversarial Networks Projects by Kailash Ahirwar
# 
# 
# https://github.com/PacktPublishing/Generative-Adversarial-Networks-Projects
# 
# 
# ****************************
# 
# 
# [4] Generative Adversarial Nets 
# 
# https://arxiv.org/pdf/1406.2661.pdf
# 
# 
# *******************************************************
# 
# [5] Perceptual Losses for Real-Time Style Transfer and Super-Resolution
# 
# https://arxiv.org/pdf/1603.08155.pdf
# 
# *******************************************************
# 
# 
# [6] "Labeled Faces in the Wild" dataset
# 
# http://vis-www.cs.umass.edu/lfw/
# 
# 
# ****************
# 
# [7]  Deep Residual Learning for Image Recognition 
# 
# https://arxiv.org/pdf/1512.03385.pdf
# 
# *****************
# 
# 
# [8] Very Deep Convolutional Networks for Large-Scale Image Recognition
# 
# https://arxiv.org/abs/1409.1556
# 
# ************************
# 
# 
# [9] ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks 
# 
# https://arxiv.org/pdf/1809.00219.pdf
# 
# *******************
# 
# [10] The relativistic discriminator: a key element missing from standard GAN
# 
# https://arxiv.org/pdf/1807.00734.pdf
# 
# 
# ********************
# 
# 
# [11] VGGFace2: A dataset for recognising faces across pose and age
# 
# https://arxiv.org/pdf/1710.08092.pdf 
# 
# *********************
# 
# 
# [12] VGGFace implementation with Keras Framework 
# 
# https://github.com/rcmalli/keras-vggface 
# 
# *********************
