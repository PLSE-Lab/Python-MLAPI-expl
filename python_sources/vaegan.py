#!/usr/bin/env python
# coding: utf-8

# In[20]:


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


# **
# 1.
# A. Download the Celeb A dataset ZIP, and the images in "img_align_celeba" are the dataset. Use a Keras Custom Data Generator. Then load attribute data in the file "list_attr_celeba.csv" into  a pandas dataframe. **

# In[21]:


import requests
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import imread
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Input, Reshape, UpSampling2D, InputLayer, Lambda, ZeroPadding2D, Cropping2D, Conv2DTranspose, BatchNormalization
from keras.utils import np_utils, to_categorical
from keras.losses import binary_crossentropy
from keras import backend as K,objectives
from keras.losses import mse, binary_crossentropy
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomNormal
import random

from IPython.core.display import HTML


# In[22]:


df = pd.read_csv('../input/list_attr_celeba.csv')


# In[23]:


df.head()


# **B. Plot 3 random images from dataset, ascertain accuracy of attributes.**

# In[24]:


ran = np.random.randint(1, 200000, size = 3)
ran


# In[25]:


fig = plt.figure(figsize = (8,8))

for i in range(0, len(ran)):
    ax = fig.add_subplot(1,3, i+1)
    number="%06d.jpg" % ran[i]
    img = imread('../input/img_align_celeba/img_align_celeba/' + number)
    ax.set_xlabel(number )
    print(df[df['image_id']==number])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.imshow(img)
    
plt.show()


# In[26]:


get_ipython().system('ls')


# In[27]:


#remove any existent file
get_ipython().system('rm -rf vae')
get_ipython().system('rm celebB.hdf5')


# In[28]:


#clone the project here
get_ipython().system('git clone https://github.com/rahul2240/vae-dcgan-celeb')
    
#rename it to vae
get_ipython().system('mv ./vae-dcgan-celeb ./vae')


# In[29]:


# run create celeb if the dataset is present in ../input/img_align_celeba/img_align_celeba otherwise change it
get_ipython().system('python vae/datasets/create_celeba.py')


# In[30]:


# remove any existent folder nad create a new one for result
get_ipython().system('rm -rf output')
get_ipython().system('mkdir output')


# **C. Create + compile Convolutional Variational Autoencoder Model (with encoder and decoder) with this face data.**
# 
# **D. Train on these images.**

# In[31]:


import os
import sys
import math
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')

# vae in folder in which files are located
from vae.models import VAE, DCGAN
from vae.datasets import load_data


models = {
    'vae': VAE,
    'dcgan': DCGAN
}

def main():
    # Parsing arguments
    
    datasets = load_data('celebB.hdf5')

    # Construct model
    
#swap here for vae and dcgan
    model = models['vae'](
        input_shape=datasets.shape[1:],
        z_dims=256,
        output='output'
    )


    # Training loop
    datasets = datasets.images * 2.0 - 1.0
    print(len(datasets))
    samples = np.random.normal(size=(100, 256)).astype(np.float32)
    model.main_loop(datasets, samples,
        epochs=200,
        batchsize=50,
        reporter=['loss', 'g_loss', 'd_loss', 'g_acc', 'd_acc'])

main()


# In[32]:


get_ipython().system('ls output/vae/results/')


# **Show the output**

# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')
img = imread('./output/vae/results/epoch_0200_batch_1000.png')
plt.figure(figsize=(10, 10))
plt.imshow(img)


# **H. Create + compile DCGAN model with same data.  Then print summaries for 1. discriminator model 2. generator model**
# 
# **I. Train model on same celeb dataset.**
# 
# **J. Generate + visualise 15 celeb faces.**

# In[ ]:


import os
import sys
import math
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')

# vae in folder in which files are located
from vae.models import VAE, DCGAN
from vae.datasets import load_data


models = {
    'vae': VAE,
    'dcgan': DCGAN
}

def main():
    # Parsing arguments
    
    datasets = load_data('celebB.hdf5')

    # Construct model
    
#swap here for vae and dcgan
    model = models['dcgan'](
        input_shape=datasets.shape[1:],
        z_dims=256,
        output='output'
    )


    # Training loop
    datasets = datasets.images * 2.0 - 1.0
    print(len(datasets))
    samples = np.random.normal(size=(100, 256)).astype(np.float32)
    model.main_loop(datasets, samples,
        epochs=200,
        batchsize=50,
        reporter=['loss', 'g_loss', 'd_loss', 'g_acc', 'd_acc'])

main()

