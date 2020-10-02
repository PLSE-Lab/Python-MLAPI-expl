#!/usr/bin/env python
# coding: utf-8

# # <a id="2">Load packages</a>

# In[ ]:


import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.utils import plot_model
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Parameters

# In[ ]:


IMG_ROWS = 28
IMG_COLS = 28
NUM_CLASSES = 10
PATH="../input/"
print(os.listdir(PATH))


# # <a id="3">Read the data</a>
# 
# We will read the two data files containing the 10-class data, KMNIST, similar to MNIST.
# 
# There are 10 different classes of images, one class for each number between 0 and 9. 
# 
# Image dimmensions are **28**x**28**.   
# 
# The train set and test set are given in two separate numpy arrays.   
# 
# We are also reading the labels for train set.
# 
# Aditionally, we will read the character class map for KMNIST, so that we can display the actual characters corresponding to the labels.
# 

# In[ ]:


train_images = np.load(os.path.join(PATH,'train-imgs.npz'))['arr_0']
test_images = np.load(os.path.join(PATH,'test-imgs.npz'))['arr_0']
train_labels = np.load(os.path.join(PATH,'train-labels.npz'))['arr_0']


# In[ ]:


char_df = pd.read_csv(os.path.join(PATH,'classmap.csv'), encoding = 'utf-8')


# # <a id="4">Convert array to image</a>

# Plot and save images.

# In[ ]:


def plot_save_sample_images_data(images, labels):
    plt.figure(figsize=(12,12))
    for i in tqdm_notebook(range(10)):
        imgs = images[np.where(labels == i)]
        lbls = labels[np.where(labels == i)]
        for j in range(10):
            plt.subplot(10,10,i*10+j+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(imgs[j], cmap=plt.cm.binary)
            plt.xlabel(lbls[j])
            img = Image.fromarray(imgs[j])
            img_name = "img_{}_{}.png".format(i,j)
            img.save(img_name)


# In[ ]:


plot_save_sample_images_data(train_images, train_labels)

