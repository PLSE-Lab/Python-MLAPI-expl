#!/usr/bin/env python
# coding: utf-8

# ### Short overview
# 
# * 8 classes of cancer tissues
# * multiclass classification
# * Kather_texture_2016_image_tiles_5000 
#     * 150 x 150 pixel in size
#     * 5000 samples

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Well actually I'm a bit confused about the meanings of all files. Let's dive into each file and folder to obtain an impression of what we can find there: 

# ### Medium - 28 x 28 MNIST like RGB images

# In[ ]:


medium_colored_data = pd.read_csv('../input/hmnist_28_28_RGB.csv')
medium_colored_data.head()


# Ah, ok, we can see that each sample belongs to one image with a size of 28x28x3 (width times hight times color channel). This yields 2352 pixels in total per image. And we have one label that holds the class target. One question remains: How are the pixels ordered? Are three consequtive columns given by weight, height and color? For me it's not very intuitive even if one reads the intro kernels. Hmm... 

# In[ ]:


medium_colored_data.shape


# Ok, 5000 samples as given by the dataset description.

# In[ ]:


example = medium_colored_data.drop("label", axis=1).values[0]
to_show = example.reshape((28,28,3))

fig, ax = plt.subplots(1,4,figsize=(20,5))
for channel in range(3):
    ax[channel].imshow(to_show[:,:,channel], cmap="gray")
    ax[channel].set_title("Channel {}".format(channel+1))
    ax[channel].set_xlabel("Width")
    ax[channel].set_ylabel("Height")
ax[3].imshow(to_show)
ax[3].set_title("All channels together")
ax[3].set_xlabel("Width")
ax[3].set_ylabel("Height")


# Ok, this looks fine. Each channel has the same spatial structure. Our reshaping worked well, but even though we don't know the order yet. Let's try to untersand, what reshape does with our flattened image row:

# In[ ]:


order_example = np.arange(0,12)
order_example


# In[ ]:


show_order = order_example.reshape(2,2,3)
print(show_order[:,:,0])
print(show_order[:,:,1])
print(show_order[:,:,2])


# Ok, we can see that the first 3 numbers are all of the same spatial coordinate but with different colors. Hence the first order-quantity is color. After that the width of the images is filled up and than the heigth. Hence the order is: color, width, height. And this should be true for our dataframe as well.

# ### Medium - 28 x 28 MNIST like L images

# In[ ]:


medium_data = pd.read_csv('../input/hmnist_28_28_L.csv')
medium_data.head()


# Ok, same but with less pixels.

# In[ ]:


print(medium_data.shape)


# Ah, these are grayscaled images. Cool, this way we can compare the image we obtained above with its grayscaled counterpart. 

# In[ ]:


def show_inarow(data, row_shape):
    example = data.drop("label", axis=1).values[0:4]
    to_show = example.reshape(row_shape)
    fig, ax = plt.subplots(1,4,figsize=(20,5))
    for image_example in range(4):
        ax[image_example].imshow(to_show[image_example,:,:], cmap="gray")
        ax[image_example].set_title("Grayscaled image {}".format(image_example))
        ax[image_example].set_xlabel("Widht")
        ax[image_example].set_ylabel("Height")


# In[ ]:


show_inarow(medium_data, (4,28,28))


# Great! The first image is the same from above without color. :-) This medium resolution could still be too low to gain nice classification results. 

# ### Big - 64 x 64 MNIST like L images
# 
# ....Why aren't here colored ones as well?.... 

# In[ ]:


big_data = pd.read_csv("../input/hmnist_64_64_L.csv")
big_data.head()


# In[ ]:


big_data.shape


# Perhaps this rosolution is still better for us than 28x28 even though colors are missing. 

# In[ ]:


show_inarow(big_data, (4,64,64))


# Uhh yes, this is far better! We can see more details of the tissues and it could be worth it to try classification with them. 

# ### Small - 8 x 8 MNIST like RGB and L images
# 
# Just for completeness we should look at the smallest images with size 8x8 even though I assume that they are far too small to gain deep insights. But perhaps they are useful somehow... we will see.

# In[ ]:


small_data = pd.read_csv("../input/hmnist_8_8_L.csv")
small_colored_data = pd.read_csv("../input/hmnist_8_8_RGB.csv")
small_data.head()


# In[ ]:


print(small_data.shape)
print(small_colored_data.shape)


# Let's take a look at the grayscaled ones first:

# In[ ]:


show_inarow(small_data, (4,8,8))


# Ok :-D This could be everything...

# In[ ]:


show_inarow(small_colored_data, (4,8,8,3))


# This doesn't look better. Ok, 8x8 is not worth it.

# ### Peak at the image folder 150x150
# 
# In the images folder *'kather_texture_2016_image_tiles_5000'* we should find 5000 images of size 150x150 pixels. Let's have a look at them to compare with our candidate of 64x62 grayscaled images. Within this folder (one step further) we should see 8 different directories that correspond to the 8 different class labels of cancer.

# In[ ]:


from os import listdir

classes_dir = listdir("../input/kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000")
classes_dir


# :-) That's nice! To gain an impression it's sufficient to look at only some examples. 

# In[ ]:


files = listdir("../input/kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/01_TUMOR")
for n in range(5):
    print(files[n])


# Ok let's wirte a small method that collects examples - one per class folder: 

# In[ ]:


from scipy.misc import imread

def show_set(basepath, classes_dir, num_file):
    fig, ax = plt.subplots(2,4,figsize=(20,10))
    for n in range(4):
        for m in range(2):
            class_idx = m * 4 + n
            path = basepath + classes_dir[class_idx] + "/"
            files = listdir(path)
            image = imread(path + files[num_file])
            ax[m,n].imshow(image)
            ax[m,n].set_title(classes_dir[class_idx])


# You can browse through the files by setting different values for num_file. Currently you will obtain the first file in each class folder:

# In[ ]:


basepath = "../input/kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/"
show_set(basepath, classes_dir, num_file=0)


# What does empty and adipose classes mean?!

# ### Peek at the large images

# In[ ]:


basepath = "../input/kather_texture_2016_larger_images_10/Kather_texture_2016_larger_images_10/"
files = listdir(basepath)
files


# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].imshow(imread(basepath + files[0]))
ax[1].imshow(imread(basepath + files[1]))


# ## Take-Away
# 
# * All images with a size higher or equal than 64x64 could be a good choice for deep learning.
# * 8x8 images are far too small to obtain meaningful insights. 
# * A good starting point could be 64x64 grayscale, especially if one is limited to hardware resources. 
# * There is an empty and an adipose class that is somehow strange and we have to find out their meanings. 
