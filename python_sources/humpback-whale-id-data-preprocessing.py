#!/usr/bin/env python
# coding: utf-8

# # Acknowledgments
# This notebook is a fork of *"Humpback Whale ID: Data and Aug Exploration"* by** Lex Toumbourou**, and contains also some parts coppied from other kernels/notebooks. Unfortunatelly, I did not written them down, so I cannot provide you a list with proper citations. Sorry!
# 
# If you happen to identify part of your work in this notebook, please, let me know, and I will add a note about it to this section.
# 
# # Intro
# This notebook provides/showcases a preprocessing of the input data for the Humpback Whale Identification Challenge. It firstly examine the data, and then it provide utilities to ease the augumentation.

# ## Important imports

# In[ ]:


import sys
from collections import Counter
import random
import itertools

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import skimage.filters


from sklearn.model_selection import train_test_split

from keras.preprocessing.image import (
    random_rotation, random_shift, random_shear, random_zoom,
    random_channel_shift, img_to_array)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


np.random.seed(42) # here, this is set to a constant for a reproducible results. You may drop this line to gain proper randomness on every run.
INPUT_DIR = '../input/whale-categorization-playground/'


# ## General utility functions

# In[ ]:


def plot_images_for_filenames(filenames, labels, rows=4):
    '''
    Loads the images from the file paths provided in filenames,
    and prints them with the provided labels.
    '''
    imgs = [plt.imread(f'{INPUT_DIR}/train/{filename}') for filename in filenames]
    
    return plot_images(imgs, labels, rows)
    
        
def plot_images(imgs, labels, rows=4):
    '''
    Plots the provided images with the labels in a grid with the provided number of rows.
    '''
    # Set figure to 13 inches x 8 inches
    figure = plt.figure(figsize=(13, 8))

    cols = len(imgs) // rows + 1

    for i in range(len(imgs)):
        subplot = figure.add_subplot(rows, cols, i + 1)
        subplot.axis('Off')
        if labels:
            subplot.set_title(labels[i], fontsize=16)
        plt.imshow(imgs[i], cmap='gray')
        
def is_grey_scale(img_path):
    """
    This checks whether the image is percievably grayscale = it returns true not only for
    single channel images, but also for multichannel (RGB) grayscale images.
    Thanks to https://stackoverflow.com/questions/23660929/how-to-check-whether-a-jpeg-image-is-color-or-gray-scale-using-only-python-stdli
    """
    im = Image.open(img_path)
    if len(im.getbands()) == 1:
        return True
    im = im.convert('RGB')
    w,h = im.size
    for i in range(w):
        for j in range(h):
            r,g,b = im.getpixel((i,j))
            if r != g != b: return False
    return True

def random_greyscale(img, p):
    '''
    Converts image to grayscale with the given probability p.
    The returned image has always three channels.
    '''
    # check whether image is not grayscale already
    if len(img.shape) == 2 or img.shape[2] == 1:
        return np.repeat(img[:, :, np.newaxis], 3, axis=2)
    
    # colour image - convert it with the given probability.
    if random.random() < p:
        return np.dot(img[...,:3], [0.299, 0.587, 0.114])
    # otherwise just return original image
    return img


def random_flip(img, p):
    '''
    Randomly flips the image from left-to-right, given the probability p.
    '''
    if random.random() < p:
        return np.flip(img, 1)
    return img

def random_blur(img, p, sigma=1.37):
    '''
    Randomly blurs the image with gaussian kernel with sigma, given the probability p. 
    '''
    if random.random() < p:
        return skimage.filters.gaussian(img / 255.0, sigma=sigma, multichannel=len(img.shape) == 3) * 255
    return img

def augmentation_pipeline(img_arr):
    '''All augumentations together'''
    img_arr = random_rotation(img_arr, 18, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = random_shear(img_arr, intensity=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = random_zoom(img_arr, zoom_range=(0.7, 1.4), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') # we do not want to zoom out, as the net may learn the artifacts caused by the filled areas.
    img_arr = random_greyscale(img_arr, 0.4)
    img_arr = random_blur(img_arr, 0.33)
    img_arr = random_flip(img_arr, 0.5)

    return img_arr


# ## Dataset-specific utility functions

# In[ ]:


IMAGE_SIZE = (160,120) # width, height
NUM_OF_AUGMENTATIONS = 5

def makeNewWhalesUnique(trainDataFrame):
    '''
    Renames all new_whale whales to new_whale_<Id>, so that every
    new whale is unique.
    Note: I am not sure wheter every new whale is unique.
    '''
    for i in range(len(trainDataFrame['Id'])):
        if trainDataFrame['Id'][i] == 'new_whale':
            trainDataFrame['Id'][i] = ('new_whale_%d' % i)
    return trainDataFrame
            
def loadImageAndAugument(imgPath, imageSize=IMAGE_SIZE, numOfAugumentations=NUM_OF_AUGMENTATIONS, inputDir=INPUT_DIR):
    '''Loads image, randomly auguments it and returns list of images suitable for use in Resnet50'''
    if not imgPath.startswith(inputDir):
        imgPath = inputDir + '/train/' + imgPath
    inputImage = Image.open(imgPath).resize(imageSize).convert('RGB')
    inputImageArray = img_to_array(inputImage)
    result = [inputImageArray / 255]
    for i in range(numOfAugumentations):
        result.append(augmentation_pipeline(inputImageArray) / 255)
    return result

def loadBatch(imagePaths, labels, startIdx=0, numOfImages=None):
    if numOfImages is None:
        numOfImages = imagePaths.count() - startIdx
    loadedLabels = []
    loadedImages = []
    #for index, imgPath in imagePaths.iteritems()[startIdx:min(startIdx+numOfImages, imagePaths.count())]:
    for index, imgPath in itertools.islice(imagePaths.iteritems(), startIdx, min(startIdx+numOfImages, imagePaths.count())):
        print(index, imgPath)
        imgArrays = loadImageAndAugument(imgPath)
        for img in imgArrays:
            loadedLabels.append(labels[index])
            loadedImages.append(img)
    return (loadedImages, loadedLabels)


def laodAndPrepareDataset(datasetPath=f'{INPUT_DIR}/train.csv', train_size=0.8, random_state=None):
    '''
    Loads the dataset and preprocess it so that it can be used to load batches for training.
    '''
    if random_state is None:
        random_state = random.randrange(2**32 - 1)
    # load data to Panda DataFrame
    train_df = pd.read_csv(datasetPath)
    # Make things unique
    train_df = makeNewWhalesUnique(train_df)
    # split to train/validation subsets in the following order:
    # trainInputImagePaths, validationInputImagePaths, trainInputLabels, validationInputLabels
    return  train_test_split(train_df['Image'], 
                             train_df['Id'],
                             train_size=train_size, 
                             test_size=1.0 - train_size,
                             random_state=random_state)


# ## Exploring the dataset

# In[ ]:


train_df = pd.read_csv('../input/whale-categorization-playground/train.csv')
train_df.head()


# Let's plot a couple of images at random.

# In[ ]:


rand_rows = train_df.sample(frac=1.)[:20]
imgs = list(rand_rows['Image'])
labels = list(rand_rows['Id'])

plot_images_for_filenames(imgs, labels)


# The competition states that it's hard because: "there are only a few examples for each of 3,000+ whale ids", so let's take a look at the breakdown of number of image per category.

# In[ ]:


num_categories = len(train_df['Id'].unique())
     
print(f'Number of categories: {num_categories}')


# There appear to be too many categories to graph count by category, so let's instead graph the number of categories by the number of images in the category.

# In[ ]:


size_buckets = Counter(train_df['Id'].value_counts().values)


# In[ ]:


plt.figure(figsize=(10, 6))

plt.bar(range(len(size_buckets)), list(size_buckets.values())[::-1], align='center')
plt.xticks(range(len(size_buckets)), list(size_buckets.keys())[::-1])
plt.title("Num of categories by images in the training set")

plt.show()


# As we can see, the vast majority of classes only have a single image in them. This is going to make predictions very difficult for most conventional image classification models.

# In[ ]:


train_df['Id'].value_counts().head(3)


# In[ ]:


total = len(train_df['Id'])
print(f'Total images in training set {total}')


# New whale is the biggest category with 810, followed by `w_1287fbc`. New whale, I believe, is any whale that isn't in scientist's database. Since we can pick 5 potential labels per id, it's probably going to make sense to always include new_whale in our prediction set, since there's always an 8.2% change that's the right one. But, to have training nice, we should rename each of its instance to be unique:

# In[ ]:


train_df = makeNewWhalesUnique(train_df)
# check whether the counts are still that bad...
train_df['Id'].value_counts().head(3)


# 
# Let's take a look at one of the classes, to get a sense what flute looks like from the same whale.

# In[ ]:


w_1287fbc = train_df[train_df['Id'] == 'w_1287fbc']
plot_images_for_filenames(list(w_1287fbc['Image']), None, rows=9)


# In[ ]:


w_98baff9 = train_df[train_df['Id'] == 'w_98baff9']
plot_images_for_filenames(list(w_98baff9['Image']), None, rows=9)


# It's very difficult to build a validation set when most classes only have 1 image, so my thinking is to perform some aggressive data augmentation on the classes with < 10 images before creating a train/validation split. Let's take a look at a few examples of whales with only one example.

# In[ ]:


one_image_ids = train_df['Id'].value_counts().tail(8).keys()
one_image_filenames = []
labels = []
for i in one_image_ids:
    one_image_filenames.extend(list(train_df[train_df['Id'] == i]['Image']))
    labels.append(i)
    
plot_images_for_filenames(one_image_filenames, labels, rows=3)


# From these small sample sizes, it seems like > 50% of images are black and white, suggesting that a good initial augementation might be to just convert colour images to greyscale and add to the training set. Let's confirm that by looking at a sample of the images.

# In[ ]:


is_grey = [is_grey_scale(f'{INPUT_DIR}/train/{i}') for i in train_df['Image'].sample(frac=0.1)]
grey_perc = round(sum([i for i in is_grey]) / len([i for i in is_grey]) * 100, 2)
print(f"% of grey images: {grey_perc}")


# It might also be worth capturing the size of the images so we can get a sense of what we're dealing with.

# In[ ]:


img_sizes = Counter([Image.open(f'{INPUT_DIR}/train/{i}').size for i in train_df['Image']])

size, freq = zip(*Counter({i: v for i, v in img_sizes.items() if v > 1}).most_common(20))

plt.figure(figsize=(10, 6))

plt.bar(range(len(freq)), list(freq), align='center')
plt.xticks(range(len(size)), list(size), rotation=70)
plt.title("Image size frequencies (where freq > 1)")

plt.show()


# ## Data Augmentation

# In[ ]:


img = Image.open(f'{INPUT_DIR}/train/ff38054f.jpg')
img_arr = img_to_array(img)
plt.imshow(img)


# ### Random rotation

# In[ ]:


imgs = [
    random_rotation(img_arr, 30, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') / 255
    for _ in range(5)]
plot_images(imgs, None, rows=1)


# ### Random shift

# In[ ]:


imgs = [
    random_shift(img_arr, wrg=0.1, hrg=0.3, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') / 255
    for _ in range(5)]
plot_images(imgs, None, rows=1)


# ### Random shear

# In[ ]:


imgs = [
    random_shear(img_arr, intensity=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') / 255
    for _ in range(5)]
plot_images(imgs, None, rows=1)


# ### Random zoom

# In[ ]:


imgs = [
    random_zoom(img_arr, zoom_range=(1.5, 0.7), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') / 255
    for _ in range(5)]
plot_images(imgs, None, rows=1)


# ### Grey scale
# 
# We want to ensure that all colour images also have a grey scale version.

# In[ ]:


imgs = [random_greyscale(img_arr / 255, 0.5)
        for _ in range(5)]

plot_images(imgs, None, rows=1)


# ### All together
# 
# Going to create an augmentation pipeline which will combine all the augs for *a* single predictions. We are not giving zoom too huge range to reduce weid lines at the side of the image.

# In[ ]:


imgs = [augmentation_pipeline(img_arr) / 255 for _ in range(20)]
plot_images(imgs, None, rows=4)


# In[ ]:


imgs = loadImageAndAugument('70238365.jpg')
plot_images(imgs, None, rows=1)


# ### Prepare the data
# 
# Prepare the data for training (and validation) process.

# In[ ]:


trainInputImagePaths, validationInputImagePaths, trainInputLabels, validationInputLabels = laodAndPrepareDataset()


# We have loaded and split the data. Seems ok... Now, lets generate some of  the data, and store it in array.

# In[ ]:


#example:
trainAugumentedImages, trainAugumentedLabels = loadBatch(trainInputImagePaths, trainInputLabels, 0, 4)


# ...and see how they look like:

# In[ ]:


plot_images(trainAugumentedImages, trainAugumentedLabels, rows=4)

