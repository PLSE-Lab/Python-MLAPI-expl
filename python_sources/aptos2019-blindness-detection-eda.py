#!/usr/bin/env python
# coding: utf-8

# ![image](https://github.com/Lexie88rus/APTOS2019/raw/master/assets/cover_image.png)

# # APTOS 2019: Blindness Detection EDA

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from random import randrange

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns


# In[ ]:


get_ipython().system(' pip install albumentations')


# In[ ]:


from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, PadIfNeeded
)


# ## Introduction

# [Diabetic retinopathy](https://en.wikipedia.org/wiki/Diabetic_retinopathy) is the leading cause of blindness among working aged adults. Millions of people suffer from this decease.
# People with diabetes can have an eye disease called diabetic retinopathy. This is when high blood sugar levels cause damage to blood vessels in the retina. These blood vessels can swell and leak. Or they can close, stopping blood from passing through. Sometimes abnormal new blood vessels grow on the retina. All of these changes can lead to blindness.
# 
# ### Stages of Diabetic Eye Disease
# 
# * __NPDR (non-proliferative diabetic retinopathy)__: With NPDR, tiny blood vessels leak, making the [retina](https://en.wikipedia.org/wiki/Retina) swell. When the [macula](https://en.wikipedia.org/wiki/Macula_of_retina) swells, it is called macular edema. This is the most common reason why people with diabetes lose their vision. Also with NPDR, blood vessels in the retina can close off. This is called macular ischemia. When that happens, blood cannot reach the macula. Sometimes tiny particles called [exudates](https://en.wikipedia.org/wiki/Exudate) can form in the retina. These can affect vision too.
# 
# * __PDR (proliferative diabetic retinopathy)__: PDR is the more advanced stage of diabetic eye disease. It happens when the retina starts growing new blood vessels. This is called neovascularization. These fragile new vessels often bleed into the [vitreous](https://en.wikipedia.org/wiki/Vitreous_body). If they only bleed a little, you might see a few dark floaters. If they bleed a lot, it might block all vision. These new blood vessels can form [scar tissue](https://en.wikipedia.org/wiki/Scar). Scar tissue can cause problems with the macula or lead to a [detached retina](https://en.wikipedia.org/wiki/Retinal_detachment). PDR is very serious, and can steal both your [central](https://www.medicinenet.com/script/main/art.asp?articlekey=8544) and [peripheral (side) vision](https://www.medicinenet.com/script/main/art.asp?articlekey=10638).
# 
# _[Source](https://www.aao.org/eye-health/diseases/what-is-diabetic-retinopathy)_

# ## Load Data

# Load training and testing csv files containing image filenames and corresponding labels (only for training set):

# In[ ]:


# load csv files with labels as pandas dataframes
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


# find out the number of images in test ans train sets
print('Number of images in training set is {}'.format(len(train)))
print('Number of images in test set is {}'.format(len(test)))


# In[ ]:


# Plot pie chart
labels = 'Train', 'Test'
sizes = len(train), len(test)

fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')

plt.title('Train and Test sets')
plt.show()


# Both training and testing datasets are not too large.
# 
# Training dataset is about 3 times greater than the testing dataset.

# ## Analyze Train Set Labels

# Plot pie chart with percentage of images of each diabetic retinopathy severity condition:

# In[ ]:


# Plot pie chart
labels = 'No DR', 'Moderate', 'Mild', 'Proliferative DR', 'Severe'
sizes = train.diagnosis.value_counts()

fig1, ax1 = plt.subplots(figsize=(10,7))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')

plt.title('Diabetic retinopathy condition labels')
plt.show()


# We can see that the training dataset is __very imbalanced__. There are ten times more images with no DR than images with the severe DR condition.
# 
# __Data augmentation__ is required to perform the classification.

# ## Visualize Sample Images

# Let's plot [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) images from the training set of different conditions:

# In[ ]:


# define paths to train and test images
TRAIN_IMG_PATH = "../input/train_images/"
TEST_IMG_PATH = "../input/test_images/"

# function to plot a grid of images
def view_fundus_images(images, title = ''):
    """
    Function to plot grid with several examples of fundus images.
    INPUT:
        train - array with filenames for images and condition labels

    OUTPUT: None
    """
    width = 5
    height = 2
    fig, axs = plt.subplots(height, width, figsize=(15,5))
    
    for im in range(0, height * width):
        # open image
        image = Image.open(os.path.join(TRAIN_IMG_PATH,images[im] + '.png'))
        i = im // width
        j = im % width
        axs[i,j].imshow(image) #plot the data
        axs[i,j].axis('off')

    # set suptitle
    plt.suptitle(title)
    plt.show()


# In[ ]:


view_fundus_images(train[train['diagnosis'] == 0][:10].id_code.values, title = 'Images without DR')


# In[ ]:


view_fundus_images(train[train['diagnosis'] == 1][:10].id_code.values, title = 'Images with Mild condition')


# In[ ]:


view_fundus_images(train[train['diagnosis'] == 2][:10].id_code.values, title = 'Images with Moderate condition')


# In[ ]:


view_fundus_images(train[train['diagnosis'] == 3][:10].id_code.values, title = 'Images with Severe condition')


# In[ ]:


view_fundus_images(train[train['diagnosis'] == 4][:10].id_code.values, title = 'Images with Proliferative DR')


# Just glancing through various images we can say:
# * Images are of different sizes. Height and width ratio varies. That is why __image cropping or padding is required__.
# * Pictures are taken with various scales. __Random cropping__ augmentation is required.
# * Lighting and colors vary greately. Augmentations which __adjust brightness and color scales__ are required.

# ## Analyze Image Sizes

# Plot histograms for image sizes (used code from [this kernel](https://www.kaggle.com/chewzy/eda-weird-images-with-new-updates) for the analysis):

# In[ ]:


def get_image_sizes(df, train = True):
    '''
    Function to get sizes of images from test and train sets.
    INPUT:
        df - dataframe containing image filenames
        train - indicates whether we are getting sizes of images from train or test set
    '''
    if train:
        path = TRAIN_IMG_PATH
    else:
        path = TEST_IMG_PATH
        
    widths = []
    heights = []
    
    images = df.id_code
    
    max_im = Image.open(os.path.join(path, images[0] + '.png'))
    min_im = Image.open(os.path.join(path, images[0] + '.png'))
        
    for im in range(0, len(images)):
        image = Image.open(os.path.join(path, images[im] + '.png'))
        width, height = image.size
        
        if len(widths) > 0:
            if width > max(widths):
                max_im = image

            if width < min(widths):
                min_im = image

        widths.append(width)
        heights.append(height)
        
    return widths, heights, max_im, min_im


# In[ ]:


# get sizes of images from test and train sets
train_widths, train_heights, max_train, min_train = get_image_sizes(train, train = True)
test_widths, test_heights, max_test, min_test = get_image_sizes(test, train = False)


# In[ ]:


print('Maximum width for training set is {}'.format(max(train_widths)))
print('Minimum width for training set is {}'.format(min(train_widths)))
print('Maximum height for training set is {}'.format(max(train_heights)))
print('Minimum height for training set is {}'.format(min(train_heights)))


# In[ ]:


print('Maximum width for test set is {}'.format(max(test_widths)))
print('Minimum width for test set is {}'.format(min(test_widths)))
print('Maximum height for test set is {}'.format(max(test_heights)))
print('Minimum height for test set is {}'.format(min(test_heights)))


# In[ ]:


# Plot Histograms and KDE plots for images from the training set
# Source: https://www.kaggle.com/chewzy/eda-weird-images-with-new-updates
plt.figure(figsize=(14,6))
plt.subplot(121)
sns.distplot(train_widths, kde=False, label='Train Width')
sns.distplot(train_heights, kde=False, label='Train Height')
plt.legend()
plt.title('Training Image Dimension Histogram', fontsize=15)

plt.subplot(122)
sns.kdeplot(train_widths, label='Train Width')
sns.kdeplot(train_heights, label='Train Height')
plt.legend()
plt.title('Train Image Dimension KDE Plot', fontsize=15)

plt.tight_layout()
plt.show()


# In[ ]:


# Plot Histograms and KDE plots for images from the test set
# Source: https://www.kaggle.com/chewzy/eda-weird-images-with-new-updates
plt.figure(figsize=(14,6))
plt.subplot(121)
sns.distplot(test_widths, kde=False, label='Test Width')
sns.distplot(test_heights, kde=False, label='Test Height')
plt.legend()
plt.title('Test Image Dimension Histogram', fontsize=15)

plt.subplot(122)
sns.kdeplot(test_widths, label='Test Width')
sns.kdeplot(test_heights, label='Test Height')
plt.legend()
plt.title('Test Image Dimension KDE Plot', fontsize=15)

plt.tight_layout()
plt.show()


# We see that we have __very different distributions of image sizes__ for train and test datasets.

# ### Plot largest and smallest images

# Let's look and the largest and the smallest images from both sets.
# 
# Image with the largest width from training set:

# In[ ]:


plt.axis('off')
plt.imshow(max_train) #plot the data


# Image with the smallest width from training set:

# In[ ]:


plt.axis('off')
plt.imshow(min_train) #plot the data


# Image with the largest width from test set:

# In[ ]:


plt.axis('off')
plt.imshow(max_test) #plot the data


# Image with the smallest width from training set:

# In[ ]:


plt.axis('off')
plt.imshow(min_test) #plot the data


# ## Playing with Augmentations

# Finally, I would like to play with some augmentations from [https://github.com/albu/albumentations](albumentation package).
# 
# This will help to have an impression of augmented dataset.

# In[ ]:


# define the dictionary for labels
diagnosis_dict = {
    0:'No DR',
    1:'Mild',
    2:'Moderate',
    3: 'Severe',
    4: 'Proliferative DR'
}


# In[ ]:


# function to plot a grid of images
def view_fundus_images_labels(train, rand_indices, aug = None, title = ''):
    """
    Function to plot grid with several examples of fundus images.
    INPUT:
        train - array with filenames for images and condition labels
        rand_indices - indices of images to plot
        title - plot title

    OUTPUT: None
    """
    width = 5
    height = 2
    counter = 0
    fig, axs = plt.subplots(height, width, figsize=(15,5))
    
    for im in rand_indices:
        # open image
        image = Image.open(os.path.join(TRAIN_IMG_PATH, train.iloc[im].id_code + '.png'))
        
        if aug is not None:
            image = aug(image=np.array(image))['image']
        
        i = counter // width
        j = counter % width
        axs[i,j].imshow(image) #plot the data
        axs[i,j].axis('off')
        
        diagnosis = train[train['id_code'] == train.iloc[im].id_code].diagnosis.values[0]
        
        axs[i,j].set_title(diagnosis_dict[diagnosis])
        counter += 1

    # set suptitle
    plt.suptitle(title)
    plt.show()


# Plot random images from the training set:

# In[ ]:


# get some random image indices from the training set
rand_indices = [randrange(len(train)) for x in range(0,10)]
rand_indices


# In[ ]:


# plot original images
view_fundus_images_labels(train, rand_indices, title = 'Original images')


# Now let's play with some albumentation filters:

# Augment the images with [CLAHE](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE):

# In[ ]:


aug = CLAHE(p=1)
view_fundus_images_labels(train, rand_indices, aug, title = 'CLAHE')


# Try adding some gaussian noise:

# In[ ]:


aug = GaussNoise(p=1)
view_fundus_images_labels(train, rand_indices, aug, title = 'GaussNoise')


# Playing with brightness and constrast:

# In[ ]:


aug = RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p = 1)
view_fundus_images_labels(train, rand_indices, aug, title = 'RandomBrightnessContrast')


# See how random brightness and contrast affect images. This filter should certainly be used for data augmentation.

# Padding images:

# In[ ]:


aug = PadIfNeeded(min_height=1024, min_width=1024, p = 1)
view_fundus_images_labels(train, rand_indices, aug, title = 'Padding Images')


# ## Conclusion
# After the EDA we can say the following:
# * The dataset __is heavily imbalanced__. __Data augmentation is required__.
# * The __distribution of sizes of images from train and test sets is different__. This will probably have an impact on classification results.
# 
# In this EDA we also explored augmented images to have an impression what augmented dataset will look like.

# ## References and Credits
# 
# 1. [Wikipedia page on diabetic retinopathy](https://en.wikipedia.org/wiki/Diabetic_retinopathy)
# 2. [What is diabetic retinopathy?](https://www.aao.org/eye-health/diseases/what-is-diabetic-retinopathy)
# 3. [iMet EDA kernel](https://www.kaggle.com/chewzy/eda-weird-images-with-new-updates)
