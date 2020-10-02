#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pylab as pl # linear algebra + plotting
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# It is sometimes a good idea to generate augmented data when the data is not equally distributed among classes, especially with image data. This becomes even more crucial when we don't have any idea about the distribution of classes in test data because there might be fewer samples from more frequent classes and more samples from classes with 1 sample in the train set for example. So, we need to generate modified versions of train images to make sure first, our learner sees enough data and secondly it is invariant to small changes like rotation or camera angles.
# 
# keras has a good utility to generate augmented data from the train set, as @YouHanLee beautifully explained in his kernel:
# 
# [https://www.kaggle.com/youhanlee/small-data-many-class-data-augmentation](https://www.kaggle.com/youhanlee/small-data-many-class-data-augmentation)
# 
# I prefer to have more control over my augmentation mechanism, so I have made the following code using skimage to generate augmented images from any photo. The same procedures are possible to do with OpenCV and even tensorflow to some extent.
# 
# Let's first look at some photos from the same class (from the most frequent class) to see what type of augmentation would be the best:

# In[ ]:


train = pd.read_csv('../input/train.csv')
print(train.Id.value_counts().head())
wale_data = {}
wale_data['w_23a388d'] = train[train.Id=='w_23a388d'].Image.values.tolist()
wale_data['w_9b5109b'] = train[train.Id=='w_9b5109b'].Image.values.tolist()


# In[ ]:


for wale_name in wale_data:
    F = pl.figure(figsize=(15,9))
    G = pl.GridSpec(3, 4, left=.01, right=.99, bottom=0.05, top=0.9, wspace=.01, hspace=0.01, figure=F)
    for i in range(12):
        im = pl.imread('../input/train/' + wale_data[wale_name][i])
        ax = pl.subplot(G[i])
        ax.imshow(im)
        ax.set_axis_off()
        ax.set_aspect('equal')
    pl.suptitle(wale_name)


# So, it seems camera angle, zoom, rotations, skewness, translations, not fully out of water tails, white balance and color intensity, and noise induced by the picture quality and water splashes are common sources of differences in images.
# 
# and, the important part to distinguish between the wales is their tale pattern. One idea might be to convert the images to grayscale and do some pre-processing, but it is not within the goals of the current kernel. We will focus next on the implementation of the above-mentioned transformations to do the augmentation.
# 
# We will implement the Affine Transform, which is a combination of translation, rotation, and scaling. This transformation preserves parallel lines.
# 
# We can do Projective Transform, which is similar to looking at the object from a different perspective, so it does not preserve parallel lines to remain parallel.
# 
# I will also do some random cropping and adding Gaussian noise to the images:

# In[ ]:


from skimage.transform import warp, AffineTransform, ProjectiveTransform
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid
from skimage.filters import gaussian
from skimage.util import random_noise
import random

def randRange(a, b):
    '''
    a utility functio to generate random float values in desired range
    '''
    return pl.rand() * (b - a) + a


def randomAffine(im):
    '''
    wrapper of Affine transformation with random scale, rotation, shear and translation parameters
    '''
    tform = AffineTransform(scale=(randRange(0.75, 1.3), randRange(0.75, 1.3)),
                            rotation=randRange(-0.25, 0.25),
                            shear=randRange(-0.2, 0.2),
                            translation=(randRange(-im.shape[0]//10, im.shape[0]//10), 
                                         randRange(-im.shape[1]//10, im.shape[1]//10)))
    return warp(im, tform.inverse, mode='reflect')


def randomPerspective(im):
    '''
    wrapper of Projective (or perspective) transform, from 4 random points selected from 4 corners of the image within a defined region.
    '''
    region = 1/4
    A = pl.array([[0, 0], [0, im.shape[0]], [im.shape[1], im.shape[0]], [im.shape[1], 0]])
    B = pl.array([[int(randRange(0, im.shape[1] * region)), int(randRange(0, im.shape[0] * region))], 
                  [int(randRange(0, im.shape[1] * region)), int(randRange(im.shape[0] * (1-region), im.shape[0]))], 
                  [int(randRange(im.shape[1] * (1-region), im.shape[1])), int(randRange(im.shape[0] * (1-region), im.shape[0]))], 
                  [int(randRange(im.shape[1] * (1-region), im.shape[1])), int(randRange(0, im.shape[0] * region))], 
                 ])

    pt = ProjectiveTransform()
    pt.estimate(A, B)
    return warp(im, pt, output_shape=im.shape[:2])


def randomCrop(im):
    '''
    croping the image in the center from a random margin from the borders
    '''
    margin = 1/10
    start = [int(randRange(0, im.shape[0] * margin)),
             int(randRange(0, im.shape[1] * margin))]
    end = [int(randRange(im.shape[0] * (1-margin), im.shape[0])), 
           int(randRange(im.shape[1] * (1-margin), im.shape[1]))]
    return im[start[0]:end[0], start[1]:end[1]]


def randomIntensity(im):
    '''
    rescales the intesity of the image to random interval of image intensity distribution
    '''
    return rescale_intensity(im,
                             in_range=tuple(pl.percentile(im, (randRange(0,10), randRange(90,100)))),
                             out_range=tuple(pl.percentile(im, (randRange(0,10), randRange(90,100)))))

def randomGamma(im):
    '''
    Gamma filter for contrast adjustment with random gamma value.
    '''
    return adjust_gamma(im, gamma=randRange(0.5, 1.5))

def randomGaussian(im):
    '''
    Gaussian filter for bluring the image with random variance.
    '''
    return gaussian(im, sigma=randRange(0, 5))
    
def randomFilter(im):
    '''
    randomly selects an exposure filter from histogram equalizers, contrast adjustments, and intensity rescaler and applys it on the input image.
    filters include: equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid, gaussian
    '''
    Filters = [equalize_adapthist, equalize_hist, adjust_log, adjust_sigmoid, randomGamma, randomGaussian, randomIntensity]
    filt = random.choice(Filters)
    return filt(im)


def randomNoise(im):
    '''
    random gaussian noise with random variance.
    '''
    var = randRange(0.001, 0.01)
    return random_noise(im, var=var)
    

def augment(im, Steps=[randomAffine, randomPerspective, randomFilter, randomNoise, randomCrop]):
    '''
    image augmentation by doing a sereis of transfomations on the image.
    '''
    for step in Steps:
        im = step(im)
    return im


# Here is the step by step result of the transformations:

# In[ ]:


im = pl.imread('../input/train/' + train.Image[0])
F = pl.figure(figsize=(15,9))
G = pl.GridSpec(2, 3, left=.01, right=.99, bottom=0.05, top=0.9, wspace=.01, hspace=0.05, figure=F)
ax = pl.subplot(G[0])
ax.imshow(im)
ax.set_axis_off()
ax.set_aspect('equal')
ax.set_title('original' + r'$\rightarrow$')
for i, step in enumerate([randomAffine, randomPerspective, randomFilter, randomNoise, randomCrop]):
    ax = pl.subplot(G[i+1])
    im = step(im)
    ax.imshow(im)
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_title(step.__name__ + (r'$\rightarrow$' if i < 4 else ''))


# The filter step is one filter selected randomly from histogram equalizers, contrast adjustments, Gaussian blurring, and intensity rescale:

# In[ ]:


im = pl.imread('../input/train/' + train.Image[0])
F = pl.figure(figsize=(15,6))
G = pl.GridSpec(2, 4, left=.01, right=.99, bottom=0.05, top=0.9, wspace=.01, hspace=0.1, figure=F)
ax = pl.subplot(G[0])
ax.imshow(im)
ax.set_axis_off()
ax.set_aspect('equal')
ax.set_title('original')
for i, filt in enumerate([equalize_adapthist, equalize_hist, adjust_log, adjust_sigmoid, randomGamma, randomGaussian, randomIntensity]):
    ax = pl.subplot(G[i+1])
    ax.imshow(filt(im))
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_title(filt.__name__ + ' on (original)')


# and, here is the final outcome, 11 augmented images compared to the original image:

# In[ ]:


im = pl.imread('../input/train/' + train.Image[0])
Aug_im = [augment(im) for i in range(11)]


# In[ ]:


F = pl.figure(figsize=(15,10))
G = pl.GridSpec(3, 4, left=.01, right=.99, bottom=0.05, top=0.9, wspace=.01, hspace=0.05, figure=F)
ax = pl.subplot(G[0])
ax.imshow(im)
ax.set_axis_off()
ax.set_aspect('equal')
ax.set_title('original')
for i in range(1, 12):
    ax = pl.subplot(G[i])
    ax.imshow(Aug_im[i-1])
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_title(f'Augmented image {i}')


# **Further Readings**
# 
# I recommend the following links to the docs and some blogs:
# 
# * http://scikit-image.org/docs/dev/api/skimage.transform.html
# * http://scikit-image.org/docs/dev/api/skimage.exposure.html
# * http://scikit-image.org/docs/dev/api/skimage.filters.html
# * http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_geometric.html
# * https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
# * https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
# * https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced
# * https://machinelearningmastery.com/image-augmentation-deep-learning-keras/
# 
# **Further Steps**
# 
# It is possible to fine-tune the transformations to avoid un-natural final images. 
# 
# I avoided using horizontal/vertical flip transformations since I think mirrored images of the whale tales might confuse the learning system in this problem.
# 
# We can write the new images on the disk, or we can use this in keras pipelines to augment while reading the data.
# 
# I hope it was helpful. Let me know if you have any critics or have a way to improve it.
