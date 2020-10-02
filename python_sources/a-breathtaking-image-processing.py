#!/usr/bin/env python
# coding: utf-8

# **Pneumothorax...**
# ![meme_lol](https://i.kym-cdn.com/photos/images/newsfeed/001/502/282/0b1.jpg)
# ... *literally*.
# 
# Now, memes aside, here I present some work on experimenting with 
# data and some ideas for images processing.
# 
# # Understading data and some ideas
# 
# * Images are **X-rays** of the lungs (both)
# * A **mask** over the image indicacte if the lung is compromised, and exactly where.
# * On most of the images, the **backbone** of the person is visible. This could help to **align the images**.
#   * Need to be careful aligning masks as well
#   * Need to store transformation for each image to perform an inverse transformation if necessary
# * Probably it would be easier for a model to **understand lungs using a single orientation** (instead of left and right)
#   * After the image is aligned with the backbone, the lungs can be separated into left and right lungs
#   * Then, the right lung image can be **mirrored horizontally**, and get a new "left" lung image!
#   * This is likely to make DA easier in the future

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import sys
import glob

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import skimage

from PIL import Image
from PIL import ImageFilter

from sklearn.svm import SVC

sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')

from mask_functions import rle2mask

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input/siim-acr-pneumothorax-segmentation-data/pneumothorax"))

sample_files = glob.glob("../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm")
df_sample_files = pd.read_csv(
    "../input/siim-acr-pneumothorax-segmentation/sample images/train-rle-sample.csv",
    header=None,
    names=['filename', 'target']
)
df_sample_files = df_sample_files.set_index('filename')

# Any results you write to the current directory are saved as output.


# In[ ]:


dpi = 220
matplotlib.rcParams['figure.dpi'] = dpi


# In[ ]:


df_sample_files


# In[ ]:


samples = [
    pydicom.dcmread(sample_file)
    for sample_file in sample_files
]
print(f"Loaded {len(samples)} samples")


# ## X-ray sample image
# 
# Here it is the first example

# In[ ]:


plt.imshow(samples[0].pixel_array, cmap='gray')


# # Process the images and try to segment the lungs
# 
# With the lungs segmented, it will make easier to align the images vertically. Here I remove any black padding from the images, then also perform a simple image equalization.

# In[ ]:


def get_biggest_blob(mask, max_blobs=1, max_val=255, inverted=False):
    """Get the biggest connected component from a mask

    Parameters
    ----------
    mask : numpy image
        A numpy image containing a mask
    max_val : int or float
        Maximum value to use in the new mask
    inverted : bool
        Returns the biggest negative mask

    Returns
    -------
    A new mask containing only the biggest blob
    """
    if inverted:
        mask = np.logical_not(mask).astype(np.uint8)

    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    sizes = stats[:, -1]

    labels = [
        (i, sizes[i])
        for i in range(1, nb_components)
    ]
    labels.sort(key=lambda l: l[1], reverse=True)
    
    mask = np.zeros(output.shape)
    mask_labels = np.zeros(output.shape)
    for i in range(min(len(labels), max_blobs)):
        mask[output == labels[i][0]] = max_val
        mask_labels[output == labels[i][0]] = i + 1

    if inverted:
        mask = np.logical_not(mask).astype(np.uint8)

    return mask, mask_labels


def get_basename(path):
    return os.path.basename(path).rsplit('.', 1)[0]


# In[ ]:


sample_index = 0

sample = samples[sample_index].pixel_array
h, w = sample.shape

# Remove black padding on some images and equalize
sample_pre = Image.fromarray(sample)
sample_pre = sample_pre.crop(sample_pre.getbbox())
sample_pre = sample_pre.resize((h, w))
sample_pre = np.asarray(sample_pre)
sample_pre = cv2.equalizeHist(sample_pre)

# Get mask for this sample
target = df_sample_files.loc[get_basename(sample_files[sample_index])].target

if target != '-1':
    sample_mask = rle2mask(target, h, w).T

    # Plot original sample and processed one
    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(sample, cmap='gray')
    ax1.imshow(sample_mask, alpha=0.3, cmap='binary')
    ax2.imshow(sample_pre, cmap='gray')
    ax2.imshow(sample_mask, alpha=0.3, cmap='binary')
else:
    print('Nothing to see in this sample')


# # Trying to separate lungs
# 
# Using thresholding and flood-filling to detect the best candidates for the lungs segmentation masks

# In[ ]:


# Apply some minimum filter to remove noises
sample_aux = Image.fromarray(sample_pre)
sample_aux = sample_aux.filter(ImageFilter.MinFilter(3))
sample_aux = np.asarray(sample_aux)

img = cv2.bitwise_not(sample_aux)

# Binary thresholding
th, im_th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV);
 
# Copy the thresholded image.
im_floodfill = im_th.copy()
 
# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
 
# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0,0), 255);
 
# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# erode and dilate to remove mini kernels of pixels
kernel = np.ones((3, 3))
im_floodfill_inv = cv2.erode(im_floodfill_inv, kernel, iterations=3)
im_floodfill_inv = cv2.dilate(im_floodfill_inv, kernel, iterations=3)

# Get the two biggest blobs
im_out, mask_labels = get_biggest_blob(im_floodfill_inv, max_blobs=2)

f, axes = plt.subplots(1, 5)

axes[0].imshow(img, cmap='gray')
axes[0].imshow(sample_mask, cmap='binary', alpha=0.3)

axes[1].imshow(im_th)
axes[2].imshow(im_floodfill)
axes[3].imshow(im_floodfill_inv)
axes[4].imshow(im_out)


# ## Create a SVM model to get the best line between the lungs
# 
# The idea here is that each lung was labeled by the `get_biggest_blob` function. The SVM model will fit a decision border line between both lungs.

# In[ ]:


label_1_points = np.flip(np.argwhere(mask_labels == 1), 1)
label_2_points = np.flip(np.argwhere(mask_labels == 2), 1)


# In[ ]:


X = np.vstack((label_1_points, label_2_points))
y = np.hstack((
    np.ones(len(label_1_points)) * 0,
    np.ones(len(label_2_points)) * 1,
))

data = np.column_stack((X, y))


# In[ ]:


# Select some random samples of pixels
train = data[np.random.choice(data.shape[0], 15000, replace=False)]


# In[ ]:


X_train = train[:, :2]
y_train = train[:, 2]


# In[ ]:


svm = SVC(kernel='linear')

svm.fit(X_train, y_train)


# In[ ]:


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, c='r', alpha=0.3);
        
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()


# In[ ]:


X0, X1 = X_train[:, 0], X_train[:, 1]

plt.scatter(X0[:300], X1[:300], c=y_train[:300])
plot_svc_decision_function(svm, plot_support=True)


# ## Get the angle of the SVM decision line and rotate the image accordingly
# 
# With the SVM coeficients ready, we can calculate the angle of the line SVM defined. To align the image, we rotate the original image to the opposite angle so as to cancel the image ratation.

# In[ ]:


import math

def get_degree_from_points(x1, y1, x2, y2):
    Y1 = max(y1, y2)
    Y2 = min(y1, y2)
    
    return math.degrees(math.acos((Y1 - Y2) / math.sqrt( (x1 - x2) ** 2 + (Y1 - Y2) ** 2)))

def get_coefs_from_points(x1, y1, x2, y2):
    degree = get_degree_from_points(x1, y1, x2, y2)

    if np.isclose(degree, 0.0):  # Straight vertical lined
        m = np.nan
        c = np.nan
    else:
        m, c = np.polyfit(
            np.array([x1, x2], dtype=np.float64),
            np.array([y1, y2], dtype=np.float64), 1)

    return m, c


def get_line_from_coefs(m, c, img, x=np.nan):
    height, width = img.shape

    y1 = 0
    y2 = height

    if not np.isnan(m) and not np.isnan(c):
        x1 = max(0, min(width, int((y1 - c) / m)))
        x2 = max(0, min(width, int((y2 - c) / m)))
    else:
        x1 = x
        x2 = x

    return (x1, y1, x2, y2)

def rotate_2d(points, center, angle=0):
    """Rotates a list of points with respect to a center point

    Parameters
    ----------

    points : numpy array
    center : numpy array containing one point
    angle : float
        Angle in degrees"""
    angle = np.deg2rad(angle)

    return center + np.dot(
        points - center,
        np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    )


# In[ ]:


print('Support vectors:', svm.support_vectors_)

# Find SVM decision line
# TODO: make this easier
a = -svm.coef_[0][0] / svm.coef_[0][1]
b = svm.intercept_[0] / svm.coef_[0][1]

def decision(x):
    return a * x - b

x1, x2 = 0, 100
y1, y2 = decision(x1), decision(x2)

print(x1, x2, y1, y2)

angle = get_degree_from_points(x1, y1, x2, y2)
print('Angle:', angle)

m, c = get_coefs_from_points(x1, y1, x2, y2)
x1, y1, x2, y2 = get_line_from_coefs(m, c, sample, x1)

points = np.array([[x1, y1], [x2, y2]])
points_ = rotate_2d(points, np.array([int(h / 2), int(w / 2)]), angle=angle)
x1_, y1_ = points_[0]
x2_, y2_ = points_[1]

m, c = get_coefs_from_points(x1_, y1_, x2_, y2_)
x1_, y1_, x2_, y2_ = get_line_from_coefs(m, c, sample, x1_)


# In[ ]:


f, axes = plt.subplots(1, 2)

axes[0].imshow(sample_pre, cmap='gray')
axes[0].plot([x1, x2], [y1, y2], 'r-')
axes[1].imshow(Image.fromarray(sample_pre).rotate(-angle), cmap='gray')
axes[1].plot([x1_, x2_], [y1_, y2_], 'r-')


# # WIP
# 
# This work is still in progress. You're free to collaborate or fork if you find it somewhat helpful :)
# 
# I haven't successfully been able to correctly detect the lungs on all samples, so probably some further processing will be necessary.
# 
# My plan is to improve the dataset, which I think it's key for getting better results on this kind of competitions. After that, I'll implement some Unet model to train to predict the mask. I'm pretty sure that making the model to learn just one kind of lung orientation will help to improve the results later.

# In[ ]:




