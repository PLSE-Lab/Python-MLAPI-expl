#!/usr/bin/env python
# coding: utf-8

# Standard imports:

# In[ ]:


import tensorflow as tf
import keras 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys


# The data is loaded from the directory into 2 numpy arrays: one for the image directorys and the other holding the class of each image. The function imload takes a list of image directorys and outputs an array of those images.

# In[ ]:


data_dir = '../input/neuron cy5 full/Neuron Cy5 Full'
folders = {'Treated': 1, 'Untreated': 0}
tags = {'B':0, 'C':36, 'D':72, 'E':108, 'F':144, 'G':180, 'AB':216, 'AC':246, 'AD':276, 'AE':307, 'AF':337, 'AG':367}
img_dirs = [None]*792
y = np.empty([792,2])

image_size = 2048

for folder in os.listdir(data_dir):
    folder_dir = os.path.join(data_dir, folder)
    bin_class = folders[folder]
    for idx, file in enumerate(os.listdir(folder_dir)):
        img_dirs[idx+bin_class*396] = os.path.join(folder_dir, file)
        y[idx+bin_class*396,:] = [1-bin_class,bin_class]


classes = dict((v, k.capitalize(),) for k, v in folders.items())
num_classes = len(classes)

def imload(file_dir):
    img = cv2.imread(file_dir, cv2.IMREAD_GRAYSCALE)
    return img  


# Here is an example training image from each class:

# In[ ]:


fig, ax = plt.subplots(1,2, figsize= 3*plt.figaspect(0.5))
for i in range(num_classes):
    ax[i].set_title(classes[i]+' example:', fontsize=20)
    im_dir = img_dirs[np.argmax(y[:,i])]
    ax[i].imshow(imload(im_dir), cmap='gray')
    ax[i].axis('Off')


#  It appears that the 'Treated' images are brighter than the 'Untreated' ones. To see whether this is the case and to find out how to solve this issue, a histogram of pixel intensities is plotted.

# In[ ]:


num_pixels = image_size**2

treat_bins = np.zeros(256)
print('Calculating for Treated images:')
for idx in range(0, 396):
    im = imload(img_dirs[idx])
    treat_bins = np.add(np.bincount(im.ravel(), minlength=256), treat_bins)
    prog = 'Progress: '+str(idx+1)+'/396'
    sys.stdout.write('\r'+prog)
sys.stdout.write('\r Done            \n')

untreat_bins = np.zeros(256)
print('Calculating for Untreated images:')
for idx in range(397, 792):
    im = imload(img_dirs[idx])
    untreat_bins = np.add(np.bincount(im.ravel(), minlength=256), untreat_bins)
    prog = 'Progress: '+str(idx-396)+'/396'
    sys.stdout.write('\r'+prog)
sys.stdout.write('\r Done            \n')


# In[ ]:


plt.figure(figsize=(20,10))
plt.title('Histograms for original images: (log-scale)', fontsize=20)
plt.bar(np.arange(len(treat_bins)),treat_bins, log=True, alpha=0.5, label='Treated intensities', color='b')
plt.bar(np.arange(len(untreat_bins)), untreat_bins, log=True, alpha=0.5, label='Untreated intensities', color='g')
plt.legend(loc='upper right', prop={'size': 20});


# This makes it noticable that the 'Treated' images really are brighter than the 'Untreated' ones! The same analysis is now performed with each images pixel instensities normalised into the full (0, 255) range.

# In[ ]:


treat_bins_norm = np.zeros(256)
print('Calculating for Treated images:')
for idx in range(0, 396):
    im = imload(img_dirs[idx]).astype('float')
    im *= 255/im.max()
    im = np.uint8(im)
    treat_bins_norm = np.add(np.bincount(im.ravel(), minlength=256), treat_bins_norm)
    prog = 'Progress: '+str(idx+1)+'/396'
    sys.stdout.write('\r'+prog)
sys.stdout.write('\r Done            \n')

untreat_bins_norm = np.zeros(256)
print('Calculating for Untreated images:')
for idx in range(397, 792):
    im = imload(img_dirs[idx]).astype('float')
    im *= 255/im.max()
    im = np.uint8(im)
    untreat_bins_norm = np.add(np.bincount(im.ravel(), minlength=256), untreat_bins_norm)
    prog = 'Progress: '+str(idx-396)+'/396'
    sys.stdout.write('\r'+prog)
sys.stdout.write('\r Done            \n')


# In[ ]:


plt.figure(figsize=(20,10))
plt.title('Histograms for normalised images: (log-scale)', fontsize=20)
plt.bar(np.arange(len(treat_bins_norm)), treat_bins_norm, log=True, alpha=0.5, label='Treated intensities', color='b')
plt.bar(np.arange(len(untreat_bins_norm)), untreat_bins_norm, log=True, alpha=0.5, label='Untreated intensities', color='g')
plt.legend(loc='upper right', prop={'size': 20});


# The distribution of images intensities has clearly become less biased, thus the images will be normalised like this for classification. The images that were shown above are now shown with this normalisation.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize= 3*plt.figaspect(0.5))
for i in range(num_classes):
    ax[i].set_title(classes[i]+' example:', fontsize=20)
    im_dir = img_dirs[np.argmax(y[:,i])]
    im = imload(im_dir).astype('float')
    im *= 255/im.max()
    im = np.uint8(im)
    ax[i].imshow(im, cmap='gray')
    ax[i].axis('Off')


# In[ ]:


treat_bins_clip = np.zeros(256)
print('Calculating for Treated images:')
for idx in range(0, 396):
    im = imload(img_dirs[idx]).astype('float')
    im = np.clip(im, 0, 25)
    im *= 255*im.max()
    im = np.uint8(im)
    treat_bins_clip = np.add(np.bincount(im.ravel(), minlength=256), treat_bins_clip)
    prog = 'Progress: '+str(idx+1)+'/396'
    sys.stdout.write('\r'+prog)
sys.stdout.write('\r Done            \n')

untreat_bins_clip = np.zeros(256)
print('Calculating for Untreated images:')
for idx in range(397, 792):
    im = imload(img_dirs[idx]).astype('float')
    im = np.clip(im, 0, 25)
    im *= 255*im.max()
    im = np.uint8(im)
    untreat_bins_clip = np.add(np.bincount(im.ravel(), minlength=256), untreat_bins_clip)
    prog = 'Progress: '+str(idx-396)+'/396'
    sys.stdout.write('\r'+prog)
sys.stdout.write('\r Done            \n')


# In[ ]:


plt.figure(figsize=(20,10))
plt.title('Histograms for clipped images: (log-scale)', fontsize=20)
plt.bar(np.arange(len(treat_bins_clip)), treat_bins_clip, log=True, alpha=0.5, label='Treated intensities', color='b')
plt.bar(np.arange(len(untreat_bins_clip)), untreat_bins_clip, log=True, alpha=0.5, label='Untreated intensities', color='g')
plt.legend(loc='upper right', prop={'size': 20});


# In[ ]:


img =np.clip(im, 0, 25).astype('float')
img /= img.max()
plt.figure(figsize=(15,15))
plt.title('Example of a clipped image')
plt.imshow(img, cmap='gray');


# In[ ]:


fig, ax = plt.subplots(2,2, figsize= 3*plt.figaspect(0.5))
for i in range(num_classes):
    ax[0,i].set_title(classes[i]+' contrast stretched:', fontsize=20)
    im_dir = img_dirs[np.argmax(y[:,i])]
    img = imload(im_dir).astype('float')
    img_cdf = np.add(treat_bins, untreat_bins).cumsum()
    img_cdf /= img_cdf.max()
    p1 = np.argmin(np.abs(img_cdf-0.01))
    p99 = np.argmin(np.abs(img_cdf-0.99))
    img -= p1
    img /= p99
    img = 255*np.clip(img, 0 ,1)
    ax[0,i].imshow(img, cmap='gray')
    ax[0,i].axis('Off')
for i in range(num_classes):
    ax[1,i].set_title(classes[i]+' original:', fontsize=20)
    im_dir = img_dirs[np.argmax(y[:,i])]
    ax[1,i].imshow(imload(im_dir), cmap='gray')
    ax[1,i].axis('Off')


# In[ ]:


fig, ax = plt.subplots(2,2, figsize= 3*plt.figaspect(0.5))
for i in range(num_classes):
    ax[0,i].set_title(classes[i]+' histogram equalised:', fontsize=20)
    im_dir = img_dirs[np.argmax(y[:,i])]
    img = imload(im_dir).astype('float')
    img_hist, bins = np.histogram(img.flatten(), 256, normed=True)
    cdf = img_hist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    img_eq = np.interp(img.flatten(), bins[:-1], cdf).reshape(img.shape)
    ax[0,i].imshow(img_eq, cmap='gray')
    ax[0,i].axis('Off')
for i in range(num_classes):
    ax[1,i].set_title(classes[i]+' original:', fontsize=20)
    im_dir = img_dirs[np.argmax(y[:,i])]
    ax[1,i].imshow(imload(im_dir), cmap='gray')
    ax[1,i].axis('Off') 


# In[ ]:


fig, ax = plt.subplots(2,2, figsize= 3*plt.figaspect(0.5))
for i in range(num_classes):
   ax[0,i].set_title(classes[i]+' histogram equalised & contrast stretched:', fontsize=20)
   im_dir = img_dirs[np.argmax(y[:,i])]
   img = imload(im_dir).astype('float')
   img -= p1
   img /= p99
   img = 255*np.clip(img, 0 ,1)
   img_hist, bins = np.histogram(img.flatten(), 256, normed=True)
   cdf = img_hist.cumsum() # cumulative distribution function
   cdf = 255 * cdf / cdf[-1] # normalize
   img = np.interp(img.flatten(), bins[:-1], cdf).reshape(img.shape)
   ax[0,i].imshow(img, cmap='gray')
   ax[0,i].axis('Off')
for i in range(num_classes):
   ax[1,i].set_title(classes[i]+' original:', fontsize=20)
   im_dir = img_dirs[np.argmax(y[:,i])]
   ax[1,i].imshow(imload(im_dir), cmap='gray')
   ax[1,i].axis('Off') 


# In[ ]:




