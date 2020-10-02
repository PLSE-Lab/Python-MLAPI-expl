#!/usr/bin/env python
# coding: utf-8

# # ALASKA2 Image Steganalysis
# 
# In this challenge, we are asked to predict whether an image contains a hidden message or not.
# 
# **Disclaimer:** I am not too familiar with steganography algorithms. All below information is gathered from listed papers. Please feel free to correct me if I'm wrong.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import matplotlib.colors
import cv2
from skimage import io
from tqdm import tqdm
import seaborn as sns

PATH = '/kaggle/input/alaska2-image-steganalysis'

train_image_names = pd.Series(os.listdir(PATH + '/Cover')).sort_values(ascending=True).reset_index(drop=True)
test_image_names = pd.Series(os.listdir(PATH + '/Test')).sort_values(ascending=True).reset_index(drop=True)
sample_submission = pd.read_csv(f'{PATH}/sample_submission.csv')

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
k=0
for i, row in enumerate(ax):
    for j, col in enumerate(row):
        img = io.imread(PATH + '/Cover/' + train_image_names[k])
        col.imshow(img)
        col.set_title(train_image_names[k])
        k=k+1
plt.suptitle('Samples from Cover Images', fontsize=14)
plt.show()


# # Steganography
# 
# Let's say you would like to communicate with someone without a third person eavesdropping on your conversation. In many applications, we already use encryption for this purpose. However, there are also situations where it might look suspicious when you are sending encrypted messages, such as in countries where free speech is suppressed.
# 
# In the latter case, you might want to hide your message in plain sight, such as in images. The image will be altered to contain a hidden message, however it will still resemble the original image as close as possible.
# 
# Another use for steganopraphy is for watermarking digital media to find when images, audio, or video files are pirated.
# 
# # Steganalysis
# Let's say Alice and Bob are trying to have a conversation with steganography. When a warden Wendy wants to eavesdrop on their conversation, she might use steganalysis for this purpose.

# # What data do we have?
# For training we have 
# * unaltered cover images for use in training taken with 50 different cameras (from smartphone to full format high end)
# * cover images altered with JMiPOD algorithm
# * cover images altered with JUNIWARD algorithm
# * cover images altered with UERD algorithm
# 
# For the submission we have
# * test images
# * samples_submission.csv
# 
# The objective is to predict whether the image in the test set has a hidden message.

# In[ ]:


for folder in os.listdir(PATH):
    try:
        print(f"Folder {folder} contains {len(os.listdir(PATH + '/' + folder))} images.")
    except:
        print(f'{folder}')


# # One the First Look
# I don't know about you but I can't see any difference between the cover image and the altered images...

# In[ ]:


fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))

folders = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']

for i, row in enumerate(ax):
    k= 0
    for j, col in enumerate(row):
        img = io.imread(PATH + '/' + folders[i] + '/'+ train_image_names[k])
        col.imshow(img)
        col.set_title(f'{train_image_names[k]}: {img.shape}')
        k=k+1
        
for row, r in zip(ax[:,0], folders):
        row.set_ylabel(r, rotation=90, size='large', fontsize=14)
plt.suptitle('Samples from Cover Images and Altered Images', fontsize=14)
plt.show()


# Embedding messages (called payload) in images causes distortion in the altered image (stego image). Therefore, 
# most steganographic methods embed the payload by minimizing a defined distortion function.
# 
# JPEG steganography does not change the pixel values directly in the original spatial domain. It uses something called discrete cosine transform (DCT) and changes the DCT coefficients, then transformes back to the spatial domain.
# 
# # J-UNIWARD 
# Universal Wavelet Relative Distortion for JPEG domain.
# 
# Holub, V., Fridrich, J. & Denemark, T. Universal distortion function for steganography in an arbitrary domain. EURASIP J. on Info. Security 2014, 1 (2014). 
# 
# Avoids making embeddings in clean edges and smooth regions but instead makes embeddings in textures and 'noisy' regions.
# 
# # UERD 
# Uniform Embedding Revisited Distortion
# 
# Guo, Linjie & Ni, Jiangqun & Su, Wenkang & Tang, Chengpei & Shi, Y.Q.. (2015). Using Statistical Image Model for JPEG Steganography: Uniform Embedding Revisited. IEEE Transactions on Information Forensics and Security. 10. 1-1. 
# 
# # J-MiPOD
# Minimizing the Power of Optimal Detector
# 
# V. Sedighi, R. Cogranne and J. Fridrich, "Content-Adaptive Steganography by Minimizing Statistical Detectability," in IEEE Transactions on Information Forensics and Security, vol. 11, no. 2, pp. 221-234, Feb. 2016.
# 
# Is an alternate approach to minimizing the distortion function.

# In[ ]:


folders = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
k=0
img_cover = io.imread(PATH + '/' + folders[0] + '/'+ train_image_names[k])
img_jmipod = io.imread(PATH + '/' + folders[1] + '/'+ train_image_names[k])
img_juniward = io.imread(PATH + '/' + folders[2] + '/'+ train_image_names[k])
img_uerd = io.imread(PATH + '/' + folders[3] + '/'+ train_image_names[k])

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))

ax[0,0].imshow(img_jmipod)
ax[0,1].imshow((img_cover == img_jmipod).astype(int)[:,:,0])
ax[0,1].set_title(f'{train_image_names[k]} Channel 0')

ax[0,2].imshow((img_cover == img_jmipod).astype(int)[:,:,1])
ax[0,2].set_title(f'{train_image_names[k]} Channel 1')
ax[0,3].imshow((img_cover == img_jmipod).astype(int)[:,:,2])
ax[0,3].set_title(f'{train_image_names[k]} Channel 2')
ax[0,0].set_ylabel(folders[1], rotation=90, size='large', fontsize=14)


ax[1,0].imshow(img_juniward)
ax[1,1].imshow((img_cover == img_juniward).astype(int)[:,:,0])
ax[1,2].imshow((img_cover == img_juniward).astype(int)[:,:,1])
ax[1,3].imshow((img_cover == img_juniward).astype(int)[:,:,2])
ax[1,0].set_ylabel(folders[2], rotation=90, size='large', fontsize=14)

ax[2,0].imshow(img_uerd)
ax[2,1].imshow((img_cover == img_uerd).astype(int)[:,:,0])
ax[2,2].imshow((img_cover == img_uerd).astype(int)[:,:,1])
ax[2,3].imshow((img_cover == img_uerd).astype(int)[:,:,2])
ax[2,0].set_ylabel(folders[3], rotation=90, size='large', fontsize=14)

plt.suptitle('Pixel Deviation from Cover Image', fontsize=14)

plt.show()


# In[ ]:


folders = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
pixels_changed = [[0, 0, 0]]
for k in tqdm(range(len(train_image_names))):
    img_cover = io.imread(PATH + '/' + folders[0] + '/'+ train_image_names[k])
    img_jmipod = io.imread(PATH + '/' + folders[1] + '/'+ train_image_names[k])
    img_juniward = io.imread(PATH + '/' + folders[2] + '/'+ train_image_names[k])
    img_uerd = io.imread(PATH + '/' + folders[3] + '/'+ train_image_names[k]) 
    pixels_changed = np.append(pixels_changed, [[((~(img_cover == img_jmipod))).sum(), 
                                                       ((~(img_cover == img_juniward))).sum(), 
                                                       ((~(img_cover == img_uerd))).sum()]], axis=0)


# In[ ]:


df_changed_pixels = pd.DataFrame(pixels_changed, columns = ['JMiPOD', 'JUNIWARD', 'UERD'])
df_changed_pixels =  df_changed_pixels[1:]
df_changed_pixels.to_csv("changed_pixels.csv",index=False)

fig = plt.figure(figsize=(10,5))
ax = sns.kdeplot(df_changed_pixels.JMiPOD)
ax = sns.kdeplot(df_changed_pixels.JUNIWARD)
ax = sns.kdeplot(df_changed_pixels.UERD)
plt.title("Number of changed pixels", fontsize=14)
plt.xlabel("Pixels", fontsize=14)
plt.show()


# # Stego Images Wtih 0 Changed Pixels
# I assume that if 0 pixels are changed between cover image and stego image that means there is no message embedded. If somebody knows whether that's true or not, please leave a comment.

# In[ ]:


fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(16, 10))

for i, algo in enumerate(['JMiPOD', 'JUNIWARD', 'UERD']):
    print(f'{algo}: There are {df_changed_pixels[df_changed_pixels[algo] == 0].JMiPOD.count()} images with 0 changed pixels.')
    for j in range(5):
        image_idx = df_changed_pixels[df_changed_pixels[algo] == 0].index[j] - 1
        img_cover = io.imread(PATH + '/' + folders[0] + '/'+ train_image_names[image_idx])
        ax[i,j].imshow(img_cover)
        ax[i,j].set_title(f'{train_image_names[image_idx]}')
    ax[i,0].set_ylabel(algo, rotation=90, size='large', fontsize=14)

plt.show()


# Ok, let's have a closer look at some of these images:

# In[ ]:


def plot_cover_and_stego(k):
    k = k - 1
    img_cover = io.imread(PATH + '/' + folders[0] + '/'+ train_image_names[k])
    img_jmipod = io.imread(PATH + '/' + folders[1] + '/'+ train_image_names[k])
    img_juniward = io.imread(PATH + '/' + folders[2] + '/'+ train_image_names[k])
    img_uerd = io.imread(PATH + '/' + folders[3] + '/'+ train_image_names[k]) 

    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(12, 10))

    ax[0,0].imshow(img_cover)
    ax[0,0].set_title(f'Cover {train_image_names[k]}')

    ax[0,1].imshow((img_cover == img_jmipod).astype(int)[:,:,0], cmap='Greys',  norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True))
    ax[0,1].set_title(f'Ch0: {(~(img_cover == img_jmipod)).astype(int)[:,:,0].sum()} pixels changed')
    ax[0,2].imshow((img_cover == img_jmipod).astype(int)[:,:,1], cmap='Greys',  norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True))
    ax[0,2].set_title(f'Ch1: {(~(img_cover == img_jmipod)).astype(int)[:,:,1].sum()} pixels changed')
    ax[0,3].imshow((img_cover == img_jmipod).astype(int)[:,:,2], cmap='Greys',  norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True))
    ax[0,3].set_title(f'Ch2: {(~(img_cover == img_jmipod)).astype(int)[:,:,2].sum()} pixels changed')
    ax[0,0].set_ylabel(folders[1], rotation=90, size='large', fontsize=14)

    ax[1,0].imshow(img_cover)
    ax[1,1].imshow((img_cover == img_juniward).astype(int)[:,:,0], cmap='Greys',  norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True))
    ax[1,1].set_title(f'Ch0: {(~(img_cover == img_juniward)).astype(int)[:,:,0].sum()} pixels changed')
    ax[1,2].imshow((img_cover == img_juniward).astype(int)[:,:,1], cmap='Greys',  norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True))
    ax[1,2].set_title(f'Ch1: {(~(img_cover == img_juniward)).astype(int)[:,:,1].sum()} pixels changed')
    ax[1,3].imshow((img_cover == img_juniward).astype(int)[:,:,2], cmap='Greys',  norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True))
    ax[1,3].set_title(f'Ch2: {(~(img_cover == img_juniward)).astype(int)[:,:,2].sum()} pixels changed')

    ax[1,0].set_ylabel(folders[2], rotation=90, size='large', fontsize=14)

    ax[2,0].imshow(img_cover)
    ax[2,1].imshow((img_cover == img_uerd).astype(int)[:,:,0], cmap='Greys',  norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True))
    ax[2,1].set_title(f'Ch0: {(~(img_cover == img_jmipod)).astype(int)[:,:,0].sum()} pixels changed')
    ax[2,2].imshow((img_cover == img_uerd).astype(int)[:,:,1], cmap='Greys',  norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True))
    ax[2,2].set_title(f'Ch1: {(~(img_cover == img_uerd)).astype(int)[:,:,1].sum()} pixels changed')
    ax[2,3].imshow((img_cover == img_uerd).astype(int)[:,:,2], cmap='Greys',  norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True))
    ax[2,3].set_title(f'Ch2: {(~(img_cover == img_uerd)).astype(int)[:,:,2].sum()} pixels changed')

    ax[2,0].set_ylabel(folders[3], rotation=90, size='large', fontsize=14)

    plt.suptitle('Pixel Deviation from Cover Image', fontsize=14)

    plt.show()
    
plot_cover_and_stego(df_changed_pixels[df_changed_pixels.JMiPOD == 0].index[4])
plot_cover_and_stego(df_changed_pixels[df_changed_pixels.JUNIWARD == 0].index[4])
plot_cover_and_stego(df_changed_pixels[df_changed_pixels.UERD == 0].index[4])

