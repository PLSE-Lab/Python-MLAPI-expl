#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# In[ ]:


# Use Lopuhin's dataset for faster image loading.
# https://www.kaggle.com/lopuhin/panda-2020-level-1-2

import glob

paths = sorted(glob.glob('../input/panda-2020-level-1-2/train_images/train_images/*_2.jpeg'))
print(len(paths))


# In[ ]:


# Use only 4000 images for demonstration.

paths = paths[:4000]


# In[ ]:


# Here comes imagehash
# https://github.com/JohannesBuchner/imagehash

import cv2
import imagehash
from tqdm import tqdm_notebook as tqdm
from PIL import Image

funcs = [
    imagehash.average_hash,
    imagehash.phash,
    imagehash.dhash,
    imagehash.whash,
    #lambda x: imagehash.whash(x, mode='db4'),
]

hashes = []
for path in tqdm(paths, total=len(paths)):

    image = cv2.imread(path)
    image = Image.fromarray(image)
    hashes.append(np.array([f(image).hash for f in funcs]).reshape(256))


# In[ ]:


# use cuda to speed up

import torch

hashes = torch.Tensor(np.array(hashes).astype(int)).cuda()


# In[ ]:


# calc similarity scores

sims = np.array([(hashes[i] == hashes).sum(dim=1).cpu().numpy()/256 for i in range(hashes.shape[0])])


# In[ ]:


sims.shape


# In[ ]:


# Let's check image pairs with similarity larget than threshold.
# You can lower threshold to find more duplicates (and more false positives).

import matplotlib.pyplot as plt

threshold = 0.96
duplicates = np.where(sims > threshold)

pairs = {}
for i,j in zip(*duplicates):
    if i == j:
        continue

    path1 = paths[i]
    path2 = paths[j]
    print(path1)
    print(path2)

    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)

    if image1.shape[0] > image1.shape[1] / 2:
        fig,ax = plt.subplots(figsize=(20,20), ncols=2)
    elif image1.shape[1] > image1.shape[0] / 2:
        fig,ax = plt.subplots(figsize=(20,20), nrows=2)
    else:
        fig,ax = plt.subplots(figsize=(20,30), nrows=2)
    ax[0].imshow(image1)
    ax[1].imshow(image2)
    plt.show()


# In[ ]:





# In[ ]:




