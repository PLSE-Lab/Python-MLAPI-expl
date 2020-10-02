#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


cd ../input


# In[ ]:


import cv2
import pydicom
from glob import glob
import matplotlib.pyplot as plt
from mask_functions import rle2mask


# In[ ]:


sample_imgs = [path for path in glob(os.path.join('sample images', '*')) if 'csv' not in path]
train_df = pd.read_csv([path for path in glob(os.path.join('sample images', '*')) if 'csv' in path][0])

data_info = []

for data in train_df.iterrows():
    _, (path, rle) = data
    data_info.append([path, rle])


# In[ ]:


for path, rle in data_info:
    img = pydicom.dcmread(os.path.join('sample images', path+'.dcm')).pixel_array
    
    if rle != '-1':
        mask = rle2mask(rle, img.shape[0], img.shape[1]).T
    
    else:
        mask = np.zeros_like(img)
    
    # Histogram Equalization
    img_hist = cv2.equalizeHist(img)
    
    
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,15))
    
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)

    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original')
#     ax1.imshow(mask, cmap='gray', alpha=0.5)  # Overlay mask

    ax2.imshow(img_hist, cmap='gray')
    ax2.set_title('Histogram Equalization')
    
    ax3.imshow(img_clahe, cmap='gray')
    ax3.set_title('Clahe Histogram Equalization')
    
    plt.show()


# In[ ]:




