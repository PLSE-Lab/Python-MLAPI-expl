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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from spectral import get_rgb, ndvi
from skimage import io
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from glob import glob
from tqdm import tqdm

# code from https://www.kaggle.com/fppkaggle/making-tifs-look-normal-using-spectral-fork
def read_tif_to_jpg(img_path):
    img = io.imread(img_path)
    img2 = get_rgb(img, [2, 1, 0]) # RGB

    # rescaling to 0-255 range - uint8 for display
    rescaleIMG = np.reshape(img2, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 255))
    rescaleIMG = scaler.fit_transform(rescaleIMG) # .astype(np.float32)
    img2_scaled = (np.reshape(rescaleIMG, img2.shape)).astype(np.uint8)
    
    return img2_scaled


# Take `file_15937` for example. When plotting the jpg file, it looks like cloudy or partly cloudy primary.

# In[ ]:


import cv2
img = cv2.imread('../input/test-jpg-v2/file_15937.jpg')
plt.imshow(img)


# However, when checking tif file, it looks totally different. 

# In[ ]:


img2 = read_tif_to_jpg('../input/test-tif-v2/file_15937.tif')
plt.imshow(img2)


# ## Test for fixed tif 

# In[ ]:


plt.figure()
img = cv2.imread('../input/test-jpg-v2/file_15937.jpg')
plt.subplot(121)
plt.imshow(img)

img2 = read_tif_to_jpg('../input/test-tif-v2/file_5975.tif')
plt.subplot(122)
plt.imshow(img2)


# Looks good!

# In[ ]:


plt.figure()
img = cv2.imread('../input/test-jpg-v2/file_2202.jpg')
plt.subplot(121)
plt.imshow(img)

img2 = read_tif_to_jpg('../input/test-tif-v2/file_4764.tif')
plt.subplot(122)
plt.imshow(img2)

