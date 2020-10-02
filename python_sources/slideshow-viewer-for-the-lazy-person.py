#!/usr/bin/env python
# coding: utf-8

# ### If you copy and run this notebook, the train images can be displayed in sequence as an animation.

# In[ ]:


get_ipython().system('pip install imagecodecs')
get_ipython().system('pip install tifffile')


# In[ ]:


import skimage.io
import matplotlib.pyplot as plt
import time
import numpy as np 
import pandas as pd
import os


# In[ ]:


df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')


# In[ ]:


num_frame = 50
interval = 10
repeat = False


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'nbagg')

import itertools
import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


def _update(frame):
    img_id = df.iloc[frame]['image_id']
    img_path = '../input/prostate-cancer-grade-assessment/train_images/' + img_id + '.tiff'

    image = skimage.io.MultiImage(img_path)[2]
    image = np.array(image)
    h,w,_ = image.shape
    scale = 256/max(h,w)
    image = cv2.resize(image, (int(w*scale),int(h*scale)))
    plt.imshow(image)

fig = plt.figure(figsize=(10, 6))

params = {
    'fig': fig,
    'func': _update,
    'fargs': (),
    'interval': interval,
    'frames': np.arange(0, num_frame*interval, 1),
    'repeat': repeat,
    'blit': True,
    'cache_frame_data': False,
}
anime = animation.FuncAnimation(**params)


# In[ ]:


idx = 0


# In[ ]:


plt.figure(figsize=(6,6))
img_id = df.iloc[idx]['image_id']
img_path = '../input/prostate-cancer-grade-assessment/train_images/' + img_id + '.tiff'

image = skimage.io.MultiImage(img_path)[2]
image = np.array(image)
plt.imshow(image)
idx += 1

