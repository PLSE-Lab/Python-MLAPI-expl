#!/usr/bin/env python
# coding: utf-8

# ### This quick bit of code could help improve the quality of the masks.
# ### Here, we ultimately want to alter the masks to differentiate each object, such that there will be a 255 pixel border around discrete objects. Blobs of, for example, cars, will be broken into individual cars.
# ### Any pixel not completely surrounded by pixels of the same label gets marked as a boundary. Boundaries are 2 pixels thick.

# ### The idea here is to build this process into a custom transformation in a dataloader for PyTorch, or iterate over all masks in the train / val directory as a preprocessing step

# In[39]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import copy
from skimage.segmentation import find_boundaries
from PIL import Image


# In[40]:


# Read in an example mask and view the critical part of it
msk = np.asarray(Image.open("../input/train_label/171206_033642600_Camera_5_instanceIds.png"))
plt.figure(figsize=(20,20))
plt.imshow(msk[1600:1900:, 1000:2100:])


# In[41]:


## use find_boundaries function; eyeball check to see how it did (which is remarkably well)
boundaries = find_boundaries(copy(msk), mode = 'thick')
plt.figure(figsize=(20,20))
plt.imshow(boundaries[1600:1900:, 1000:2100:])


# In[42]:


# since the `boundaries` array is a boolean, it is simple to use it to set pixels to 255 on the original mask where the bool is True
msk_boundaries = copy(msk)
msk_boundaries[boundaries] = 255

plt.figure(figsize=(20,20))
plt.imshow(msk_boundaries[1600:1900:, 1000:2100:])

