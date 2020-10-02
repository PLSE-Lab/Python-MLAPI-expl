#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import numpy as np
import pandas as pd
import pydicom

from skimage.measure import label,regionprops
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt


# ## Raw image

# In[ ]:


d = pydicom.dcmread('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/19.dcm')


# In[ ]:


img = d.pixel_array


# In[ ]:


fig = plt.figure(figsize=(12, 12))

plt.imshow(img)


# ## Rescale and create binary mask
# The bright region inside the lungs are the blood vessels or air. A threshold of -400 HU is used at all places because it was found in experiments. 

# In[ ]:


img = (img + d.RescaleIntercept) / d.RescaleSlope
img = img < -400


# In[ ]:


fig = plt.figure(figsize=(12, 12))

plt.imshow(img)


# ## Cleaning border

# In[ ]:


img = clear_border(img)


# In[ ]:


fig = plt.figure(figsize=(12, 12))

plt.imshow(img)


# ## Remove small region

# In[ ]:


img = label(img)

fig = plt.figure(figsize=(12, 12))

plt.imshow(img)


# In[ ]:


areas = [r.area for r in regionprops(img)]
areas.sort()
if len(areas) > 2:
    for region in regionprops(img):
        if region.area < areas[-2]:
            for coordinates in region.coords:                
                img[coordinates[0], coordinates[1]] = 0
img = img > 0


# In[ ]:


fig = plt.figure(figsize=(12, 12))

plt.imshow(img)


# ## Conclusion 
# The determining of a mask for the lungs is the starting point in the algorithm for determining the volume of the lungs by CT images. The next step is the correct integration of all CT images to determine the volume of the lungs.
