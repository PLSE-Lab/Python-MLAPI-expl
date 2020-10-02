#!/usr/bin/env python
# coding: utf-8

# ## Bone-Segmentation
# 
# ### Simple
# 
#  Here we have the simple task of segmenting the calcified bone tissue from the air and background in the image

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# just once to setup the plots in the notebook (% is for 'magic' commands in jupyter)

import numpy as np # linear algebra
from skimage.io import imread # for reading images
import matplotlib.pyplot as plt # for showing figures


# In[ ]:


bone_image = imread('../input/bone.tif')
print('Loading bone image shape: {}'.format(bone_image.shape))


# The first step is to show the image data and the associated histogram

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 4))
ax1.imshow(bone_image, cmap = 'bone')
_ = ax2.hist(bone_image.ravel(), 20)


# In[ ]:


silly_thresh_value = 55
thresh_image = bone_image > silly_thresh_value

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20, 10))
ax1.imshow(bone_image, cmap = 'bone')
ax1.set_title('Original Image')
ax2.imshow(thresh_image, cmap = 'jet')
ax2.set_title('Thresheld Image')


# In[ ]:


# import the needed morphological operations
from skimage.morphology import binary_erosion, binary_dilation, binary_opening, binary_closing
## your code


# In[ ]:


# import a few filters
from skimage.filters import gaussian, median
## your code


#  - Apply a threshold and then a morphological operation to segment the calcified tissue well
#  - Calculate the porosity (air) volume fraction for each threshold (_hint_: Image Features)
#  - Improve the images using morphological operations and recalculate the porosity (_hint_: Morphological Image Operations)
#  - Fill in all the holes and estimate the bone volume to total volume an important diagnostic metric in biomechanics (_hint_: Fill Holes)
#  - Total volume is __not__ the total volume of image but rather the size of the bone inside the image
#  - Add a filtering block, which works best on these data?

# We can now apply a very silly threshold and see how the results look

# ### ManyThresh
# 
# Here we try a number of different thresholds to identify the bone better.
# 
# 1. Adjust the list of thresholds to cover a reasonable range
# 1. Adjust the morphological operation to it produces the best result
# 1. Add many more steps around the ideal value how does the curve look?

# In[ ]:


threshold_list = [10, 20, 200]

fig, m_ax = plt.subplots(2, len(threshold_list), figsize = (15, 6))
for c_thresh, (c_ax1, c_ax2) in zip(threshold_list, m_ax.T):
    bone_thresh = bone_image > c_thresh
    # your code here
    c_ax1.imshow(bone_thresh, cmap = 'jet')
    c_ax1.set_title('Bone @ {}, Image'.format(c_thresh))
    c_ax1.axis('off')
    
    # do cells
    cell_thresh = bone_image < c_thresh
    # your code here
    c_ax2.imshow(cell_thresh, cmap = 'jet')
    c_ax2.set_title('Cell @ {}, Image'.format(c_thresh))
    c_ax2.axis('off')
    


# ## Cell-Segmentation
# 
# The cell segmentation takes the process one step further and tries to identify the cells by looking for holes inside the bone tissue. 
# 
# 1. Adjust the threshold to the best value as determined before
# 1. Adjust the iterations for the morphological operation 
# 1. Tweak settings to segment cells better
# 
# 
# 
# 1. Make a plot of the X position against the maximum intensity, is there any trend? What might this be from?
# 1. _Optional_: Add 'Haralick' feature analysis to the output, plot some of the parameters against position - Apply a threshold and then a morphological operation to segment the calcified tissue well
#  - Calculate the porosity (air) volume fraction for each threshold (_hint_: Image Features)
#  - Improve the images using morphological operations and recalculate the porosity (_hint_: Morphological Image Operations)
#  - Fill in all the holes and estimate the bone volume to total volume an important diagnostic metric in biomechanics (_hint_: Fill Holes)
#  - Total volume is __not__ the total volume of image but rather the size of the bone inside the image
#  - Add a filtering block, which works best on these data?

# In[ ]:




