#!/usr/bin/env python
# coding: utf-8

# # Openslide
# 
# ### As a new user of openslide I wanted to explore this library and apply it to the PANDA dataset.
# 
# ### I was searching in the public notebooks but couldn't find any dedicated to openslide.
# 
# ## So here it is!
# 
# ### In this notebook you will learn how to extract embedded information and metadata from our Generic tiled TIFF PANDA dataset using OpenSlide and DeepZoom Classes.
# 
# ### Here is the link for the API https://openslide.org/api/python/
# 
# ## Enjoy!

# # Abstract
# 
# OpenSlide Python is a Python interface to the OpenSlide library.
# 
# OpenSlide is a C library that provides a simple interface for reading whole-slide images, also known as virtual slides, which are high-resolution images used in digital pathology. These images can occupy tens of gigabytes when uncompressed, and so cannot be easily read using standard tools or libraries, which are designed for images that can be comfortably uncompressed into RAM. Whole-slide images are typically multi-resolution; OpenSlide allows reading a small amount of image data at the resolution closest to a desired zoom level.

# # Imports and Loading

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os

import numpy as np
import openslide
from openslide import deepzoom
from matplotlib import pyplot as plt


# In[ ]:


#Images / Masks Directories
images_dir = "../input/prostate-cancer-grade-assessment/train_images/"
masks_dir = "../input/prostate-cancer-grade-assessment/train_label_masks/"

#Files
image_files = os.listdir(images_dir)
mask_files = os.listdir(masks_dir)
mask_files_cleaned = [i.replace("_mask", "") for i in mask_files]

#Clean Images without Masks
images_with_masks = list(set(image_files).intersection(mask_files_cleaned))
len(image_files), len(mask_files), len(images_with_masks)


# In[ ]:


image_file = '00928370e2dfeb8a507667ef1d4efcbb.tiff'
mask_file = '00928370e2dfeb8a507667ef1d4efcbb_mask.tiff'


# ## OpenSlide object
# 
# An open whole-slide image.

# In[ ]:


image = openslide.OpenSlide(os.path.join(images_dir, image_file))
mask = openslide.OpenSlide(os.path.join(masks_dir, mask_file))


# In[ ]:


print(type(image))
type(mask)


# In[ ]:


img_level_count = image.level_count
mask_level_count = mask.level_count
print(img_level_count, mask_level_count)


# There are 3 levels of magnification

# In[ ]:


detect_format = openslide.PROPERTY_NAME_VENDOR
print(detect_format)


# In[ ]:


image_size1 = image.level_dimensions[0]
image_size2 = image.level_dimensions[1]
image_size3 = image.level_dimensions[2]
print(image_size1, image_size2, image_size3)


# Here is an example of 3 level dimensions in the same image

# In[ ]:


mask_size1 = mask.level_dimensions[0]
mask_size2 = mask.level_dimensions[1]
mask_size3 = mask.level_dimensions[2]
print(mask_size1, mask_size2, mask_size3)


# Same dimensions for the 3 levels of masks

# In[ ]:


image_dwn1 = image.level_downsamples[0]
image_dwn2 = image.level_downsamples[1]
image_dwn3 = image.level_downsamples[2]
print(image_dwn1, image_dwn2, image_dwn3)


# A list of downsample factors 1, 4 and 16 for each level of the slide

# In[ ]:


mask_dwn1 = mask.level_downsamples[0]
mask_dwn2 = mask.level_downsamples[1]
mask_dwn3 = mask.level_downsamples[2]
print(mask_dwn1, mask_dwn2, mask_dwn3)


# Still the same for the masks

# In[ ]:


image.properties


# In[ ]:


mask.properties


# In[ ]:


image.associated_images


# In[ ]:


mask.associated_images


# In[ ]:


image.read_region((0, 0), 2 , (512, 512))


# Read a specfic region of the WSI based on location, level and size of region

# In[ ]:


mask.read_region((0, 0), 2 , (512, 512))


# In[ ]:


image.get_best_level_for_downsample(18)


# In[ ]:


image.get_thumbnail((300,300))


# ## DeepZoomGenerator
# 
# OpenSlide Python provides functionality for generating individual Deep Zoom tiles from slide objects. This is useful for displaying whole-slide images in a web browser without converting the entire slide to Deep Zoom or a similar format.
# 
# Not useful in this competition but good to know for further applications.

# In[ ]:


zoom = openslide.deepzoom.DeepZoomGenerator(image, tile_size=254, overlap=1, limit_bounds=False)


# In[ ]:


print(type(zoom))


# In[ ]:


zoom.level_count


# In[ ]:


zoom.tile_count


# In[ ]:


zoom.level_tiles


# In[ ]:


zoom.level_dimensions


# In[ ]:


zoom.get_dzi('png')


# In[ ]:


format(zoom)


# In[ ]:


zoom.get_tile(13, (2, 5))


# In[ ]:


zoom.get_tile_coordinates(13, (2, 5))


# In[ ]:


zoom.get_tile_dimensions(13, (2, 5))


# In[ ]:




