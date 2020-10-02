#!/usr/bin/env python
# coding: utf-8

# The purpose of this kernel is to get a better idea of what kind of data we are working with.

# In[ ]:


import numpy as np
import pandas as pd
import pydicom
import os
import matplotlib.pyplot as plt


# ## What are DICOMs??
# DICOMs are a format used for storing medical scanning data. They store a collection of metadata and image data. Let's have a look.

# In[ ]:


ds = pydicom.dcmread('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_000039fa0.dcm')
ds


# I think what we can see there that this is a CT scan (modality CT), the image stored is a 512 by 512 pixels. Not sure what the other fields mean yet.
# 
# The image is stored in the Pixel Data, let's take a look.

# In[ ]:


print('pixel_array:', ds.pixel_array)
print('center:',ds.pixel_array[206:306,206:306])
print('dimensions:', ds.pixel_array.shape)
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)


# Yep, that's a cranium. The image is indeed a 512 by 512 image but those are some odd pixel values.
# 
# Turns out they are probably [Hounsfield units](https://en.wikipedia.org/wiki/Hounsfield_scale) and it basically measures how much radiation is passing right through. Air has an HU value of -1000, bone has values between 300 and 400, or 1800 and 1900, depending on the type. Water is 0. The values we see here do not quite correspond to these because they need to be [rescaled](https://blog.kitware.com/dicom-rescale-intercept-rescale-slope-and-itk/).

# In[ ]:


rescale_intercept = ds[('0028','1052')].value
rescale_slope = ds[('0028','1053')].value

rescaled_hu = rescale_slope * ds.pixel_array + rescale_intercept


# The black areas in the image have an unscaled value of -2000. these actually provide us with no information. The outer grey areas have an unscaled value of 0, and a scaled value of -1024, which is basically the value of air. We can see this transition if we zoom in close enough.
# 
# We will do this with the unscaled data as the tansition is visually clearer.

# In[ ]:


print('Transition zone:')
print(ds.pixel_array[75:85,65:75])


# We can also take a look at the transition between the air and the cranium. The transition is a bit blurry, possibly due to hair.

# In[ ]:


print('Air-to-cranium transition zone:')
print(rescaled_hu[105:115,160:170])


# Honestly, the machine learning probably does not care that much about this rescaling, but it helps a bit in comprehension. The rescaled image looks the same as the unscaled one.

# In[ ]:


plt.imshow(rescaled_hu, cmap=plt.cm.bone)


# ## What is windowing?
# As best I can tell, windowing are some guidance values which help the viewer clarify which values to focus on. Essentially, if a value is beyond the "window", assign a max value to it.
# 
# In this image, we have a windowing center of 30 and a windowing width of 80, giving us an effective range of -10 to 70.

# In[ ]:


y_min = 0
y_max = 255
window_center = ds[('0028','1050')].value
window_width = ds[('0028','1051')].value

windowed_hu = rescaled_hu.copy()
min_val = window_center - window_width / 2
max_val = window_center + window_width / 2

# we want pixels with min_val to have score zero
windowed_hu = windowed_hu - min_val;

# we want pixels with the max value to have a score of 255
windowed_hu = windowed_hu * y_max / max_val

# we want to contrain all other values
windowed_hu = np.clip(windowed_hu, y_min, y_max)

## have a look
plt.imshow(windowed_hu, cmap=plt.cm.bone)


# Well then, those are eyes. 
# 
# I'm not sure this particular windowing is very helpful. 
# 
# Ryan Epp's [Gradient & Sigmoid Windowing](https://www.kaggle.com/reppic/gradient-sigmoid-windowing) kernel has a lot more useful forms of windowing, including using multiple windowing ranges and keeping them all as "channels" in an image.

# In[ ]:




