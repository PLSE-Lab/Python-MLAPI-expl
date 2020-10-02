#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[ ]:


DATASET_ROOT_PATH = "/kaggle/input/turkish-lira-banknote-dataset"

# Get all images
files = sorted(glob.glob(os.path.join(DATASET_ROOT_PATH, "**/*.png")))


# In[ ]:


len(files)    # Total 6000 images


# In[ ]:


import cv2

img = cv2.imread(files[0])
height, width, channel = img.shape
    
print(f"Height: {height} Width: {width} Channel: {channel}")
        


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
def show_image(img_path):
    img = mpimg.imread(img_path)
    plt.figure()
    plt.imshow(img)


# In[ ]:


image_idxs = [0, 1000, 2000, 3000, 4000, 5000]

for image_idx in image_idxs:
    print(f"File path: {files[image_idx]}")
    show_image(files[image_idx])

