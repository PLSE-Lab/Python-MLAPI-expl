#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import cv2
import pandas as pd
import numpy as np
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage import data
from skimage.color import label2rgb
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# In[5]:


image_path = "../input/stage2_test_final/0a8dfb6763ec049b57454e6b5605f872cdf41cd13951fb2a2f31a6e3c8667712/images/0a8dfb6763ec049b57454e6b5605f872cdf41cd13951fb2a2f31a6e3c8667712.png"


# In[6]:


image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# In[7]:


plt.imshow(image)
plt.show()


# In[8]:


# apply threshold
thresh = threshold_otsu(image)
if thresh:
    bw = closing(image > thresh, square(3))
else:
    ret, bw = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)


# In[9]:


print(bw)
plt.imshow(bw)
print(thresh)


# In[10]:


def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
#     print(run_lengths)
    return run_lengths


# In[13]:


# label image regions
label_image = label(bw)
plt.imshow(label_image)
imag_label_overlay = label2rgb(label_image, image=image)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(imag_label_overlay)

for region in regionprops(label_image):
    # take regions with large enough areas
    temp_image_null = np.zeros(label_image.shape)
    for coord in region.coords:
        temp_image_null[coord[0]][coord[1]] = 1
#     print(temp_image_null)
    plt.imshow(temp_image_null)
    plt.show()
    rle_encoding(temp_image_null)
        
ax.set_axis_off()
plt.tight_layout()
plt.show()


# In[14]:


import glob


# ###  Now using clustering for different nuclei in the cells
# #### Storing the run length encoded results in clustering_submit.csv

# In[16]:


df_array = []
image_count = 0
for image in glob.glob("../input/stage2_test_final/*/images/*"):
    image_count += 1
    image_path = image
    
    # convert image to bw
    input_image = cv2.imread(image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    # apply threshold
    thresh = threshold_otsu(input_image)
    if thresh:
        bw = closing(input_image > thresh, square(3))
    else:
        ret, bw = cv2.threshold(input_image, 0, 255, cv2.THRESH_BINARY)
        
    # label image regions
    label_image = label(bw)

    count = 0
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area > 70:
            count += 1
            temp_image_null = np.zeros(label_image.shape)
            for coord in region.coords:
                temp_image_null[coord[0]][coord[1]] = 1
                
            if len(np.unique(temp_image_null)) > 1:
                array = rle_encoding(temp_image_null)
                rle_string = str(array).replace(",", "").strip("[").strip("]")
                temp_df = pd.DataFrame([{"ImageID": image_path.split("/")[-1].split(".")[0], "EncodedPixels": rle_string}], columns=["ImageID", "EncodedPixels"])
                df_array.append(temp_df)
#                 print(temp_df)
    print(image_count, image_path.split("/")[-1].split(".")[0], count)
df = pd.concat(df_array, ignore_index=True)


# In[17]:


df.head(10)


# In[18]:


df.to_csv("submission.csv", index=False)


# In[ ]:




