#!/usr/bin/env python
# coding: utf-8

# # Create labels from the RLE encoded masks.

# # Import files

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np 
import pandas as pd
import os
import cv2
import time
import zipfile

import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm_notebook as tq
from PIL import Image


# # Mask to image

# In[ ]:


start = time.time()

input_dir = "../input/"
train_img_dir = "../input/train_images/"

category_num = 4 + 1

def make_mask_img(segment_df):
    seg_width = 1600
    seg_height = 256
    seg_img = np.full(seg_width*seg_height, category_num-1, dtype=np.int32)
    for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):
        if pd.isna(encoded_pixels): continue
        pixel_list = list(map(int, encoded_pixels.split(" ")))
        for i in range(0, len(pixel_list), 2):
            start_index = pixel_list[i] -1 
            index_len = pixel_list[i+1] 
            seg_img[start_index:start_index+index_len] = int(class_id) - 1
    seg_img = seg_img.reshape((seg_height, seg_width), order='F')
   
    return seg_img


# In[ ]:


train_df = pd.read_csv(input_dir + "train.csv")
train_df[['ImageId', 'ClassId']] = train_df['ImageId_ClassId'].str.split('_', expand=True)
train_df.head()


# In[ ]:


train_df.shape


# In[ ]:


images = train_df["ImageId"].unique() 
images


# It takes around 13 min to create labels

# In[ ]:


get_ipython().system('mkdir -p "/kaggle/working/labels-np/"')
get_ipython().system('mkdir -p "/kaggle/working/labels-img/"')


# In[ ]:


zip_np = zipfile.ZipFile('labels-np.zip', 'w', zipfile.ZIP_DEFLATED)
zip_img = zipfile.ZipFile('labels-img.zip', 'w', zipfile.ZIP_DEFLATED)

for image in images:
    df = train_df[train_df['ImageId']==image]
    mask = make_mask_img(df)
    npf = "/kaggle/working/labels-np/" + image.split('.')[0]
    imgf = "/kaggle/working/labels-img/"+ image.split('.')[0] + '.png'
    
    np.save(npf, mask)
    zip_np.write(npf + ".npy", image.split('.')[0] + ".npy")
    
    img_mask_3_chn = np.dstack((mask, mask, mask))
    cv2.imwrite(imgf, img_mask_3_chn)
    zip_img.write(imgf, image.split('.')[0] + '.png')
    
    os.remove(npf + ".npy")
    os.remove(imgf)
    
zip_np.close()
zip_img.close()


# In[ ]:


# !unzip -l labels-np.zip | less


# In[ ]:


end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("Execution Time  {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


# In[ ]:




