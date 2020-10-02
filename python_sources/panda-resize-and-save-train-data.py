#!/usr/bin/env python
# coding: utf-8

# Dataset available here: https://www.kaggle.com/xhlulu/panda-resized-train-data-512x512

# In[ ]:


import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import openslide
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm.notebook import tqdm
import skimage.io
from skimage.transform import resize, rescale


# ## Load dataframe

# In[ ]:


train_labels = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')


# In[ ]:


train_labels.head()


# In[ ]:


data_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'


# In[ ]:


mask_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/'
mask_files = os.listdir(mask_dir)


# ## Speed tests

# In[ ]:


img_id = train_labels.image_id[0]
path = data_dir + img_id + '.tiff'


# In[ ]:


get_ipython().run_line_magic('time', 'biopsy = openslide.OpenSlide(path)')
get_ipython().run_line_magic('time', 'biopsy2 = skimage.io.MultiImage(path)')


# In[ ]:


get_ipython().run_line_magic('timeit', 'img = biopsy.get_thumbnail(size=(512, 512))')
get_ipython().run_line_magic('timeit', 'out = resize(biopsy2[-1], (512, 512))')
get_ipython().run_line_magic('timeit', 'out = cv2.resize(biopsy2[-1], (512, 512))')
get_ipython().run_line_magic('timeit', 'out = Image.fromarray(biopsy2[-1]).resize((512, 512))')


# In[ ]:


out = cv2.resize(biopsy2[-1], (512, 512))

get_ipython().run_line_magic('timeit', "Image.fromarray(out).save(img_id+'.png')")
get_ipython().run_line_magic('timeit', "cv2.imwrite(img_id+'.png', out)")


# Conclusion: skimage is fastest for loading, cv2 is fastest for resizing and saving.

# ### Try loading masks

# In[ ]:


mask = skimage.io.MultiImage(mask_dir + mask_files[1])
img = skimage.io.MultiImage(data_dir + mask_files[1].replace("_mask", ""))


# In[ ]:


mask[-1].shape, img[-1].shape


# ## Start here

# In[ ]:


save_dir_isup_grade_0 = "/kaggle/isup_grade_0"
save_dir_isup_grade_1 = "/kaggle/isup_grade_1"
save_dir_isup_grade_2 = "/kaggle/isup_grade_2"
save_dir_isup_grade_3 = "/kaggle/isup_grade_3"
save_dir_isup_grade_4 = "/kaggle/isup_grade_4"
save_dir_isup_grade_5 = "/kaggle/isup_grade_5"

os.makedirs(save_dir_isup_grade_0, exist_ok=True)
os.makedirs(save_dir_isup_grade_1, exist_ok=True)
os.makedirs(save_dir_isup_grade_2, exist_ok=True)
os.makedirs(save_dir_isup_grade_3, exist_ok=True)
os.makedirs(save_dir_isup_grade_4, exist_ok=True)
os.makedirs(save_dir_isup_grade_5, exist_ok=True)


# In[ ]:


for img_id, isup_grade in tqdm(zip(train_labels.image_id, train_labels.isup_grade)):
    load_path = data_dir + img_id + '.tiff'
    if(isup_grade == 0): save_path = save_dir_isup_grade_0 + img_id + '.png'
    if(isup_grade == 1): save_path = save_dir_isup_grade_1 + img_id + '.png'
    if(isup_grade == 2): save_path = save_dir_isup_grade_2 + img_id + '.png'
    if(isup_grade == 3): save_path = save_dir_isup_grade_3 + img_id + '.png'
    if(isup_grade == 4): save_path = save_dir_isup_grade_4 + img_id + '.png'
    if(isup_grade == 5): save_path = save_dir_isup_grade_5 + img_id + '.png'
        
    biopsy = skimage.io.MultiImage(load_path)
    img = cv2.resize(biopsy[-1], (512, 512))
    cv2.imwrite(save_path, img)


# In[ ]:


save_dir_mask_isup_grade_0 = "/kaggle/isup_grade_mask_0"
save_dir_mask_isup_grade_1 = "/kaggle/isup_grade_mask_1"
save_dir_mask_isup_grade_2 = "/kaggle/isup_grade_mask_2"
save_dir_mask_isup_grade_3 = "/kaggle/isup_grade_mask_3"
save_dir_mask_isup_grade_4 = "/kaggle/isup_grade_mask_4"
save_dir_mask_isup_grade_5 = "/kaggle/isup_grade_mask_5"

os.makedirs(save_dir_mask_isup_grade_0, exist_ok=True)
os.makedirs(save_dir_mask_isup_grade_1, exist_ok=True)
os.makedirs(save_dir_mask_isup_grade_2, exist_ok=True)
os.makedirs(save_dir_mask_isup_grade_3, exist_ok=True)
os.makedirs(save_dir_mask_isup_grade_4, exist_ok=True)
os.makedirs(save_dir_mask_isup_grade_5, exist_ok=True)


# In[ ]:


'''
for mask_file in tqdm(mask_files):
    load_path = mask_dir + mask_file
    save_path = save_mask_dir + mask_file.replace('.tiff', '.png')
    
    mask = skimage.io.MultiImage(load_path)
    img = cv2.resize(mask[-1], (512, 512))
    cv2.imwrite(save_path, img)
'''


# In[ ]:


get_ipython().system('tar -czf isup_grade_0.tar.gz ../isup_grade_0*.png')
get_ipython().system('tar -czf isup_grade_1.tar.gz ../isup_grade_1*.png')
get_ipython().system('tar -czf isup_grade_2.tar.gz ../isup_grade_2*.png')
get_ipython().system('tar -czf isup_grade_3.tar.gz ../isup_grade_3*.png')
get_ipython().system('tar -czf isup_grade_4.tar.gz ../isup_grade_4*.png')
get_ipython().system('tar -czf isup_grade_5.tar.gz ../isup_grade_5*.png')


# In[ ]:


get_ipython().system('tar -czf train_label_isup_grade_0_masks.tar.gz ../isup_grade_mask_0/*.png')
get_ipython().system('tar -czf train_label_isup_grade_1_masks.tar.gz ../isup_grade_mask_1/*.png')
get_ipython().system('tar -czf train_label_isup_grade_2_masks.tar.gz ../isup_grade_mask_2/*.png')
get_ipython().system('tar -czf train_label_isup_grade_3_masks.tar.gz ../isup_grade_mask_3/*.png')
get_ipython().system('tar -czf train_label_isup_grade_4_masks.tar.gz ../isup_grade_mask_4/*.png')
get_ipython().system('tar -czf train_label_isup_grade_5_masks.tar.gz ../isup_grade_mask_5/*.png')

