#!/usr/bin/env python
# coding: utf-8

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
import glob


# As the organizer has already mentioned that 
# 
# **train_label_masks: Segmentation masks showing which parts of the image led to the ISUP grade. Not all training images have label masks.**
# 
# Therefore, this simple notebook aims to check how many images in the training images folder that are not labelled with masks.

# In[ ]:


train = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')
train_images_path = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'
train_label_mask_path = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/'
img_type = "*.tiff"


# In[ ]:


def sanity_tally(train_images_path, train_label_mask_path, img_type):
    total_img_list = [os.path.basename(img_name) for img_name
                      in glob.glob(os.path.join(train_images_path, img_type))]
    ## get the image_name
    total_img_list = [x[:-5] for x in total_img_list]
    
    mask_img_list  = [os.path.basename(img_name) for img_name 
                      in glob.glob(os.path.join(train_label_mask_path, img_type))]
    
    # note that the image name in train_label_mask will always be in this format: abcdefg_mask.tiff; therefore I needed to
    # remove the last 10 characters to tally with the images in train_images.
    mask_img_list  = [x[:-10] for x in mask_img_list]
    set_diff1      = set(total_img_list) - set(mask_img_list)
    set_diff2      = set(mask_img_list)  - set(total_img_list)
    
    if set(total_img_list)  == set(mask_img_list):
        print("Sanity Check Status: True")
    else:
        print("Sanity Check Status: Failed. \nThe elements in train_images_path but not in the train_label_mask_path is {} and the number is {}.\n\n\nThe elements in train_label_mask_path but not in train_images_path is {} and the number is {}".format(
                set_diff1, len(set_diff1), set_diff2, len(set_diff2)))
    
    return set_diff1, set_diff2


# In[ ]:


set_diff1, set_diff2 = sanity_tally(train_images_path,train_label_mask_path, img_type)


# From the sanity check above, and assuming that each image's name in the train_images should necessarily match the ones in train_label_masks, we can deduce that the masked images in train_label_masks is a subset of the images in the train_images. That is to say, all masked images in train_label_masks has a corresponding image in the train_images, but there exists 100 images in train_images that do not have a mask. Whether we decide to keep these "un-labelled" images is up to you to decide.

# For simplicity sake, I want to have a bijective relationship between the train_images folder and the train_label_masks folder; and since the set difference is small (100), I can make do to delete these 100 images that are not **annotated** by the pathologists.

# In[ ]:


remove_images = list(set_diff1)
new_train = train[~train.image_id.isin(remove_images)]


# In[ ]:


new_train = new_train.reset_index(drop=True)


# In[ ]:


new_train


# Next, I conveniently borrowed [xhlulu's panda resize and save train data kernel](https://www.kaggle.com/xhlulu/panda-resize-and-save-train-data) and save the image as png file, where all of them are resized to 512x512.

# In[ ]:


save_dir = "/kaggle/train/"
os.makedirs(save_dir, exist_ok=True)

train_images_path = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'
train_label_mask_path = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/'
img_type = "*.tiff"


# In[ ]:


for img_id in tqdm(new_train.image_id):
    load_path = train_images_path + img_id + '.tiff'
    save_path = save_dir + img_id + '.jpg'
    
    biopsy = skimage.io.MultiImage(load_path)
    img = cv2.resize(biopsy[-1], (512, 512))
    cv2.imwrite(save_path, img)


# In[ ]:


for img_id in tqdm(new_train.image_id):
    load_path = train_label_mask_path + img_id + '_mask' + '.tiff'
    save_path = save_dir + img_id + '.jpg'
    
    biopsy = skimage.io.MultiImage(load_path)
    img = cv2.resize(biopsy[-1], (512, 512))
    cv2.imwrite(save_path, img)


# In[ ]:


get_ipython().system('tar -czf images.tar.gz ../train/*.png')

