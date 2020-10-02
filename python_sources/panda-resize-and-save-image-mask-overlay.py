#!/usr/bin/env python
# coding: utf-8

# Dataset available here: https://www.kaggle.com/rohitsingh9990/image-mask-overlay-512x512

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
import PIL
from skimage.transform import resize, rescale


# ## Load dataframe

# In[ ]:


train = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')


# In[ ]:


train.head()


# In[ ]:


data_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'
mask_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/'
images = os.listdir(mask_dir)


# ## Start here

# In[ ]:


def overlay_mask_on_slide(image_id, center='radboud', alpha=0.8, max_size=(800, 800)):
    """Show a mask overlayed on a slide."""
    
    
    slide = openslide.OpenSlide(os.path.join(data_dir, f'{image_id}.tiff'))
    mask = openslide.OpenSlide(os.path.join(mask_dir, f'{image_id}_mask.tiff'))
    slide_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])
    mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
    mask_data = mask_data.split()[0]
        
        
    # Create alpha mask
    alpha_int = int(round(255*alpha))
    if center == 'radboud':
        alpha_content = np.less(mask_data.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)
    elif center == 'karolinska':
        alpha_content = np.less(mask_data.split()[0], 1).astype('uint8') * alpha_int + (255 - alpha_int)

    alpha_content = PIL.Image.fromarray(alpha_content)
    preview_palette = np.zeros(shape=768, dtype=int)

    if center == 'radboud':
        # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}
        preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)
    elif center == 'karolinska':
        # Mapping: {0: background, 1: benign, 2: cancer}
        preview_palette[0:9] = (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]) * 255).astype(int)

    mask_data.putpalette(data=preview_palette.tolist())
    mask_rgb = mask_data.convert(mode='RGB')
    overlayed_image = PIL.Image.composite(image1=slide_data, image2=mask_rgb, mask=alpha_content)
    overlayed_image.thumbnail(size=max_size, resample=0)
    
    slide.close()
    mask.close()   
          
    return overlayed_image


# In[ ]:


mkdir train_overlay_images


# In[ ]:


save_dir = "train_overlay_images/"
os.makedirs(save_dir, exist_ok=True)


for img_id in tqdm(images[:5]):
    img_id = img_id.replace('_mask.tiff', '')
    save_path = save_dir + img_id + '.png'
    provider = train[train.image_id == img_id]['data_provider'].values[0]
    overlay = overlay_mask_on_slide(img_id, center = provider)
    img = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (512, 512))
    cv2.imwrite(save_path, img)


# In[ ]:


get_ipython().system('tar -czf train_overlay_images.tar.gz train_overlay_images/*.png')


# If you like this kernel also visit following kernels:
# * https://www.kaggle.com/xhlulu/panda-resize-and-save-train-data
# * For EDA: https://www.kaggle.com/rohitsingh9990/panda-eda-better-visualization-simple-baseline
# * For ResNext Inference: https://www.kaggle.com/rohitsingh9990/panda-resnext-inference

# In[ ]:




