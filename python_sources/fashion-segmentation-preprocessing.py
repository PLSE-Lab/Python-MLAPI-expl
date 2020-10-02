#!/usr/bin/env python
# coding: utf-8

# "developed by : aipythoner@gmail.com" 
# 
# I'm a tensorflow(keras) user and try to learn how fastai library work, but prefer to do it on realworld datasets
# 
# Developed model and results can be found on my [Github](https://github.com/pykeras?tab=repositories)

# In[ ]:


import numpy as np
import pandas as pd
import glob, os, json, cv2, skimage, shutil
from fastai import *
from fastai.vision import * 
from progressbar import ProgressBar
from skimage import io


# In[ ]:


train_image_path = glob.glob('../input/imaterialist-fashion-2020-fgvc7/train/*.jpg')
train_csv = pd.read_csv('../input/imaterialist-fashion-2020-fgvc7/train.csv', index_col=['ImageId'])
train_csv.head(1)
# train_image_path = train_image_path[0:5]


# In[ ]:


with open('../input/imaterialist-fashion-2020-fgvc7/label_descriptions.json') as f:
    label_descriptions = json.load(f)
    
label_names = [x['name'] for x in label_descriptions['categories']]

 # 1 for background


# 46 apparel objects (27 main apparel items and 19 apparel parts) from description

# In[ ]:


num_categories = 27 + 1 #(add 1 for background)


# In[ ]:


def create_mask(df):
    mask_h = df.at[0, 'Height']
    mask_w = df.at[0, 'Width']
    mask = np.full(mask_w*mask_h, num_categories-1, dtype=np.int32)
    
    for encode_pixels, encode_labels in zip(df.EncodedPixels.values, df.ClassId.values):
        pixels = list(map(int, encode_pixels.split(' ')))
        for i in range(0,len(pixels), 2):
            start_pixel = pixels[i]-1 #index from 0
            len_mask = pixels[i+1]-1
            end_pixel = start_pixel + len_mask
            if int(encode_labels) < num_categories - 1:
                mask[start_pixel:end_pixel] = int(encode_labels)
            
    mask = mask.reshape((mask_h, mask_w), order='F')
    return mask


# In[ ]:


if not os.path.exists('./masks'):
    os.makedirs('./masks')


# In[ ]:


for file in ProgressBar()(train_image_path):
    file_name = file.split('/')[-1]
    file_id = file_name.split('.')[0]
    df = train_csv.loc[file_id]
    if "Series" in str(type(df)):
        df = DataFrame([df.to_list()],  columns=['EncodedPixels', 'Height',                                                   'Width','ClassId',  'AttributesIds'])
        
    try:
        mask = create_mask(df.reset_index())
    except:
        print(file_id)
    
    mask_rgb = np.dstack((mask, mask, mask))
    cv2.imwrite('./masks/'+file_id+'.png', mask_rgb)

#     plt.imsave('./masks/'+file_name, mask)

#     io.imsave('./masks/'+file_name, mask)


# In[ ]:


img_path = '../input/imaterialist-fashion-2020-fgvc7/train/'
label_path = '../input/fashion-segmentation-preprocessing/masks/'
get_label = lambda x: label_path + (str(x).split('/')[-1]).split('.')[0] + '.png'


# In[ ]:


img = open_image(train_image_path[1])
mask = open_mask(get_label(train_image_path[1]))
_,axs = plt.subplots(1,3, figsize=(10,10))
img.show(ax=axs[0], title='no mask')
img.show(ax=axs[1], y=mask, title='masked')
mask.show(ax=axs[2], title='mask only', alpha=1.)

