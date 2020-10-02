#!/usr/bin/env python
# coding: utf-8

# ## Divide images into 4 pieces & Merge randomly
# 
# Inspired by [@peter](https://www.kaggle.com/c/global-wheat-detection/discussion/149805)'s awesome article, I generated some images which is made by spliting & randomly merging train_input_images. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if 'jpg' in filename:
            continue
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import gc
import os
import time
import random
import collections
import uuid

from glob import glob
from datetime import datetime

import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)


# This code is from [@shonenkov/training-efficientdet](https://www.kaggle.com/shonenkov/training-efficientdet/data)

# In[ ]:


df = pd.read_csv('../input/global-wheat-detection/train.csv')

bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
for i, column in enumerate(['x', 'y', 'w', 'h']):
    df[column] = bboxs[:,i]
df.drop(columns=['bbox'], inplace=True)


# In[ ]:


df.head()


# ## Augmentation
# 1. divide an image into 4 pieces(puzzles) with bbox
# 2. make pool of puzzles of all original images (in train_data)
# 3. sample 4 pieces & merge (repeat k-times)
# 4. save as a dataframe & jpg images

# In[ ]:


def adjust_bbox_in_image(
    i_w,
    i_h,
    bbox,
    min_bbox_size=20):
    """crop bbox if it cover the edge of the cropped image"""
    
    x, y, w, h = bbox
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    if i_w < x+w:
        w -= (x+w-i_w)
    if i_h < y+h:
        h -= (y+h-i_h)
    
    on_border = x < 3 or y < 3 or i_w-3 < (x+w) or i_h-3 < (y+h)
    under_min_size = w < min_bbox_size or h < min_bbox_size
    if on_border and under_min_size:
        return None
    
    return (x, y, w, h)


def make_puzzles(
    image_id,
    bboxes,
    min_bbox_size=20,
    image_root='/kaggle/input/global-wheat-detection/train/'):
    """divide given image into 4 pieces with bboxes"""

    img_path = os.path.join(image_root, '{}.jpg'.format(image_id))
    
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    row, col, _ = image.shape
    
    y_div = int(row/2)
    x_div = int(col/2)
    lt = image[:y_div, :x_div] # left top
    rt = image[:y_div, x_div:] # right top
    lb = image[y_div:, :x_div] # left bottom
    rb = image[y_div:, x_div:] # right bottom
    
    lt_bboxes, rt_bboxes, lb_bboxes, rb_bboxes = [], [], [], []
    
    for bbox in bboxes:
        x, y, w, h = bbox
        
        # Quandrant-2
        lt_x, lt_y = x, y
        if lt_y < y_div and lt_x < x_div:
            _bbox = adjust_bbox_in_image(col/2, row/2, bbox, min_bbox_size=min_bbox_size)
            if _bbox:
                lt_bboxes.append(_bbox)
        
        # Quandrant-1
        rt_x, rt_y = x+w-1, y
        if rt_y < y_div and x_div <= rt_x:
            _bbox = (bbox[0]-col/2, bbox[1], bbox[2], bbox[3])
            _bbox = adjust_bbox_in_image(col/2, row/2, _bbox, min_bbox_size=min_bbox_size)
            if _bbox:
                rt_bboxes.append(_bbox)
            
        # Quandrant-3
        lb_x, lb_y = x, y+h-1
        if y_div <= lb_y and lb_x < x_div:
            _bbox = (bbox[0], bbox[1]-row/2, bbox[2], bbox[3])
            _bbox = adjust_bbox_in_image(col/2, row/2, _bbox, min_bbox_size=min_bbox_size)
            if _bbox:
                lb_bboxes.append(_bbox)
        
        # Quandrant-4
        rb_x, rb_y = x+w-1, y+h-1
        if y_div <= rb_y and x_div <= rb_x:
            _bbox = (bbox[0]-col/2, bbox[1]-row/2, bbox[2], bbox[3])
            _bbox = adjust_bbox_in_image(col/2, row/2, _bbox, min_bbox_size=min_bbox_size)
            if _bbox:
                rb_bboxes.append(_bbox)
    
    puzzle_bbox_pairs = [
        (lt, lt_bboxes),
        (rt, rt_bboxes),
        (lb, lb_bboxes),
        (rb, rb_bboxes)
    ]
    
    return puzzle_bbox_pairs


def merge_random_4_puzzles(puzzle_bbox_pairs):
    lt_img, lt_bboxes = puzzle_bbox_pairs[0]
    rt_img, rt_bboxes = puzzle_bbox_pairs[1]
    lb_img, lb_bboxes = puzzle_bbox_pairs[2]
    rb_img, rb_bboxes = puzzle_bbox_pairs[3]
    
    row, col, ch = lt_img.shape
    x_div = col
    y_div = row
    
    empty_img = np.zeros((row*2, col*2, ch), np.uint8)
    
    empty_img[:y_div,:x_div,:] = lt_img
    empty_img[:y_div,x_div:,:] = rt_img
    empty_img[y_div:,:x_div,:] = lb_img
    empty_img[y_div:,x_div:,:] = rb_img
    
    _lt_bboxes = lt_bboxes[:]
    _rt_bboxes = rt_bboxes[:]
    for i, bbox in enumerate(_rt_bboxes):
        x, y, w, h = bbox
        _rt_bboxes[i] = (x+x_div, y, w, h)
    
    _lb_bboxes = lb_bboxes[:]
    for i, bbox in enumerate(_lb_bboxes):
        x, y, w, h = bbox
        _lb_bboxes[i] = (x, y+y_div, w, h)
    
    _rb_bboxes = rb_bboxes[:]
    for i, bbox in enumerate(_rb_bboxes):
        x, y, w, h = bbox
        _rb_bboxes[i] = (x+x_div, y+y_div, w, h)
        
    merged_bbox = _lt_bboxes + _rt_bboxes + _lb_bboxes + _rb_bboxes
    
    return (empty_img, merged_bbox)
    

def visualize_4_image_bbox(puzzle_bbox_pairs):
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax = ax.flatten()

    labels = ['left top', 'right top', 'left bot', 'right bot']

    for i in range(4):
        image, bboxes = puzzle_bbox_pairs[i]
        for row in bboxes:
            x, y, w, h = (int(n) for n in row)
            cv2.rectangle(image,
                          (x, y),
                          (x+w, y+h),
                          (220, 0, 0), 3)
        ax[i].set_axis_off()
        ax[i].imshow(image)
        ax[i].set_title(labels[i], color='yellow')

        
def visualize_image_bbox(image, bboxes):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    for row in bboxes:
        x, y, w, h = (int(n) for n in row)
        cv2.rectangle(image,
                      (x, y),
                      (x+w, y+h),
                      (220, 0, 0), 3)
    ax.imshow(image)


# ## Make puzzle pool

# In[ ]:


from tqdm import tqdm


# In[ ]:


pool = []
image_ids = set(df.image_id.values)

for i, image_id in tqdm(enumerate(image_ids)):
    
    filtered = df[df['image_id'] == image_id]
    bboxes = filtered[['x', 'y', 'w', 'h']].values
    
    puzzles = make_puzzles(image_id, bboxes, min_bbox_size=20)
    
    pool += puzzles


# ## Generate Augmented k-Images

# In[ ]:


k = 1000


# In[ ]:


AugData = collections.namedtuple('AugData', 'image_id,width,height,source,x,y,w,h')


# In[ ]:


aug_data = []

os.makedirs('./merged_puzzles', exist_ok=True)

for i in tqdm(range(k)):
    random.shuffle(pool)
    a = ([bbox for img, bbox in pool[:4]])
    merged_image, merged_bboxes = merge_random_4_puzzles(pool[:4])
    ih, iw, ch = merged_image.shape
    image_id = str(uuid.uuid4())
    for bbox in merged_bboxes:
        x, y, w, h = bbox
        aug_data.append(AugData(image_id=image_id, width=iw, height=ih, source='aug', x=x, y=y, w=w, h=h))
    merged_image = cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./merged_puzzles/{}.jpg'.format(image_id), merged_image)        


# In[ ]:


aug_data[:10]


# In[ ]:


aug_df = pd.DataFrame(data=aug_data)
aug_df.to_csv('merged_puzzles.csv', index=False)


# In[ ]:




