#!/usr/bin/env python
# coding: utf-8

# ## Motivation
# 
# The Global Wheat Detection dataset is collected from different places and they are different. Let's see some images from different sources.

# In[ ]:


import os.path as osp
import pandas as pd

data_dir = '/kaggle/input/global-wheat-detection'

labels = pd.read_csv(osp.join(data_dir, 'train.csv'))
sources = labels.source.unique()


# In[ ]:


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def get_source_labels(source):
    result = dict()
    
    source_labels = labels[labels.source == source]
    source_image_ids = source_labels.image_id.unique()
    for image_id in source_image_ids:
        image_labels = labels[labels.image_id == image_id]
        assert np.all(image_labels.source == source)
        
        width = image_labels.width.iloc[0]
        assert np.all(image_labels.width == width)
        
        height = image_labels.height.iloc[0]
        assert np.all(image_labels.height == height)
        
        bboxes = [list(map(int, eval(bbox))) for bbox in image_labels.bbox]
        
        result[image_id] = {
            'width': width,
            'height': height,
            'bboxes': bboxes
        }
        
    return result

def show_random_images(source_labels, source, nrows=2, ncols=4, gt=True):
    image_ids = np.random.choice(list(source_labels.keys()), nrows * ncols).reshape((nrows, ncols))
    
    f, axes = plt.subplots(nrows, ncols)
    f.set_figwidth(ncols * 10)
    f.set_figheight(nrows * 10)
    
    for row, (row_ids, row_axes) in enumerate(zip(image_ids, axes)):
        for col, (image_id, ax) in enumerate(zip(row_ids, row_axes)):
            image = cv.imread(osp.join(data_dir, 'train', image_id + '.jpg'), cv.IMREAD_COLOR)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            if gt:
                bboxes = source_labels[image_id]['bboxes']
                for x, y, w, h in bboxes:
                    cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            ax.imshow(image)
            ax.set_axis_off()
            
    plt.suptitle('{} random images for {}'.format(nrows * ncols, source), fontsize=40)
    plt.show()


# In[ ]:


for source in sources:
    source_labels = get_source_labels(source)
    show_random_images(source_labels, source)


# As we can see the sources are very different from each other. The main interest for object detection is bounding boxes as we compute the loss for each bounding box. Therefore we should be careful with differences in bounding boxes between different sources. 
# 
# 
# ## Imbalance
# 
# First, lets see on the number of images and bounding bboxes for different sources.

# In[ ]:


sources, images_count, bboxes_count = [], [], []
for source in labels.source.unique():
    source_labels = labels[labels.source == source]
    sources.append(source)
    images_count.append(source_labels.image_id.unique().shape[0])
    bboxes_count.append(source_labels.shape[0])    

fig, axes = plt.subplots(1, 2)
fig.set_figwidth(20)
fig.set_figheight(10)

axes[0].pie(images_count, labels=sources)
axes[0].set_title('Images count', fontsize=30)

axes[1].pie(bboxes_count, labels=sources)
axes[1].set_title('Bboxes count', fontsize=30)

plt.show()


# There is high imbalance in images count and bboxes count. It means that our model will suffer on inrae_1, arvalis_3 and usask_1 sources. The easiest way to fix this is upsampling.

# In[ ]:


source_multipliers = {source: int(np.round(np.max(images_count) / image_count)) 
                      for source, image_count in zip(sources, images_count)}
source_multipliers


# We can simply increase images count for small sources just creating 5 new copies for inrae_1 (1 existing + 5 new = 6 total). This will give us images and bboxes distributions much more balanced.

# In[ ]:


multiplied_images_counts, multiplied_bboxes_counts = [], []
for source, image_count, bb_count, multiplier in zip(sources, images_count, bboxes_count, list(source_multipliers.values())):
    multiplied_images_counts.append(multiplier * image_count)
    multiplied_bboxes_counts.append(multiplier * bb_count)
    
fig, axes = plt.subplots(1, 2)
fig.set_figwidth(20)
fig.set_figheight(10)

axes[0].pie(multiplied_images_counts, labels=sources)
axes[0].set_title('Multiplied images count', fontsize=30)

axes[1].pie(multiplied_bboxes_counts, labels=sources)
axes[1].set_title('Multiplied bboxes count', fontsize=30)

plt.show()


# This simple procedure increased my LB score by 0.03.

# ## Bounding box coverage
# 
# One more interesting thing is to see bboxes distribution along images for different sources.

# In[ ]:


def show_bbox_coverage(source_labels, source):
    result = np.zeros((1024, 1024), dtype=np.uint32)
    for image_meta in source_labels.values():
        for x, y, w, h in image_meta['bboxes']:
            result[y: y + h, x: x + w] += 1
            
    plt.figure(figsize=(10, 10))
    plt.matshow(result, fignum=0)
    plt.title('Bboxes coverage ' + source, fontsize=20)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# In[ ]:


for source in sources:
    source_labels = get_source_labels(source)
    show_bbox_coverage(source_labels, source)


# We can also see some imbalance in bboxes distribution. We can try to use flips augmentation as a baseline to fix it.
# 
# Good luck!

# In[ ]:




