#!/usr/bin/env python
# coding: utf-8

# ### Overview
# In this kernel I want to demonstrate the content of "train_ship_segmentations_boxes.csv" dataset that I created. It is an extended version of "train_ship_segmentations.csv" file from Airbus Ship Detection Challenge in which in addition to pixel masks I included the information about rotating bounding boxes.

# In[ ]:


import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math


# In[ ]:


BOXES_PATH = '../input/rotating-bounding-boxes-for-ship-localization/train_ship_segmentations_boxes.csv'
IMG_PATH = '../input/airbus-ship-detection/train/'


# The meaning of columes is the following. "ImageId" is name of an image from train dataset (https://www.kaggle.com/c/airbus-ship-detection/data); "EncodedPixelsPixel" is Run-Length Encoding of the mask; "x" and "y" are X and Y coordinate of box center; "lx" and "ly" are length of the box along X and Y; and "angle" is the rotation angle of the box in radians.

# In[ ]:


box_df = pd.read_csv(BOXES_PATH)
box_df.head()


# In[ ]:


#convert RLE mask into 2d pixel array
def encode_mask(mask, shape=(768,768)):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    s = mask.split()
    for i in range(len(s)//2):
        start = int(s[2*i]) - 1
        length = int(s[2*i+1])
        img[start:start+length] = 1
    return img.reshape(shape).T

#get bounding box for a mask
def get_bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

#add padding to the bounding box
def get_bbox_p(img, padding=5):
    x1,x2,y1,y2 = get_bbox(img)
    lx,ly = img.shape
    x1 = max(x1-padding,0)
    x2 = min(x2+padding+1, lx-1)
    y1 = max(y1-padding,0)
    y2 = min(y2+padding+1, ly-1)
    return x1,x2,y1,y2

#convert parameters of the box for plotting
def convert_box(box):
    rot1 = math.cos(box[4])
    rot2 = math.sin(box[4])
    bx1 = box[0] - 0.5*(box[2]*rot1 - box[3]*rot2)
    bx2 = box[1] - 0.5*(box[2]*rot2 + box[3]*rot1)
    return (bx1,bx2,box[2],box[3],box[4]*180.0/math.pi)

def get_rec(box,width=1):
    b = convert_box(box)
    return patches.Rectangle((b[0],b[1]),b[2],b[3],b[4],linewidth=width,edgecolor='g',facecolor='none')


# In[ ]:


#plot image, mask, zoomed image, and zoomed mask with rotating bounding boxes
def show_box(idx):
    row = box_df.iloc[idx]
    name, encoding, x, y, lx, ly, rot = row.ImageId, row.EncodedPixels,         row.x, row.y, row.lx, row.ly, row.angle
    if(type(encoding) == float): return #empty image

    mask = encode_mask(encoding)
    box = (x,y,lx,ly,rot)
    image = np.asarray(Image.open(os.path.join(IMG_PATH,name)))
    
    fig,ax = plt.subplots(2, 2, figsize=(16, 16))
    ax[0,0].imshow(image)
    ax[0,1].imshow(mask)
    ax[0,0].add_patch(get_rec(box))
    ax[0,1].add_patch(get_rec(box))
    
    y1,y2,x1,x2 = get_bbox_p(mask,10)
    box_c = (x-x1,y-y1,lx,ly,rot)
    ax[1,0].imshow(image[y1:y2,x1:x2,:])
    ax[1,1].imshow(mask[y1:y2,x1:x2])
    ax[1,0].add_patch(get_rec(box_c,3))
    ax[1,1].add_patch(get_rec(box_c,3))
    
    for item in ax.flatten():
        item.axis('off')
    plt.show()


# Below I provide several examples of rotating bounding boxes.

# In[ ]:


show_box(20)


# In[ ]:


show_box(19)


# In[ ]:


show_box(11)


# In[ ]:


show_box(10)


# In[ ]:


show_box(31)


# In[ ]:


show_box(1)


# In[ ]:


show_box(39)


# In[ ]:




