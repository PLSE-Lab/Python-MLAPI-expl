#!/usr/bin/env python
# coding: utf-8

# # How to recover rotating bounging boxes from given data. Lossless. <br>
# I'm trying to code as simple as possible.

# In[ ]:


import numpy as np 
import pandas as pd 
from skimage.data import imread
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2


# In[ ]:


# set images to bigger size
mpl.rcParams['figure.figsize'] = [8.0, 8.0]


# In[ ]:


ImageId = '002fdcf51.jpg'
img = imread('../input/train_v2/' + ImageId)
masks = pd.read_csv("../input/train_ship_segmentations_v2.csv", index_col="ImageId")


# ## First, what image we have:

# In[ ]:


plt.imshow(img)
plt.show()


# ## Get mask of image from RLE
# How to turn RLE into image mask you probably know from other kernels. Nothing special here.

# In[ ]:


def rle_decode(mask_rle, shape):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


# In[ ]:


rle_mask = masks.EncodedPixels[ImageId].tolist()[1] # this image has two ships, we'll use bigger one
mask = rle_decode(rle_mask, (768, 768))
plt.imshow(mask)
plt.show()


# ## Bounding box
# Now let's encode this ship into box with x, y, width, height:

# In[ ]:


x, y, w, h = cv2.boundingRect(mask)
rect1 = cv2.rectangle(img.copy(),(x,y),(x+w,y+h),(0,255,0),3) # not copying here will throw an error
print("x:{0}, y:{1}, width:{2}, height:{3}".format(x, y, w, h))
plt.imshow(rect1)
plt.show()


# ## Rotating Bounding Box
# And finally

# In[ ]:


_,contours,_ = cv2.findContours(mask.copy(), 1, 1) # not copying here will throw an error
rect = cv2.minAreaRect(contours[0]) # basically you can feed this rect into your classifier
(x,y),(w,h), a = rect # a - angle


# Draw it

# In[ ]:


box = cv2.boxPoints(rect)
box = np.int0(box) #turn into ints
rect2 = cv2.drawContours(img.copy(),[box],0,(0,0,255),3)

plt.imshow(rect2)
plt.show()


# # Voila !
