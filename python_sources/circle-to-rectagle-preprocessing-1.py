#!/usr/bin/env python
# coding: utf-8

# # During the exploration phase I found that the black areas around the images could be a problem.
# # So I looked for some ways to normalized this and found stretching works very well.
# # My initial idea was to transform the Circle to a Rectangle and I searched for some 2D morph libraries using maped triangles. This looks promissing to morphing faces but not for this task.
# # So I found a simple line-by-line approximation worked pretty well with this images. The idea is remove the left and right black portion of images stretching the non-black area to the rectangle limits.
# # This preprocessing showed to be very stable and improved our scores by 0.01+ :)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import glob

filenames = glob.glob( '../input/aptos2019-blindness-detection/train_images/*' )


# In[ ]:


def morph_func( img, subgauss=False ):
    img = img.astype( np.uint8 )

    findblack = np.sum( img, axis=2 )
    findblack = findblack - np.min(findblack)
    findblack[findblack > np.mean(findblack)] = np.mean(findblack)
    findblack = findblack / np.max(findblack)
    borders = 2+np.sum( findblack[:,:int(img.shape[1]/2)]<0.33, axis=1 )

    for i in range( img.shape[0] ):
        k = borders[i]
        img[i] = img[i, np.linspace(k,img.shape[1]-k,num=img.shape[1]).astype(int),: ]   
        
    img = cv2.resize( img, (512,512) )
    if subgauss==True:
        img = cv2.addWeighted(img,4, cv2.GaussianBlur(img, (0,0) , sigmaX=13) ,-4 ,127)
        
    return img

def print_preprocessing( fn ):
    img_circle = cv2.imread( fn )
    img_circle = cv2.cvtColor(img_circle, cv2.COLOR_BGR2RGB)
    img_circle = cv2.resize( img_circle, (512,512) )

    img_rect = morph_func(img_circle)

    plt.imshow( np.hstack((img_circle, img_rect))  )


# In[ ]:


print_preprocessing( filenames[0] )


# In[ ]:


print_preprocessing( filenames[11] )


# In[ ]:


print_preprocessing( filenames[111] )


# In[ ]:


print_preprocessing( filenames[1111] )


# In[ ]:


print_preprocessing( filenames[2222] )


# In[ ]:


print_preprocessing( filenames[3333] )


# In[ ]:


print_preprocessing( filenames[2019] )


# In[ ]:


print_preprocessing( filenames[908] )


# In[ ]:


print_preprocessing( filenames[5] )


# In[ ]:


print_preprocessing( filenames[6] )


# In[ ]:


print_preprocessing( filenames[7] )


# In[ ]:


print_preprocessing( filenames[9] )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




