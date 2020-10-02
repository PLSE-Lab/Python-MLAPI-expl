#!/usr/bin/env python
# coding: utf-8

# # Grouping images by hue, nuclei number and nuclei overlapping
# 
# This is an attempt to group the training images by generating some features and grouping them by using the K-means algorithm.
# 
# The goal was to group the images by color levels (hue and value) and level of nuclei detection difficulty. We estimate the level of difficulty by measuring the total number of masks and the total number of isolated masks in the union of masks.
# 
# I hope it can be used as a starting point for those interested in trying to partition the data themselves.

# In[1]:


import numpy as np
import pandas as pd
import glob
import os
import tqdm
from matplotlib import pyplot as plt
from skimage.io import imread 
from skimage.color import rgb2hsv
from sklearn.cluster import KMeans
from skimage.morphology import label as sk_label

id_list = glob.glob(os.path.join('..','input','stage1_train','[0-9a-z]*'))
id_list = [ os.path.basename(i) for i in id_list ]
print('Number of train images: %s'%(len(id_list)))


# In[2]:


def count_masks(id, height, width):
    mask_list = glob.glob(os.path.join('..','input','stage1_train',id,'masks','*.png'))
    mask_list = [ os.path.basename(i) for i in mask_list]
    mask = np.zeros((height,width))
    n_masks = len(mask_list)
    for i,m in enumerate(mask_list):
        mask = np.maximum(mask, imread(os.path.join('..','input','stage1_train',id,'masks',m)))
    mask = mask>0
    return n_masks, sk_label(mask).max()


# In[3]:


image_size   = np.empty((len(id_list),2))
image_hsv    = np.empty((len(id_list),3))
image_masks  = np.empty((len(id_list)))
image_masks2 = np.empty((len(id_list)))
images = []
for i,ids in tqdm.tqdm(enumerate(id_list)):
    img = imread(os.path.join('..','input','stage1_train',ids,'images',ids+'.png'))
    images.extend([img])
    assert img.shape[2]==4
    image_size[i,...] = img.shape[0:2]
    img_hsv = rgb2hsv(img[...,0:3])
    image_hsv[i,...] = [np.mean(img_hsv[:,:,0])  ,np.mean(img_hsv[:,:,1]) , np.mean(img_hsv[:,:,2])]
    image_masks[i], image_masks2[i] = count_masks(ids, img.shape[0], img.shape[1])

df = pd.DataFrame({'Id':id_list,
                   'images':images,
                   'height':image_size[:,0],
                   'width':image_size[:,1],
                   'H':image_hsv[:,0],
                   'S':image_hsv[:,1],
                   'V':image_hsv[:,2],
                   'nmask':image_masks,
                   'nmask2':image_masks2})


# In[4]:


df['mask_diff'] = (df['nmask'] - df['nmask2'])/df['nmask']
df['nmask_norm'] = df['nmask']/80
kmeans = KMeans(n_clusters=8,random_state=2018).fit(np.array(df[['nmask_norm','H','V','mask_diff']]))
df['cluster'] = kmeans.labels_


# In[5]:


df.groupby('cluster').mean()


# In[6]:


def show_cluster(cl):
    df_e = df[df['cluster']==cl].reset_index()
    plt.figure(figsize=(20,2*(df_e.shape[0]//8+1)))
    for i in range(df_e.shape[0]):
        plt.subplot(df_e.shape[0]//8+1,8,i+1)
        plt.imshow(df_e['images'][i])
    plt.show()


# In[ ]:


show_cluster(0)


# In[ ]:


show_cluster(1)


# In[ ]:


show_cluster(2)


# In[ ]:


show_cluster(3)


# In[ ]:


show_cluster(4)


# In[ ]:


show_cluster(5)


# In[ ]:


show_cluster(6)


# In[ ]:


show_cluster(7)

