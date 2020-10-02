#!/usr/bin/env python
# coding: utf-8

# The code below selects 20 256x256 tiles for each image and mask based on the maximum number of infected tissue pixels in masks. The kernel also provides computed image stats. Please check my kernels to see how to use this data. 
# ![](https://i.ibb.co/RzSWP56/convert.png)

# In[ ]:


import os, gc
import cv2
import skimage.io
import openslide
from tqdm.notebook import tqdm
import zipfile
import pandas as pd
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import PIL
from IPython.display import Image, display


# In[ ]:


TRAIN = '../input/prostate-cancer-grade-assessment/train_images/'
MASKS = '../input/prostate-cancer-grade-assessment/train_label_masks/'
OUT_TRAIN = 'train.zip'
OUT_MASKS = 'masks.zip'
sz = 256
N = 16
TIFF_LEVEL = 1


# In[ ]:


names = [name[:-10] for name in os.listdir(MASKS)]
# i = 9
# pos = (int) (len(names) / 11)
# start = (pos*i)
# end = (pos*(i+1))
# names = names[ start : end]
# len(names)


# In[ ]:


done_names = ['4cbde67c6d4feb90b93497fa08b413f7']
names = [x for x in names if x in done_names]
len(names)


# In[ ]:


# names = [x for x in names if x not in done_names]
# print(len(names))
# names = names[0:1856]


# In[ ]:


def read_img(ID, path, level=2):
    image_path = path + ID + '.tiff'
    img = skimage.io.MultiImage(image_path)
    img = img[level]
    return img

def read_mask_img(ID, path, level=2):
    image_path = path + ID + '_mask.tiff'
    img = skimage.io.MultiImage(image_path)
    img = img[level]
    return img

def img_to_gray(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def mask_img(img_gray, tol=210):
    mask = (img_gray < tol).astype(np.uint8)
    return mask


# In[ ]:


img_mask = read_mask_img(names[0], MASKS)
# img_mask = img_mask * 255

img1 = read_img(names[0], TRAIN)
img_gray1 = img_to_gray(img1)
mask1 = mask_img(img_gray1)


# In[ ]:


# fig, ax = plt.subplots(3, 2, figsize=(20, 25))

# for i, img in enumerate(names[:1]):
#     image = skimage.io.MultiImage(os.path.join(TRAIN,names[i]+'.tiff'))[TIFF_LEVEL]
#     mask = openslide.OpenSlide(os.path.join(MASKS,names[i]+'_mask.tiff'))
#     ax[i][0].imshow(image)
#     ax[i][0].set_title(f'{img} - Original - Label: {names[i]}')
    
#     mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
#     cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
        
#     ax[i][1].imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
#     ax[i][1].set_title(f'{img} - Mask - Label: {names[i]}')


# In[ ]:


def tile(img, mask = 0, idxs = None, index = 0):
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    if mask == 0:
        img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                    constant_values=255)
    else:
        img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                    constant_values=0)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        if mask == 0:
            img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
        else:
            img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=0)
    if index == 0:
        idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[::-1][:N]
    
    img = img[idxs]
    return img, idxs


# In[ ]:


x_tot,x2_tot = [],[]
# names = [name[:-10] for name in os.listdir(MASKS)]
with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out, zipfile.ZipFile(OUT_MASKS, 'w') as mask_out:
    for name in tqdm(names):
        masks, idxs = tile(skimage.io.MultiImage(os.path.join(MASKS,name+'_mask.tiff'))[TIFF_LEVEL], mask = 1)
#         for i in range(N):
#             mask = cv2.imencode('.png',masks[i][:,:,0])[1]
#             mask_out.writestr(f'{name}_{i}.png', mask)
#             mask = None
#             gc.collect()
        masks = None
        gc.collect()
        
        imgs, idxs = tile(skimage.io.MultiImage(os.path.join(TRAIN,name+'.tiff'))[TIFF_LEVEL], idxs = idxs, index = 1)
        for i in range(N):
            x_tot.append((imgs[i]/255.0).reshape(-1,3).mean(0))
            x2_tot.append(((imgs[i]/255.0)**2).reshape(-1,3).mean(0))
            #if read with PIL RGB turns into BGR
            img = cv2.imencode('.png',cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR))[1]
            img_out.writestr(f'{name}_{i}.png', img)
            img = None
            gc.collect()
            
        imgs = None
        idxs = None
        gc.collect()


# In[ ]:


#image stats
img_avr =  np.array(x_tot).mean(0)
img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
print('mean:',img_avr, ', std:', np.sqrt(img_std))

