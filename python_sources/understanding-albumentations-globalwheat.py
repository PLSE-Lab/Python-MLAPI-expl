#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect
from albumentations.pytorch.transforms import ToTensorV2
get_ipython().run_line_magic('matplotlib', 'inline')

import os
import random
from PIL import Image

#To get reproducible transformations
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


# In[ ]:


TRAIN_IMAGES_PATH = '../input/global-wheat-detection/train'
df = pd.read_csv('../input/global-wheat-detection/train.csv')


# ### Display Original Images

# In[ ]:


def get_bbox(img):
    df_img = df[df.image_id==img][['bbox']]
    
    bbox_list = []
    for i in df_img.iterrows():
        xmin,ymin,width,height = np.fromstring(i[1][0][1:-1],sep=',')
        bbox_list.append([xmin,ymin,width,height])
    return bbox_list
    
    
def display_images(rows, cols, image_paths):
    fig, ax = plt.subplots(rows,cols, figsize=(15,5))
    plt.suptitle('Original Images')
    for j in range(cols):
        arr = Image.open(image_paths[j])
        img_id = image_paths[j][-13:-4]
        ax[j].set_title(img_id)
        ax[j].imshow(arr)
        ax[j].axis('off')
        
        bboxes = get_bbox(img_id)
        
        for bbox in bboxes:            
            ax[j].add_patch(rect((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none'))
        

img_list =[TRAIN_IMAGES_PATH + '/'+ s + '.jpg' for s in pd.unique(df.image_id)[0:3].tolist()]

display_images(1,3,img_list)


# # Transformations
# 
# This is my attempt to understand transformations/albumentations. I took some of the transformations mentioned in [this](https://www.kaggle.com/shonenkov/training-efficientdet/notebook) kernel.
# 
# [Albumentations - ToGray](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ToGray)
# 

# In[ ]:



display_images(1,3,img_list)

fig, ax = plt.subplots(1,3, figsize=(15,5))
plt.suptitle("Transformed - ToGray()")
seed_everything(SEED)
transforms_1 = A.Compose([A.ToGray(p=0.4)]) #higher probability greater chance of getting gray scale

for i,img_path in enumerate(img_list):    
    arr = np.array(Image.open(img_path))
    img_id = img_path[-13:-4]
    bboxes = get_bbox(img_id)
    
    arr = transforms_1(**{"image": arr})['image']
    ax[i].set_title(img_id)
    ax[i].imshow(arr)
    for bbox in bboxes:  
            ax[i].add_patch(rect((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none'))
    ax[i].axis('off')


# Only two images got gray scale transformation. If we set probability to higher value we can get more grayscale transformation (and vice versa)

# ## HorizontalFlip
# 
# [Albumentations HorizontalFlip](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.HorizontalFlip)

# In[ ]:


display_images(1,3,img_list)

fig, ax = plt.subplots(1,3, figsize=(15,5))
plt.suptitle("Transformed - HorizontalFlip")
seed_everything(SEED)
#We need bbox_params so that we get correct bboxes after flipping. 
transforms_1 = A.Compose([A.HorizontalFlip(p=0.5)],
                         bbox_params=A.BboxParams(format='coco', min_area=0, 
                                               min_visibility=0, label_fields=['labels']) ) 

for i,img_path in enumerate(img_list):    
    arr = np.array(Image.open(img_path))
    img_id = img_path[-13:-4]
    bboxes = get_bbox(img_id)
    #Here labels are all ones because wheat is the only class we have. 
    #For e.g.: Labels would be different if we have bboxes for different objects for e.g cat, dog in the same image. 
    transform = transforms_1(**{"image": arr , "bboxes": bboxes, "labels":np.ones(len(bboxes))})
    arr = transform['image']
    bboxes = transform['bboxes']
    ax[i].set_title(img_id)
    ax[i].imshow(arr)
    for bbox in bboxes:  
            ax[i].add_patch(rect((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none'))
            
    ax[i].axis('off')
    


# Third image ('7b72ea0fb') isn't flipped. First two images did get flipped.

# ### Resize 
# 
# [Albumentations Resize](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Resize)

# In[ ]:


display_images(1,3,img_list)

fig, ax = plt.subplots(1,3, figsize=(15,5))
plt.suptitle("Transformed - Resize")
seed_everything(SEED)
#We need bbox_params so that we get correct bboxes after flipping. 
transforms_1 = A.Compose([A.Resize(height=512, width=512, p=1),], #p=1 because we want all images resized
                         bbox_params=A.BboxParams(format='coco', min_area=0, 
                                               min_visibility=0, label_fields=['labels']) ) 

for i,img_path in enumerate(img_list):    
    arr = np.array(Image.open(img_path))
    img_id = img_path[-13:-4]
    bboxes = get_bbox(img_id)
    #Here labels are all ones because wheat is the only class we have. 
    #For e.g.: Labels would be different if we have bboxes for different objects for e.g cat, dog in the same image. 
    transform = transforms_1(**{"image": arr , "bboxes": bboxes, "labels":np.ones(len(bboxes))})
    arr = transform['image']
    bboxes = transform['bboxes']
    ax[i].set_title(img_id + ' ' + str(arr.shape))  #original images were 1024,1024,3
    ax[i].imshow(arr)
    for bbox in bboxes:  
            ax[i].add_patch(rect((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none'))
            
    ax[i].axis('off')
    


# ### Cutout
# 
# [Albumentations Cutout](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Cutout)
# 
# - No bboxes required here

# In[ ]:


display_images(1,3,img_list)

fig, ax = plt.subplots(1,3, figsize=(15,5))
plt.suptitle("Transformed - Cutout")
seed_everything(SEED)
transforms_1 = A.Compose( [A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=[255,255,255], p=0.5)] )
#fill value = [255,255,255] = White
#If you change p=0.7 all three images should have cutout.

for i,img_path in enumerate(img_list):    
    arr = np.array(Image.open(img_path))
    img_id = img_path[-13:-4]
    transform = transforms_1(**{"image": arr })
    arr = transform['image']
    ax[i].set_title(img_id + ' ' + str(arr.shape))  #original images were 1024,1024,3
    ax[i].imshow(arr)
    ax[i].axis('off')


# ### ToTensorV2 
# 
# [Albumentations ToTensorV2](https://albumentations.readthedocs.io/en/latest/api/pytorch.html#albumentations.pytorch.transforms.ToTensorV2)

# In[ ]:


transforms_1 = A.Compose( [A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=[255,255,255], p=0.5)] )

for i,img_path in enumerate(img_list[0:1]):    
    arr = np.array(Image.open(img_path))
    img_id = img_path[-13:-4]
    transform = transforms_1(**{"image": arr })
    arr = transform['image']
    print(type(arr))

    
#Converting to torch.tensor - which will be required for modeling
transforms_1 = A.Compose( [A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=[255,255,255], p=0.5),
                           ToTensorV2() ] )

for i,img_path in enumerate(img_list[0:1]):    
    arr = np.array(Image.open(img_path))
    img_id = img_path[-13:-4]
    transform = transforms_1(**{"image": arr })
    arr = transform['image']
    print(type(arr))


# ### ToGray + Resize + HorizontalFlip + Cutout

# In[ ]:


display_images(1,3,img_list)

fig, ax = plt.subplots(1,3, figsize=(15,5))
plt.suptitle("Transformed - ToGray + Resize + HorizontalFlip + Cutout")
seed_everything(SEED)
transforms_1 = A.Compose( [ A.ToGray(p=0.4),
                            A.Resize(height=512, width=512, p=1),
                            A.HorizontalFlip(p=0.5), 
                            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=[255,255,255], p=0.5)],
                          bbox_params=A.BboxParams(format='coco', min_area=0, 
                                               min_visibility=0, label_fields=['labels'])  )
#fill value = [255,255,255] = White
#If you change p=0.7 all three images should have cutout.

for i,img_path in enumerate(img_list):    
    arr = np.array(Image.open(img_path))
    img_id = img_path[-13:-4]
    bboxes = get_bbox(img_id)
    transform = transforms_1(**{"image": arr , "bboxes": bboxes, "labels":np.ones(len(bboxes))})
    arr = transform['image']
    bboxes = transform['bboxes']
    ax[i].set_title(img_id + ' ' + str(arr.shape))  #original images were 1024,1024,3
    ax[i].imshow(arr)
    for bbox in bboxes:  
            ax[i].add_patch(rect((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none'))
            
    ax[i].axis('off')
    


# Note that not all transformations got applied to all images due to setting different probabilities. Thankfully the seeds work, so results are reproducible. 

# In[ ]:




