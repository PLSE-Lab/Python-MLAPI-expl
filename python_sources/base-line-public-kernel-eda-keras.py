#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm_notebook

import collections


get_ipython().run_line_magic('matplotlib', 'inline')

print(os.listdir("../input"))


# ## EDA

# In[ ]:


df_train = pd.read_csv('../input/train.csv')


# In[ ]:


df_train.head()


# In[ ]:


print("Train Sample Num = ",len(df_train))
print("Null Ratio in train = ",np.sum(df_train['EncodedPixels'].isnull())/len(df_train))


# In[ ]:


len(os.listdir("../input/train_images/"))


# In[ ]:


df_sub = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


df_sub.head()


# In[ ]:


print("Test Sample Num = ",len(df_sub))


# In[ ]:


len(os.listdir("../input/test_images/"))


# In[ ]:


def rle2mask(rle, imgshape):
    width = imgshape[0]
    height= imgshape[1]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )


# ### Let's visualize Defects
#  - Class Id 1:  Red
#  - Class Id 2:  Green
#  - Class Id 3:  Blue
#  - Class Id 4:  Yellow

# In[ ]:


num_show_img = 10

plt.figure(figsize=(40, 40))
for i in range(num_show_img):
    img_file = '../input/train_images/'+df_train['ImageId_ClassId'][i*4].split('_')[0]
    img = cv2.imread(img_file)
    mask = np.zeros_like(img)
    
    for j in range(4):
        if type(df_train['EncodedPixels'][i*4+j]) is not str:
            continue
            
        each_mask = rle2mask(df_train['EncodedPixels'][i*4+j],(np.size(img,0),np.size(img,1)))

        if j == 3:
            mask[:,:,0] = each_mask*255
            mask[:,:,1] = each_mask*255
        else:
            mask[:,:,j] = each_mask*255
        
    plt.subplot(num_show_img,1,i+1)
    plt.imshow(np.concatenate([img,mask],axis=0))
    plt.title(img_file)


# ### Let's remove train images without any defects 

# In[ ]:


df_train_removed = df_train[df_train['EncodedPixels'].notnull()].reset_index(drop=True)
print("Num of Defected Train Image = ",len(df_train_removed))
df_train_removed.head()


# In[ ]:


df_train_removed['ImageId'] = df_train_removed['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

counter = collections.Counter(list(df_train_removed['ImageId']))
print("uniq_ImageId = ",len(counter))
print(counter.most_common()[:5])


# In[ ]:


plt.hist(counter.values(),bins=3)


# ```
# All Train Image:        12,568
# ================================
# 
# 0 defect Train Image:    5,902
# 1 defect Train Image:    6,239
# 2 defect Train Image:      425
# 3 defect Train Image:        2
# ```

# In[ ]:





# ## Keras Segmentation Base Line

# ### To be continued...

# In[ ]:




