#!/usr/bin/env python
# coding: utf-8

# ## Load necessary packages

# In[ ]:


import multiprocessing as mp
import pandas as pd
import numpy as np
import cv2
import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")


# ## Build directory

# In[ ]:


directory = "../input/test_preprocessed/"
if not os.path.exists(directory):
    os.makedirs(directory)
    
df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
DATA_ROOT = '../input/aptos2019-blindness-detection/test_images/'
OUTPUT_DIR = directory
SIZE = 224


# ## Preprocess image

# In[ ]:


def crop_image_from_gray(img,tol=7):
    # Taken from https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img

def circle_crop(img):  
    # Taken from  https://www.kaggle.com/taindow/pre-processing-train-and-test-images
    """
    Create circular crop around image centre    
    """    
    
    img = crop_image_from_gray(img)    
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    
    return img 

def preprocess_image(df, run_root=DATA_ROOT, out_root=OUTPUT_DIR, size=SIZE):
    df = df.reset_index()
    for i in tqdm.tqdm(range(df.shape[0])):
        item = df.iloc[i]
        path = run_root+item.id_code+'.png'
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = circle_crop(img)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(OUTPUT_DIR + item.id_code + '.png',img) 


# ## Preprocess image parallelly

# In[ ]:


n_cpu = mp.cpu_count()
pool = mp.Pool(n_cpu)
n_cnt = df.shape[0] // n_cpu
dfs = [df.iloc[n_cnt*i:n_cnt*(i+1)] for i in range(n_cpu)]
dfs[-1] = df.iloc[n_cnt*(n_cpu-1):] 
res = pool.map(preprocess_image, [x_df for x_df in dfs])
pool.close()


# ## Compare raw and processed images

# In[ ]:


fig, axes = plt.subplots(1, 2)
img_raw = cv2.imread(DATA_ROOT + df.iloc[0].id_code + '.png')
img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

img = cv2.imread(OUTPUT_DIR + df.iloc[0].id_code + '.png', cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

axes[0].imshow(img_raw)
axes[1].imshow(img)
print(img_raw.shape, img.shape)


# ## Remove the directory at the end

# In[ ]:


if os.path.exists(directory):
    get_ipython().system("rm -r '../input/test_preprocessed'")


# In[ ]:


print(os.listdir("../input"))

