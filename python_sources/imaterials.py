#!/usr/bin/env python
# coding: utf-8

# ## Imports 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import cv2

import torch
import torch.utils.data as data
import torchvision
import torchvision.utils as utils
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input/train/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_csv_path = "../input/train.csv"
train_path = "../input/train"
test_path = "../input/test"
train_names = os.listdir(train_path)
test_names = os.listdir(test_path)

SIZE = (512, 512)


# In[ ]:


# trdf = pd.read_csv(train_csv_path)


# In[ ]:


# trdf.ClassId.value_counts()[trdf.ClassId.value_counts() > 20]


# In[ ]:


trdf.head()


# In[ ]:


pixels_example = trdf.EncodedPixels.values[0]


# In[ ]:


fig, ax = plt.subplots(1, 2)
fig.set_size_inches((12, 4))
ax[0].hist(trdf.Height);
ax[1].hist(trdf.Width);


# In[ ]:


img = plt.imread(train_path + train_names[0])
plt.imshow(img);


# In[ ]:





# ## Data loader

# In[ ]:





# In[ ]:





# In[ ]:





# ## Utils

# In[ ]:


def img_generator(img_dir):
    img_names = os.listdir(img_dir)
    for name in img_names:
        img_p = os.path.join(img_dir, name)
        yield plt.imread(img_p), name
        
def resize_imgs(imgs_dir, target_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    imgs_gen = img_generator(imgs_dir)
    for i, (img, name) in enumerate(imgs_gen):
        if i % 100 == 0:
            print(i)
        resized = cv2.resize(img, target_size, cv2.INTER_CUBIC)
        plt.imsave(os.path.join(out_dir, name), resized)


# In[ ]:


# resize test
# resize_imgs(test_path, SIZE, "test_512")
resize_imgs(train_path_path, SIZE, "train_512")


# In[ ]:


len(test_names)


# In[ ]:


os.makedirs("test_resize")


# In[ ]:


os.listdir(".")


# In[ ]:


rm "test_resize/" -r


# In[14]:


im_p = test_path +"/" + test_names[233]
img = plt.imread(im_p)
plt.imsave("test.jpg", img)
# plt.imshow(img)


# In[16]:


df = pd.DataFrame([1, 1, 1])
df.to_csv("test.model")


# In[17]:


ls


# In[ ]:




