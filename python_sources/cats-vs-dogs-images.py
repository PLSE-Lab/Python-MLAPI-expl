#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, cv2, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


def read_image2(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    b,g,r = cv2.split(img)
    img2 = cv2.merge([r,g,b])
    return cv2.resize(img2, (256, 256), interpolation=cv2.INTER_CUBIC)


# In[ ]:


train_dir = '../input/train/'
test_dir = '../input/test/'
train_dogs =   [train_dir+i for i in os.listdir(train_dir) if 'dog' in i]


# In[ ]:


train_cats =   [train_dir+i for i in os.listdir(train_dir) if 'cat' in i]


# In[ ]:


train_dogs[0:5]


# In[ ]:


train_cats[0:5]


# In[ ]:


def show_cats_and_dogs(idx):
    cat = read_image2(train_cats[idx])
    dog = read_image2(train_dogs[idx])
    pair = np.concatenate((cat, dog), axis=1)
    plt.figure(figsize=(10,5))
    plt.imshow(pair)
    plt.show()
    
for idx in range(5,10):
    show_cats_and_dogs(idx)


# In[ ]:




