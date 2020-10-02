#!/usr/bin/env python
# coding: utf-8

# ## import modules and define models

# In[ ]:


import numpy as np # linear algebra
import pandas as pd
pd.set_option("display.max_rows", 101)
import os
print(os.listdir("../input"))
import cv2
import json
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["font.size"] = 15
import seaborn as sns
from collections import Counter
from PIL import Image
import math
import seaborn as sns
from collections import defaultdict
from pathlib import Path
import cv2
from tqdm import tqdm
from plotly import express as px


# In[ ]:


input_dir = "../input/shopee"


# ## read all text data
# #### file description
# * train/train - folder of training images
# * test/test - folder of test images (classifying these images)
# * train.csv - training annotations which provide classes for products (ClassId = [1,..., 41])
# 

# In[ ]:


train_df = pd.read_csv("../input/shopee/train.csv")
test_df = pd.read_csv("../input/shopee/test.csv")


# In[ ]:


train_df.head()


# ### First, check the number of each class.

# In[ ]:


train_class_count = train_df.category.value_counts()
px.bar(train_class_count,color=train_class_count.index)


# In[ ]:


train_df["path"]=train_df['category'].map(lambda x: "../input/shopee/train/train/"+str(x).zfill(2)+"/")+                                                                train_df["filename"]
train_df["path"][0]


# In[ ]:


# preview the images first
for cls in range(42):
    print(str(cls))
    temp = train_df[train_df["category"]==cls]
    plt.figure(figsize=(100,50))
    x, y = 5, 8
    plt.title(cls)
    for i in range(40): 
        image = Image.open(temp["path"].values[i])
        plt.subplot(y, x, i+1)
        plt.imshow(image,interpolation='nearest')
    
    plt.show()

