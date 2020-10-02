#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from os import listdir
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
print(os.listdir("../input"))


# In[ ]:


train_df=pd.read_csv("../input/train.csv")


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


train_df_nan=train_df[train_df['labels'].isna()]


# In[ ]:


train_df_nan.count()


# In[ ]:


img=Image.open("../input/train_images/100241706_00004_2.jpg")
plt.imshow(img)


# **Looking at the unicode translation file**

# In[ ]:


translation_df=pd.read_csv("../input/unicode_translation.csv")


# In[ ]:


translation_df.head(10)


# Still work in progress. Any comments are appreciated
