#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <hr> 
# 
# 1. For this competition, we will be mostly matching images based on their local features
# 
# 2. The visual recognition problem is central to computer vision research. From robotics to information retrieval, many desired applications demand the ability to identify and localize categories, places, and objects
# 
# 3. A local image feature is a tiny patch in the image that's invariant to image scaling, rotation and change in illumination. 
# 
# 4. A good local feature is like the piece you start with when solving a jigsaw puzzle, except on a much smaller scale. It's the eye of the cat or the corner of the table, not a piece on a blank wall.
# 
# **The extracted local features must be:**
# 
# 1. Repeatable and precise differentiating from similar image
# 2. Distinctive to the image, so images with different structure will not have them.
# 3. There could be hundreds or thousands of such features in an image. An image matcher algorithm could still work if some of the features are blocked by an object or badly deformed due to change in brightness or exposure. Many local feature algorithms are highly efficient and can be used in real-time applications.
# 4. Refer to given link for more details: https://pdfs.semanticscholar.org/5255/490925aa1e01ac0b9a55e93ec8c82efc07b7.pdf
# 
# **Algorithm supporting this problem statement are:**
# 
# 1. https://docs.opencv.org/3.3.1/dc/d0d/tutorial_py_features_harris.html
# 2. https://docs.opencv.org/3.3.1/da/df5/tutorial_py_sift_intro.html
# 3. https://docs.opencv.org/3.3.1/df/dd2/tutorial_py_surf_intro.html
# 4. https://docs.opencv.org/3.3.1/df/d0c/tutorial_py_fast.html
# and many more
# 
# 
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import glob
import os

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:



train_df = pd.read_csv('../input/landmark-retrieval-2020/train.csv')
train_df.shape,train_df.info()


# In[ ]:


train_df.head(10)


# **Validating Column distribution**

# In[ ]:


train_df.hist(column='landmark_id')


# Landmark Count

# In[ ]:


Values_Count=train_df['landmark_id'].value_counts()


# In[ ]:


Values_Count


# **Shape Of File**

# **Test and Index data**

# In[ ]:


test_list = glob.glob('../input/landmark-retrieval-2020/test/*/*/*/*')
index_list = glob.glob('../input/landmark-retrieval-2020/index/*/*/*/*')


# In[ ]:


print( 'Query', len(test_list), ' test images in ', len(index_list), 'index images')


# **Display**

# In[ ]:


plt.rcParams["axes.grid"] = False
f, axarr = plt.subplots(4, 3, figsize=(28, 24))

curr_row = 0
for i in range(12):
    example = cv2.imread(test_list[i])
    example = example[:,:,::-1]
    
    col = i%4
    axarr[col, curr_row].imshow(example)
    if col == 3:
        curr_row += 1
            
#     plt.imshow(example)
#     plt.show()


# In[ ]:




