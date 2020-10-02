#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from matplotlib import cm as CM
import PIL.Image as Image
import os
import h5py


# ### List file 

# In[ ]:


print(os.listdir("../input/shanghaitech_with_people_density_map/ShanghaiTech/part_A/train_data/"))


# In[ ]:


print(os.listdir("../input/shanghaitech_with_people_density_map/ShanghaiTech/part_A/train_data/images"))


# In[ ]:


print(os.listdir("../input/shanghaitech_with_people_density_map/ShanghaiTech/part_A/train_data/ground-truth-h5"))


# ### View image and its density map

# In[ ]:


image_path = "../input/shanghaitech_with_people_density_map/ShanghaiTech/part_A/train_data/images/IMG_261.jpg"
density_map_path = "../input/shanghaitech_with_people_density_map/ShanghaiTech/part_A/train_data/ground-truth-h5/IMG_261.h5"


# In[ ]:


from matplotlib import pyplot as plt

#now see a sample from ShanghaiA
plt.imshow(Image.open(image_path))


# In[ ]:



gt_file = h5py.File(density_map_path,'r')
groundtruth = np.asarray(gt_file['density'])

plt.imshow(groundtruth,cmap=CM.jet)


# In[ ]:




