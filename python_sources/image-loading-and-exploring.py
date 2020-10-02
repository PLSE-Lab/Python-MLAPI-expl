#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_df = pd.read_csv("../input/train_relationships.csv")
test_df = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


train_df.head()


# In[ ]:


len(train_df)


# In[ ]:


test_data = np.random.beta(a=1,b=1 , size= (100, 100, 3) )
plt.imshow(test_data)


# <h4>Opening images with PIL</h4>

# In[ ]:


import glob
path = '../input/train/' + train_df.p1[100]
image_datas = [Image.open(f) for f in glob.glob(path + "/*.jpg", recursive=True)]


# In[ ]:


image_datas


# #### Lets see first 4 images

# In[ ]:


f, ax = plt.subplots(1,4 ,  figsize=(50,20))
for i in range(4):
    ax[i].imshow(image_datas[i])


# In[ ]:


path = '../input/train/' + train_df.p2[100]
image_datas = [Image.open(f) for f in glob.glob(path + "/*.jpg", recursive=True)]
image_datas


# #### Its 4 images of family member

# In[ ]:


f, ax = plt.subplots(1,4 ,  figsize=(50,20))
for i in range(4):
    ax[i].imshow(image_datas[i])

