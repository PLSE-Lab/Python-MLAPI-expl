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


train_df=pd.read_csv("../input/train.csv")


# In[ ]:


train_df.info()


# In[ ]:


train_df.head()


# In[ ]:


#Total Nans
train_df['EncodedPixels'].isnull().sum()


# In[ ]:


import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tqdm
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


trainfiles=os.listdir("../input/train_images/")


# In[ ]:


len(trainfiles)


# In[ ]:


image_size=[]
for image_id in trainfiles:
    img=Image.open("../input/train_images/"+image_id)
    width,height=img.size
    image_size.append((width,height))
   


# In[ ]:


image_size_df=pd.DataFrame(image_size,columns=["width","height"])
image_size_df.head()


# Image width distribution

# In[ ]:


image_size_df.width.hist()


# Image height distribution

# In[ ]:


image_size_df.height.hist()


# In[ ]:


testfiles=os.listdir("../input/test_images/")


# In[ ]:


len(testfiles)


# In[ ]:


image_size=[]
for image_id in testfiles:
    img=Image.open("../input/test_images/"+image_id)
    width,height=img.size
    image_size.append((width,height))


# In[ ]:


test_image_size_df=pd.DataFrame(image_size,columns=["width","height"])
test_image_size_df.head()


# In[ ]:


__import__('pandas').read_csv('../input/sample_submission.csv',converters={'EncodedPixels':lambda e:''}).to_csv('submission.csv',index=False)


# In[ ]:




