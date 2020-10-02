#!/usr/bin/env python
# coding: utf-8

# This kernel is based on pre-trained TF model from the Tensorflow Hub.
# 
# The kernel was inspired by [Vikram's Kernel](https://www.kaggle.com/vikramtiwari/baseline-predictions-using-inception-resnet-v2) and [xhlulu Kernel](https://www.kaggle.com/xhlulu/intro-to-tf-hub-for-object-detection).
# 
# # How to get predictions ?
# 
# I have used Inception-ResNet. This means that the inference will be slower, but the accuracy is better as compared to MobileNet v2.
# 
# If you are using Kaggle Kernels split the image id's into bunch of 25000 and run the kernels 4 times if you get error code 137 [My kernel](https://www.kaggle.com/akashdeepjassal/tf-hub-inception-resnet?scriptVersionId=17639169)
# 

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


base='../input/'
for dir in (os.listdir(base)):
    if (dir == 'open-images-2019-object-detection'):
        1
    else:
        print(dir,os.listdir(base+dir))


# In[ ]:


base_dir='../input/'
df1=pd.read_csv(base_dir+'tf-hub-end/'+'submission_75k.csv')
df1.head()


# In[ ]:


base_dir='../input/'
df2=pd.read_csv(base_dir+'tf-hub-75k/'+'submission_75k.csv')
df2.head()


# In[ ]:


base_dir='../input/'
df3=pd.read_csv(base_dir+'tf-hub-25-to-50-k/'+'submission_50k.csv')
df3.head()


# In[ ]:


base_dir='../input/'
df4=pd.read_csv(base_dir+'tf-hub/'+'submission_25k.csv')
df4.head()


# In[ ]:


df_final=pd.concat([df1, df2, df3, df4])
df_final.head()


# In[ ]:


df1.shape, df2.shape, df3.shape, df4.shape


# In[ ]:


df_final.shape


# In[ ]:


df_final.head()


# In[ ]:


df_final.to_csv('submission.csv',index=None)


# In[ ]:




