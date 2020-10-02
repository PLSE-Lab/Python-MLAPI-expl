#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sys
import pandas as pd
from skimage.io import imread, imsave
from skimage.color import gray2rgb
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from functools import reduce
import os


# In[ ]:


# Upload submission
sub = pd.read_csv("../input/ashrae-energy-prediction/sample_submission.csv")


# In[ ]:


#Dataset loaded for blending 
#https://www.kaggle.com/yamsam/ashrae-highway-kernel-route2
#https://www.kaggle.com/purist1024/ashrae-simple-data-cleanup-lb-1-08-no-leaks
#https://www.kaggle.com/wentixiaogege/ashrae-maybe-this-can-make-public-lb-some-useful
#https://www.kaggle.com/huanglinyan/ashrae-may-make-it-up-to-1-0
#https://www.kaggle.com/khoongweihao/ashrae-leak-validation-bruteforce-heuristic-search/output
#https://www.kaggle.com/yamsam/ashrae-leak-validation-and-more


# In[ ]:


df0 = pd.read_csv("../input/ashrae-public/ashrae-highway-kernel-route2.csv") #1.03
df1 = pd.read_csv("../input/ashrae-public/ashrae-leak-validation-and-more.csv") #0.99
df2 = pd.read_csv("../input/ashrae-public/ashrae-leak-validation-bruteforce-heuristic-search.csv") #0.98
df3 = pd.read_csv("../input/ashrae-public/ashrae-may-make-it-up-to-1-0.csv") #1
df4 = pd.read_csv("../input/ashrae-public/ashrae-maybe-this-can-make-public-lb-some-useful.csv") #1.1
df5 = pd.read_csv("../input/ashrae-public/ashrae-simple-data-cleanup-lb-1-08-no-leaks.csv") #1.08


# In[ ]:


blend1 = df0['meter_reading']*0.5 + df4['meter_reading']*0.3 + df5['meter_reading']*0.2


# In[ ]:


sub['meter_reading'] = blend1
df11 = sub


# In[ ]:


blend2 = df11['meter_reading']*0.1 + df1['meter_reading']*0.15 + df2['meter_reading']*0.6 + df3['meter_reading']*0.15


# In[ ]:


sub1 = pd.read_csv("../input/ashrae-energy-prediction/sample_submission.csv")


# In[ ]:


sub1['meter_reading'] = blend2
sub1.head()


# In[ ]:


sub1.to_csv(f'submission.csv', index=False, float_format='%g')

