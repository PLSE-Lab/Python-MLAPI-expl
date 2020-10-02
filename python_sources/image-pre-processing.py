#!/usr/bin/env python
# coding: utf-8

# 3/10 - First download of project images. Decided to familiarize myself with some of the images to see if any visual patterns emerge. After seeing about 25 or so, it's clear that there are some very significant differences in the types of images being presented. So far, i'm seeing at least 3 classes of photos;  My initial thought is that I will need to review all the images to see just how many classes I can identify, then design pre-processing techniques for each of the classes. I'm thinking I can use openCV to convert to greyscale, and then test foreground/background  thresholds to get an appropriate count and or using blobdetection (will come back to this). 

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




