#!/usr/bin/env python
# coding: utf-8

# The idea of this Notebook is to easily upload the pretrained model darknet53.conv.74 as download can take several minutes and I could not find a public notebook with the file. Add the notebook in your input. Size is 155Mb. You can use it to train your Yolo model with your custom object.Includes also the whole darknet.

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


# In[ ]:


get_ipython().system('git clone https://github.com/AlexeyAB/darknet')


# In[ ]:


get_ipython().system('mkdir darknet_model')


# In[ ]:


get_ipython().system('cd ./darknet_model')
get_ipython().system('wget https://pjreddie.com/media/files/darknet53.conv.74 -O ./darknet/darknet53.conv.74')

