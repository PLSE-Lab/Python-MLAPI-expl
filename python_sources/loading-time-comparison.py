#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
from time import time
from tqdm import tqdm
from skimage import io
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install /kaggle/input/fastparquet/python_snappy-0.5.4-cp36-cp36m-linux_x86_64.whl')
get_ipython().system('pip install /kaggle/input/fastparquet/thrift-0.13.0-cp36-cp36m-linux_x86_64.whl')
get_ipython().system('pip install /kaggle/input/fastparquet/fastparquet-0.3.2-cp36-cp36m-linux_x86_64.whl')


# In[ ]:


filepaths = glob('/kaggle/input/bengaliai/256_train/256/*.png')


# In[ ]:


len(filepaths)


# **Loading from .png**, thanks to Peter. https://www.kaggle.com/c/bengaliai-cv19/discussion/122467

# In[ ]:


for path in tqdm(filepaths):
    img = io.imread(path)


# **Loading from .parquet**

# In[ ]:


for i in tqdm(range(4)):
    df = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_{}.parquet'.format(i))


# **Loading from .parquet with fastparquet**, thanks to Vladislav https://www.kaggle.com/vladislavleketush/fast-parquet-loading-example

# In[ ]:


for i in tqdm(range(4)):
    df = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_{}.parquet'.format(i), engine='fastparquet')


# **Loading from .feather**, thanks to corochann https://www.kaggle.com/corochann/bangali-ai-super-fast-data-loading-with-feather

# In[ ]:


for i in tqdm(range(4)):
    df = pd.read_feather('/kaggle/input/bengaliaicv19feather/train_image_data_{}.feather'.format(i))

