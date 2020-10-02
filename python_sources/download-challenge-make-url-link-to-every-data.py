#!/usr/bin/env python
# coding: utf-8

# # It might be possible to download all the data by running this method in a split to fit into the notebook's storage

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('cp -r /kaggle/input/prostate-cancer-grade-assessment/train_label_masks/060f60eeecf1ab502526a34db9caaf8e_mask.tiff .')


# In[ ]:


from IPython.display import FileLinks
FileLinks(".")

