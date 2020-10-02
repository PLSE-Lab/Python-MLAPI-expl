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


get_ipython().system('pip install awscli')


# In[ ]:


get_ipython().system('aws s3 --no-sign-request sync s3://open-images-dataset/train "./images/train"')


# In[ ]:


get_ipython().system('aws s3 --no-sign-request sync s3://open-images-dataset/validation "./images/validation"')


# In[ ]:


get_ipython().system('aws s3 --no-sign-request sync s3://open-images-dataset/test "./images/test"')

