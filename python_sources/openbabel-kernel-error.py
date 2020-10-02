#!/usr/bin/env python
# coding: utf-8

# It seems that after Kaggle released its new UI for kernels there have been a lot of bugs. For example, in this kernel, a core file gets outputted instead of the structures dataframe. However, if the line import openbabel as ob is commented out, the kernel runs fine. But, I have many kernels that import openbabel and they have worked fine in the past. I would like to know if other people are experiencing this same issue.

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


structures = pd.read_csv("../input/structures.csv")


# In[ ]:


get_ipython().system('conda install -y -c openbabel openbabel ')
import openbabel as d


# In[ ]:


structures.to_csv('structures.csv', index=False)

