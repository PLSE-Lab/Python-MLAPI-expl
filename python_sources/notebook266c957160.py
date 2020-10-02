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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/Dataset_spine.csv')
print (data)


# In[ ]:


data=data.rename(columns={'Col1':'pelvic incidence', 'Col2':'pelvic tilt', 'Col3': 'lumbar lordosis angle', 'Col4':'sacral slope', 'Col5': 'pelvic radius', 'Col6':'degree spondylolisthesis', 'Col7': 'pelvic slope',  'Col8': 'Direct tilt' , 'Col9':'thoracic slope' , 'Col10': 'cervical tilt' , 'Col11':'sacrum angle' , 'Col12':'scoliosis slope'  })


# In[ ]:


data


# In[ ]:


data = data.as_matrix()


# In[ ]:


data


# In[ ]:


data=data[:,:-1]


# In[ ]:


data


# In[ ]:




