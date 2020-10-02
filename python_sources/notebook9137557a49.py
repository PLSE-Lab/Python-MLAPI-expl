#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tflearn
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/train.csv')


# In[ ]:


data


# In[ ]:


data_tf,labels = tflearn.data_utils.load_csv('../input/train.csv',has_header=True,target_column=1,n_classes=2)
data_tf


# In[ ]:


labels


# In[ ]:


for i in range(len(data_tf)):
    data_tf[i][3] = 1. if data_tf[i][3]=='female' else 0.


# In[ ]:


data_tf


# In[ ]:


data =data_tf.

