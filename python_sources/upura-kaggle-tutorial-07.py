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


data1 = pd.read_csv('../input/titanictutorialensemble20190524/submission-03.csv')
data2 = pd.read_csv('../input/titanictutorialensemble20190524/submission-06.csv')
data3 = pd.read_csv('../input/titanictutorialensemble20190524/200-lines-randomized-search-lgbm-82-3.csv')


# In[ ]:


data1.head()


# In[ ]:


data2.head()


# In[ ]:


data3.head()


# In[ ]:


ensembled = pd.DataFrame()
ensembled['PassengerId'] = data1['PassengerId']
ensembled['Survived'] = (data1['Survived'] + data2['Survived'] + data3['Survived']) // 2
ensembled.head()


# In[ ]:


ensembled.to_csv("submission.csv", index = False)
get_ipython().system('ls .')


# In[ ]:


get_ipython().system('head submission.csv')

