#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/data.csv")
df.head()


# In[ ]:


df.drop('id',axis = 1,inplace = True)
df.drop('Unnamed: 32',axis = 1,inplace = True)


# In[ ]:


target = df['diagnosis']
data = df.drop('diagnosis',axis = 1)


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


sbx = data.boxplot()
for item in sbx.xaxis.get_ticklabels():
	item.set_rotation(45)


# In[ ]:


mean_cols = [x for x in data.columns if 'mean' in x]
print(mean_cols)
col_types = [x.rstrip('mean') for x in mean_cols]
print(col_types)
se_cols = [x + 'se' for x in col_types]
worst_cols = [x +'worst' for x in col_types]


# In[ ]:


clist = mean_cols + se_cols + worst_cols
print([x for x in data.columns if x not in clist])
print([x for x in clist if x not in data.columns])
print(len(mean_cols),len(se_cols),len(worst_cols))


# In[ ]:


data.columns


# In[ ]:


sbx = data[mean_cols].boxplot()
for item in sbx.xaxis.get_ticklabels():
	item.set_rotation(45)
sbx.get_ylim()


# In[ ]:




