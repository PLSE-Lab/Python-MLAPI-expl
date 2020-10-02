#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv")


# In[ ]:


train.shape


# In[ ]:


train.head(5)


# First, we've got a lot of work to do to deal with the columns

# In[ ]:


import h2o


# In[ ]:


mssc = train['MSSubClass']


# In[ ]:


type(mssc)


# In[ ]:


trainclean = pd.DataFrame({'mssc': mssc})


# In[ ]:


trainclean.head()


# In[ ]:


mssc.value_counts()


# In[ ]:


train.ix[train['MSSubClass'] == 20, ['Id', 'LotArea', 'Street']]


# In[ ]:


train_h2o = h2o.import_frame("../input/train.csv")


# In[ ]:


train.info(verbose=True)


# In[ ]:


train['Alley'].unique()


# In[ ]:


train['Alley'].value_counts()


# In[ ]:


train['Alley'].isnull().sum()


# In[ ]:


train['Alley'] = train['Alley'].fillna('No Alley')


# In[ ]:


train['Alley'].value_counts()


# In[ ]:


train.groupby('Alley')['SalePrice'].mean()


# In[ ]:


df = pd.DataFrame(np.random.rand(10, 5), columns=['Alley'])


# In[ ]:


sns.boxplot( x = 'Alley', y='SalePrice', data=train, showfliers=False)


# In[ ]:


train['PoolQC'].value_counts()


# In[ ]:


train['Pool'].value_counts()


# In[ ]:




