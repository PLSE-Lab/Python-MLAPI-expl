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


data = pd.read_csv('../input/globalterrorismdb_0616dist.csv',encoding = "ISO-8859-1",low_memory=False)


# In[ ]:


data.columns


# In[ ]:


data.index


# In[ ]:


data.head()


# In[ ]:


data.head(100)


# In[ ]:


type (data)
data ['country_txt']


# In[ ]:


data


# In[ ]:


Germany


# In[ ]:


data.head()


# In[ ]:


Germany=data.loc[data['country_txt'] == 'Germany']


# In[ ]:


print (Germany)


# In[ ]:


Germany.head()


# In[ ]:


Germany.describe()


# In[ ]:


Germany.approxdate.value_counts()


# In[ ]:


Germany.approxdate.isnull().value_counts()


# In[ ]:


Germany.shape


# In[ ]:


Germany.columns


# In[ ]:


print(Germany.columns)


# In[ ]:


Germany.plot.plotter(x='iyear',y='imonth', color='r')


# In[ ]:




