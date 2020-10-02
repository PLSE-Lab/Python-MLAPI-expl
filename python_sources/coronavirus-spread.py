#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')


# In[ ]:


data.head(60)


# In[ ]:


data['time'] = pd.to_datetime(data['Last Update'])


# In[ ]:


import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# In[ ]:


ax = sns.lineplot(x="time", y="Confirmed", data=data)


# In[ ]:


ax = sns.lineplot(x="time", y="Deaths", data=data)


# In[ ]:


ax = sns.distplot(data['Confirmed'])


# In[ ]:


ax = sns.distplot(data['Deaths'])


# In[ ]:


ax = sns.distplot(data['Confirmed'].cumsum())


# In[ ]:


ax = sns.distplot(data['Deaths'].cumsum())


# In[ ]:


data['Confirmed'].cumsum().plot()


# In[ ]:


data['Deaths'].cumsum().plot()


# In[ ]:




