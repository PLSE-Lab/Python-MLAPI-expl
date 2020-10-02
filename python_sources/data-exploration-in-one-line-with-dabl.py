#!/usr/bin/env python
# coding: utf-8

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


get_ipython().system('pip install dabl')


# In[ ]:


data = pd.read_csv('/kaggle/input/icc-test-cricket-runs/ICC Test Batting Figures.csv', encoding='ISO-8859-1')


# In[ ]:


data.head()


# In[ ]:


data['Country'] = data.Player.str.extract("\(([\w/]+)\)")
data.head()


# In[ ]:


country_counts = data.Country.value_counts()
frequent_countries = country_counts.index[country_counts > 10]
data = data[data.Country.isin(frequent_countries)]


# In[ ]:


data.Country.value_counts()


# In[ ]:


import dabl
dabl.plot(data, target_col='Country')


# In[ ]:




