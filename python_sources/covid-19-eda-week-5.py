#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_profiling import ProfileReport

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train[train['Country_Region'] == 'US'].head()


# In[ ]:


test[test['Country_Region'] == 'US'].head()


# In[ ]:


train[train['Country_Region'] == 'US'].County.unique()


# In[ ]:


train[train['Country_Region'] == 'US'].County.unique().shape


# In[ ]:


test[test['Country_Region'] == 'US'].County.unique().shape


# In[ ]:


train_profile = ProfileReport(train, title='Pandas Profiling Report', html={'style':{'full_width':True}})
train_profile


# In[ ]:


test_profile = ProfileReport(test, title='Pandas Profiling Report', html={'style':{'full_width':True}})
test_profile


# In[ ]:




