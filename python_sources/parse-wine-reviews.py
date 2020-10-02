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


winereviews = pd.read_csv("/kaggle/input/wine-reviews/winemag-data-130k-v2.csv")
winereviews.shape


# In[ ]:


winereviews.columns
#winereviews=winereviews.dropna()
#pinots=winereviews.query('variety.str.contains("pinot")', engine='python')

#winereviews[["variety","price","points"]].head()


# In[ ]:


dataset_likePinot=winereviews.query('variety.str.contains("Pinot")', engine='python')


# In[ ]:


dataset_likePinot.plot.scatter(x='points',y='price')


# In[ ]:


#dataset_likePinot.describe(include='all')
dataset_likePinot.drop(['region_2','taster_twitter_handle','taster_name'],axis=1,inplace=True)


# In[ ]:


dataset_likePinot[-7:]

