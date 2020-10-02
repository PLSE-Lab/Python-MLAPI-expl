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


df = pd.read_excel('/kaggle/input/covid19-tweets/coronavirusinindia.xlsx')


# In[ ]:


#df.head(n=20)
df.style.background_gradient(subset=['id'], cmap='Blues', axis=None)        .background_gradient(subset=['conversation_id'], cmap='Reds', axis=None)        .background_gradient(subset=['created_at'], cmap='Greens', axis=None)        .background_gradient(subset=['user_id'], cmap='Greys', axis=None)        .background_gradient(subset=['replies_count'], cmap='PuRd', axis=None)        .background_gradient(subset=['retweets_count'], cmap='Set3', axis=None)        .background_gradient(subset=['likes_count'], cmap='summer', axis=None) 


# In[ ]:


df.dtypes

