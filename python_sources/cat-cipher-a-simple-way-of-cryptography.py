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


# # This is my cryptography playground :))

# In[ ]:


word = 'catcodes'


# In[ ]:


lst = [char for char in word]


# In[ ]:


df = pd.DataFrame(lst, columns = ['word'])


# In[ ]:


df


# In[ ]:


df['word'] = df['word'].astype('category').cat.codes


# In[ ]:


df['word'].values


# # Well, given the result, could you solve it? 
# 
# remember, those sentence doesnt have complete alphabet.
