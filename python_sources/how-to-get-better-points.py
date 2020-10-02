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


#  I just copy other peoples code. and execute. 
#  get the Public Leaderboard points and Private Leaderboard points. 
#  Now I just see how to submit. 
#  but I dont know how to get better Public Leaderboard points. 
#  I want get more points so i study many things, but that's not helpful
#  do you know how to get better Public Leaderboard points? 1points 
#  if u know that teach me plz. 

# In[ ]:


test = pd.read_csv('../input/google-quest-challenge/test.csv')
train = pd.read_csv('../input/google-quest-challenge/train.csv')


# In[ ]:


train.info()


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


train.shape


# In[ ]:


test.shape


# to be continue
