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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


ipldf = pd.read_csv('../input/matches.csv')
print(ipldf.head(5))


# In[ ]:


print(ipldf.describe())


# In[ ]:


print(ipldf.columns)
print(ipldf.values)


# In[ ]:


print(ipldf.groupby('season').size())
print(ipldf.groupby('season').size().plot(kind='bar'))


# **Number of matches in season wise: ohh ..! 76 matches in 2013 **

# In[ ]:


idf = pd.DataFrame(ipldf.groupby('toss_winner').size())
# print(idf.columns)
print(idf.plot(kind='bar'))


# **Highest toss winner : MI **

# **Will check with more wins............!**

# In[ ]:


idf2 = ipldf.groupby('winner').size().plot(kind='bar')
print(idf2)


# **Now it's time to check with MOM.**

# In[ ]:


idf = ipldf.groupby('player_of_match').size().to_frame('size')
print(idf.sort_values(by='size', ascending=False).head(10).plot(kind='bar', figsize=(15,5)))


# ohhh it's for Chris Gayle

# In[ ]:




