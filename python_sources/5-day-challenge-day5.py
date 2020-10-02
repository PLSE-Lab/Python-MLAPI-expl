#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chisquare, chi2_contingency, chi2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.


# In[2]:


cereal = pd.read_csv('../input/cereal.csv')
cereal.head(10)


# In[3]:


cereal.describe(include = ['O'])


# In[4]:


cereal['shelf'] = cereal['shelf'].apply(str)
cereal.describe(include = ['O'])


# In[12]:


mfr=cereal["mfr"]
type=cereal["type"]

#Calculate chi-square
print(chisquare(mfr.value_counts()))


# In[11]:


print(chisquare(type.value_counts()))

