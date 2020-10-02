#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The sinking Titanic is one of the most notorious shipwredcks in the history. 1502 passenger died including passengers and crew. 
# 
# <font color = "Red"> 
# Content: 
# 
# [1. Load and Check Data ](#1)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:





# <a id = "1" ><a/>
# ## Load and Check Data

# In[ ]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId = test_df["PassengerId"] #I did that because we will use this variable later and I wanted to save it and by this way it will not corrupted.


# In[ ]:


train_df.columns


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe().T


# In[ ]:





# In[ ]:





# In[ ]:




