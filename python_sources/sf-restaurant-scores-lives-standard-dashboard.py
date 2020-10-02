#!/usr/bin/env python
# coding: utf-8

# Hello! Welcome to my Kernal for the exercise Dashboarding with Notebooks.
# 
# The dataset I have chosen is the SF Restaurant Scores - LIVES Standard dataset. I chose this dataset because I am a food technologist and this dataset is close to my domaine.
# 
# The goals is to represent the inspection score of each restauarent
# 
# Let's begin by loading in the lastest updated dataset. Thank you for reading!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


## Load in dataset, and see information ##
rs = pd.read_csv("../input/restaurant-scores-lives-standard.csv")
print(rs.info())


# In[ ]:


rs.head()


# 
# What are the score compare by city?
# 

# In[ ]:


rsn = rs.groupby("risk_category").size()
lth = range(len(rsn))
plt.bar(lth, rsn)
plt.xticks(lth, rsn.index, rotation=90)
plt.show()


# In[ ]:




