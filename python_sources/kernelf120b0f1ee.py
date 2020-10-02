#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Graph processing
import seaborn as sns; sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/DATA.csv")
df.info()


# In[ ]:


df.describe()


# There is no missing values in the dataset. 

# **Data Distribution**

# In[ ]:


sns.lineplot(x="year", y="manufacturing", data=df)


# The **Manufacturing** industry in India have boomed after 2012 and have reached the maximum during the Modi Government. But During Manmohan Singh Government, from 2011-2012 the Manufacturing had a rapid growth

# In[ ]:


sns.lineplot(x="year", y="electricity gas & water supply", color="coral", data=df)


# **Electricy,Gas and Water Supply** have incresed post 2012. From 2011-2012, the supply have seen a rapid growth before Modi Government. 

# In[ ]:


sns.lineplot(x="year", y="construction", color="purple", data=df)


# The graph shows the development in the field of **Construction**. We can see clearly from the Graph that, the major development in the Construction field happened between the year **2011-2012**. 

# In[ ]:


sns.lineplot(x="year", y="industry", color="navy", data=df)


# For Industries, India have seen a boom in between the year 2011-2012 during Manmohan Singh Goverment and then a gradual growth during Modi Government.

# In[ ]:


sns.lineplot(x="year", y="industry", color="navy", data=df)

