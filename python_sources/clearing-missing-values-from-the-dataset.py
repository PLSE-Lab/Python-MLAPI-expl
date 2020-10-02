#!/usr/bin/env python
# coding: utf-8

# **Clearing missing values by using Missingno library**

# In[5]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib_venn as venn
from math import pi
from pandas.tools.plotting import parallel_coordinates
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[6]:


data = pd.read_csv("../input/aps_failure_test_set.csv")


# **Reviewing the data**

# In[7]:


data.info()


# In[8]:


data.head()


# **Adjusting the data for clearing**

# In[25]:


sample_data = data.loc[:,['ab_000','ac_000','ad_000']]
sample_data.ab_000.replace(['na'],np.nan, inplace = True)
sample_data.ac_000.replace(['na'],np.nan, inplace = True)
sample_data.ad_000.replace(['na'],np.nan, inplace = True)
data_missingno = pd.DataFrame( sample_data.head(20))


# In[26]:


data_missingno


# **Using missingno library**
# * There is a graph which is the left hand side of the table like a histogram. This graph shows number of non-missing values.
# * The left side of the table, there is first and last numbers of the samples.
# * Black color means there is a non-missing value.
# * White color means there is a missing value.

# In[27]:


import missingno as msno
msno.matrix(data_missingno)
plt.show()


# **Different type of demonstration of the missing values. (Bar Plot)**
# * The left hand side of the bars show percentage of values.
# * The right hand side of the bars show number of samples.
# * The top of the the bars show number of non-missing values.
# * The bottom of the each bar shows name of the feature.

# In[28]:


msno.bar(data_missingno)
plt.show()

