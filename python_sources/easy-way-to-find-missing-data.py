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


# # Missing Data

# This notebook will show two simple ways to figure out missing data for machine learning modeling,

# ### 1. Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# ### 2. Data Loading

# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# ### 3. Solution

# There is 2 simple ways to check missing values in a data frame.

# #### 3.1 Sum of all missing values

# Getting sum of all ```True``` values for each column.

# In[ ]:


train.isnull().sum()


# #### 3.2 Heatmap

# In case you have a dataset with not large amount of features, you can visually show missing data.

# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(train.isnull(), cmap="YlGnBu")
plt.tight_layout()


# # THANKS
