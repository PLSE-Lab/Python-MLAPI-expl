#!/usr/bin/env python
# coding: utf-8

# **DATA Description**

# **Importing Packages**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas_profiling # pandas profiling
import matplotlib.pyplot as plt                                    # Plotting library for Python programming language and it's numerical mathematics extension NumPy
import seaborn as sns                                              # Provides a high level interface for drawing attractive and informative statistical graphics
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Importing Dataset**

# In[ ]:


medical_data = pd.read_csv("../input/insurance.csv")
medical_data.head()


# In[ ]:


medical_data.shape


# There are **1338 observations** and **7 columns** in the dataset

# In[ ]:


medical_data.info()


# No **missing data** in the dataset

# In[ ]:


medical_data.describe(include='all')


# * Contains data of people between Age 18-64. Children are categorized separately.
# * Most of the insurers are Males (676),non-smokers(1064) and from the Southeast region(364) out of the whole dataset.

# **DATA Distribution**

# * **Age Distribution**

# In[ ]:


plt.hist(medical_data.age,facecolor='green')
plt.show()


# * Most of the Insurers are from the Age-group 18-20. Rest all are equally distributed.

# * **Gender Distribution**

# In[ ]:


plt.hist(medical_data.sex,facecolor='green')
plt.show()

