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


# PHC- 2018 Data set performing some basic anayicts

# In[ ]:


#import required libraries
import pandas as pd
import matplotlib.pyplot as plt


# import .xls file 

# In[ ]:


#import data set
phc_data = pd.read_excel('/kaggle/input/phc2018/datafile.xls')


# **Renaming columns**

# In[ ]:


phc_data.columns = ["Sl_no","State","Required","Sanctioned","In Position","Vacant","Shortfall"]


# **Print top 5 data in phc data set**

# In[ ]:


phc_data.head()


# **Print top 10 data in phc data set**

# In[ ]:


phc_data.head(10)

Print last 5 data
# In[ ]:


phc_data.tail()

Print Last 10 data  
# In[ ]:


phc_data.tail(10)


# In[ ]:


phc_data.info()

print data set NAN count
# In[ ]:


phc_data.isnull().sum()


# In[ ]:


phc_data.isna().sum()


# In[ ]:


all_row_max =phc_data.max()
all_row_max


# **Remove last row in dataset**

# In[ ]:


#selecting all the row except last
phc_data = phc_data[:-1]
phc_data.tail()


# **Fill zero instated on NAN**

# In[ ]:


phc_data.fillna(0, inplace=True)

