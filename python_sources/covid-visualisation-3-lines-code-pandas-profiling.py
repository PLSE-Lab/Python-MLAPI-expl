#!/usr/bin/env python
# coding: utf-8

# # In this notebook i am going to use pandas profiling,If u are not aware of this,please visit my below notebook where i clearly explained about pandas profiling.

# https://www.kaggle.com/sainathkrothapalli/pandas-profiling-tutorial-on-titanic-dataset

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Loading data,shape,describe(),info() methods

# In[ ]:


data=pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
data.head()


# In[ ]:


data.shape


# In[ ]:


data.describe()


# In[ ]:


data.info()


# # EDA Pandas Profiling

# Importing the ProfileReport class from pandas_profiling library

# In[ ]:


from pandas_profiling import ProfileReport


# Now create an object named profile_report.

# In[ ]:


profile_report=ProfileReport(data, title='Covid-19 Report')


# We can see our report using to_widgets() method as shown below.

# In[ ]:


profile_report.to_widgets()


# We can also represent this report in the form of html page and sent the report to any one.It can be done using to_file method.

# In[ ]:


#profile_report.to_file('Htmlfile.html')

