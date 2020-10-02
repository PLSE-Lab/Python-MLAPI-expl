#!/usr/bin/env python
# coding: utf-8

# 

# # AutoViz
# we all had a lot of difficulties in visualizing the basic plots like bar chart for various categorical varibales , histograms for the 
# 
# various distributions measurement etc.
# 
# It is such a pain to keep writing the same code over and over again for same tasks for different number of datasets.
# 
# Not many use tableau , qliksense and lots of many other visualisation tools which provides some cool features on vizualising the dataset.
# 
# Here comes the pythons **AutoViz** Library where you can visualise the entire dataset with just a single line of code.
# 
# you need to input only the values like:
# 1. **filename** - which is the name of the file
# 2. **Sep** - The seperators that are used in the dataset
#     . example: ',' for csv files
# 3. **Target** - The target variable in the dataset.
# 
# you can install AutoViz onto your systems by using 
# 
# **pip install autoviz**
# 
# Head over to the below link to learn more about the project
# 
# [https://pypi.org/project/autoviz/ ]
# 
# 

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


# In[ ]:


get_ipython().system('pip install autoviz')


# In[ ]:


from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()


# In[ ]:


df = pd.read_csv("/kaggle/input/train.csv")


# In[ ]:


filename = '/kaggle/input/train.csv'
sep = ','
target = "Loan_Status"
dft = AV.AutoViz(filename, sep, target, df, header=0, verbose=0, lowess=False, chart_format='svg', max_rows_analyzed=150000, max_cols_analyzed=30)


# In[ ]:




