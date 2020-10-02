#!/usr/bin/env python
# coding: utf-8

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


# # ONE LINE PYTHON SERIES
# 
# One Line Python is beginners guide for shortest way to learn python tips and tricks.
# 
# This Series show you how to analyze data with minimum effort. You don't need to know high level python programming for good looking notebooks. Just follow series and use what you have learned in your notebooks.
# 
# **I hope you find this notebook helpful and some <font color="red"><b>UPVOTES</b></font> would be very much appreciated.**
# 
# ### First code is <font color="green"><b>pandas_profiling</b></font>
# 
# This magical code shows general view - variable distributions - correlations - missing values in one place
# 
# Follow the below code
# 
# 
# 
# 
# 
# 
# See you on other one line Python series
# 
# ## [One Line Python - Part 2 - Image Link](https://www.kaggle.com/medyasun/one-line-python-part-2-image-link)
# ## [One Line Python - Web Scraping For Beginners](https://www.kaggle.com/medyasun/one-line-python-web-scraping-for-beginners)
# 
# * Tip:  **!pip install -U pandas-profiling**  Use that code for latest version of Pandas Profiling. Thanks to simon
# 

# In[ ]:


import pandas_profiling        #.  we need to import pandas profiling method
hd=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
hd.profile_report()          

