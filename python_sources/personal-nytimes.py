#!/usr/bin/env python
# coding: utf-8

# This is the first kernel I have started and attempted to perform analyis without forking, the goal is to produce some visualizations that are actually useful

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Importing modules equivalent to forked kernel

# In[ ]:


from textblob import TextBlob

import warnings 
warnings.filterwarnings('ignore')

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
init_notebook_mode(connected=True)
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# Reading in the first data set, taking a peek at some introductory information to structure and format of the data

# In[ ]:


April_2017 = pd.read_csv('../input/ArticlesApril2017.csv')
print(April_2017.info())


# In[ ]:


April_2017.head()


# In[ ]:


April_2017.tail()


# Here I am starting to poke at the data with some quick plots to tease out any immediate trends

# In[ ]:


sorted(April_2017)


# I then isolate newDesk as its own dataFrame this way I can manipulate that specific column and plot that data

# In[ ]:


April_2017.shape


# In[ ]:


newDesk_df = April_2017[['newDesk']].copy()
newDesk_df.shape


# Now I have to encode this dataFrame from categorical data to numerical data that vibes with my plotting methods. I show here that the column only contains object data types

# In[ ]:


newDesk_df.dtypes


# Then I use .plot(kind="") to generate a bar graph

# In[ ]:


newDesk_df['newDesk'].value_counts().plot(kind='bar')

