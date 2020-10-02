#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy import stats
sales=pd.read_csv('../input/sales.csv')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory




# Any results you write to the current directory are saved as output.


# 
#  ## <font color=turquoise >Exploratory Analysis of Data </font> 

# In[4]:


plt.hist(sales['Phones'],color='teal')
plt.title('Histogram of Sales of Phones')
plt.xlabel('Phones')
plt.ylabel('Sales Frequency')
plt.show()


# In[5]:


plt.hist(sales['Tvs'],color='turquoise')
plt.title('Histogram of Sales of Tvs')
plt.xlabel('Tvs')
plt.ylabel('Sales Frequency')
plt.show()


# In[8]:



sales.describe()


# ## <font color=red>TTEST</font> 

# **Hypothesis Test **
# - Ho: there is no significant difference between sale of phones and sale of Tvs
# - H1: There is a difference between Phone and Tvs sales

# In[9]:


ttest=stats.ttest_ind(sales['Phones'],sales['Tvs'])
print ('t-test independent', ttest)


# **Rule** <br>
# > <br>
# if 'p <0.05' reject ho <br>
# else do not reject ho, <br>
# <br>
# in this case p(0.88) >0.05 and we can reject Ho at 5% significant level and conclude that there is no significant difference between the means sales of Tvs and Phones

# In[ ]:





# In[ ]:




