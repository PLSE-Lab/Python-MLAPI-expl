#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/HR_comma_sep.csv')
print(data.describe())


# In[ ]:


#print(data.info)
print(data.head())
print('average_montly_hours vs left')
plt.plot(data['average_montly_hours'],data['left'], 'r')
plt.show()
print('the above shows that people tend to leave when underutillized')


# In[ ]:


print('satisfaction_level vs left')
plt.plot(data['satisfaction_level'],data['left'], 'r')
plt.show()
print('the above shows that satisfaction level is not a big player to decide if an employee stays back')


# In[ ]:


print('number_project vs time_spend_company along with a plotting for people left')
plt.plot(data['time_spend_company'],data['number_project'], 'ro',
         data['time_spend_company'],data['left'], 'r+',
         data['left'],data['number_project'], 'r*') 
#plt.show()
#print('the above shows that less number of projects contribute to the employee stack leaving more')


# In[ ]:




