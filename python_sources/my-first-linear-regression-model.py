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


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[ ]:


data=pd.read_csv('/kaggle/input/years-of-experience-and-salary-dataset/Salary_Data.csv')


# In[ ]:


data


# In[ ]:


data.describe()


# In[ ]:


y=data['Salary']
x1=data['YearsExperience']


# In[ ]:


plt.scatter(x1,y)
plt.xlabel('Years of Experience',fontsize=22)
plt.ylabel('Income',fontsize=22)
plt.show()


# In[ ]:


x=sm.add_constant(x1)
result=sm.OLS(y,x).fit()
result.summary()


# In[ ]:


plt.scatter(x1,y)
yhat=9449.9623 * x1 + 2.579e+04
line=plt.plot(x1,yhat,lw=4,c='orange',label='Fittest Regression Line')
plt.xlabel('Years of Experience',fontsize=22)
plt.ylabel('Income',fontsize=22)
plt.show()

