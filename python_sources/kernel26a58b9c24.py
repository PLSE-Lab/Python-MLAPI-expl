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


# 

# In[ ]:


covid_in = pd.read_excel('/kaggle/input/weekly-data/new_data.xlsx')
covid_it = pd.read_csv('/kaggle/input/covid19-in-italy/covid19_italy_province.csv') 
print(covid_it)
hospital_bed=pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')
#hospital_bed 
#covid_in


# In[ ]:


import matplotlib.pyplot as plt
#pd.crosstab(index=covid_in['ConfirmedIndianNational'],columns=covid_in['Date'])
plt.scatter(covid_in['ConfirmedIndianNational'],covid_in['Week'])
plt.title("NUMBER OF CASES PER WEEK")
plt.xlabel('Number of cases')
plt.ylabel('Week')
plt.figure(figsize=[18,18])

plt.show()


# In[ ]:


import seaborn as sns
plt.figure(figsize=[18,18])

plot=sns.boxplot(x="ConfirmedIndianNational",y="State/UnionTerritory",data=covid_in)
plot.set_title('Number of Cases per State as per current data')


# In[ ]:


sns.countplot(x='ConfirmedIndianNational',hue="State/UnionTerritory",data=covid_in)
plt.figure(figsize=[100,50])


# In[ ]:




