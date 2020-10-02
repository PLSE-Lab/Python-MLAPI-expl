#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/2016-FCC-New-Coders-Survey-Data.csv")


# In[ ]:


import pylab as P


# In[ ]:


data['Age'].hist()
P.show()


# In[ ]:


import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


data['BootcampPostSalary'].hist()
P.show


# In[ ]:


Majors = data['SchoolMajor'].value_counts()
MajorsPlt = Majors[:30].plot(kind='bar')
MajorsPlt.set_title('Majors')
MajorsPlt.set_xlabel('Name')


# In[ ]:


DegreeType = data['SchoolDegree'].value_counts()
DegreeTypePlot = DegreeType.plot(kind='bar')
DegreeTypePlot.set_xlabel('Type of Degree')
DegreeTypePlot.set_title('Frequency')


# In[ ]:


matplotlib.pyplot.scatter(data['Age'], data['BootcampPostSalary'])
matplotlib.pyplot.show()


# In[ ]:


matplotlib.pyplot.scatter(data['ChildrenNumber'], data['BootcampPostSalary'])
matplotlib.pyplot.show()


# In[ ]:


matplotlib.pyplot.scatter(data['Income'], data['BootcampPostSalary'])
matplotlib.pyplot.show()


# In[ ]:


Language = data['LanguageAtHome'].value_counts()
LangPlot = Language[:10].plot(kind='bar')
LangPlot.set_xlabel('Language')
LangPlot.set_title('Languages spoken at home')


# In[ ]:


matplotlib.pyplot.scatter(data['MonthsProgramming'], data['BootcampPostSalary'])
matplotlib.pyplot.show()


# In[ ]:


JobPref = data['JobPref'].value_counts()
JobPrefPlot = JobPref[:5].plot(kind='bar')
JobPrefPlot.set_xlabel('Job Preferences')
JobPrefPlot.set_title('Preferences for future jobs')


# In[ ]:


plt.scatter(data['ExpectedEarning'], data['Income'], c='red', alpha=1.0)
plt.show()


# In[ ]:


plt.scatter(data['ExpectedEarning'], data['Age'], c='blue', alpha=1.0)
plt.show()


# In[ ]:




