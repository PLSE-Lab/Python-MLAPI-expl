#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# INFORMTION ABOUT THE DATASET.

# In[ ]:


performance=pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
performance.head()


# In[ ]:


performance.describe()


# In[ ]:


performance.info()


# In[ ]:


student=performance.drop(['test preparation course','lunch','race/ethnicity'],axis=1)
student


# In[ ]:


student.iloc[5:995]


# In[ ]:


student.plot( figsize=(5,5),subplots=True)


# REGRESSION MODEL 

# In[ ]:


lm=LinearRegression()
x=student['math score'].values.reshape(-1,1)
y=student['writing score'].values.reshape(-1,1)
f=lm.fit(x,y)
z=lm.predict(x)
z


# In[ ]:


lm.coef_


# In[ ]:


lm.intercept_


# In[ ]:


plt.scatter(x,y)


# In[ ]:


student.corr()


# NORMAISING THE DATA

# In[ ]:


math_score=(student['math score']-student['math score'].mean())/student['math score'].std()
reading_score=(student['reading score']-student['reading score'].mean())/student['reading score'].std()
writing_score=(student['writing score']-student['writing score'].mean())/student['writing score'].std()
math_score


# In[ ]:


reading_score


# In[ ]:


writing_score


# TOTAL NUMBER OF MALE,FEMALE CANDIDATES 

# In[ ]:


male_candidate=student['gender']=='male'
male_candidate.value_counts()


# NUMBER OF CANDIDATES WITH MORE THAN 50 IN MATH SUBJECT

# In[ ]:


Pass_maths=student['math score']>=50
Pass_maths.value_counts()


# AVERAGE CANDIDATES PASS

# In[ ]:


Pass_maths.mean()


# NUMBER OF CANDIDATES WITH MORE THAN 50 IN READING SUBJECT

# In[ ]:


Pass_reading=student['reading score']>=50
Pass_reading.value_counts()


# AVERAGE CANDIDATES PASS

# In[ ]:


Pass_reading.mean()


# NUMBER OF CANDIDATES WITH MORE THAN 50 IN WRITING SUBJECT

# In[ ]:


Pass_writing=student['writing score']>=50
Pass_writing.value_counts()


# AVERAGE CANDIDATES PASS

# In[ ]:


Pass_writing.mean()


# FROM ABOVE IT CAN BE SEEN THAT 91% OF CANDIDATES ARE PASS IN READING SUBJECT WHICH IS THE HIGHEST OF OTHER TWO SUBJECTS.

# SUBJECTS WISE COMPARISION OF SCORES VIA DIFFERENT CHARTS

# In[ ]:


student.plot.hist(figsize=(17,7),alpha=0.6)
plt.xlabel('Marks obtained')
plt.ylabel('Number of students')
plt.show()


# In[ ]:


student.plot.kde(figsize=(6,6))


# In[ ]:


student.plot.scatter(x='math score',y='reading score')


# In[ ]:


student.plot.scatter(x='math score',y='writing score')


# In[ ]:


student.plot.scatter(x='writing score',y='reading score')


# SHOWING NUMBER OF MALE,FEMALE CANDIDATES PASS-FAIL PERCENTILE IN RESPECTIVE SUBJECTS

# In[ ]:


group=student.groupby('gender')['math score','reading score','writing score'].mean()
group


# In[ ]:


group.plot()


# In[ ]:


group.plot.pie(figsize=(12,10),subplots=True,autopct='%1.1f%%',)

