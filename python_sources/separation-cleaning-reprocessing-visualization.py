#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import os
print(os.listdir("../input"))
plt.rcParams['figure.figsize'] = (12.0, 8.0) 
plt.rcParams['figure.dpi'] = 300 
# Any results you write to the current directory are saved as output.


# In[ ]:


dataStudent = pd.read_csv('../input/students.csv',encoding='utf-8')
dataTeacher = pd.read_csv('../input/professionals.csv',encoding='utf-8')


# #  Understanding of data

# In[ ]:


dataTeacher.head()


# In[ ]:


dataStudent.head()


# In[ ]:


dataStudent.info()


# In[ ]:


dataTeacher.info()


# # Simple processing of data

# ### Separate individual labels

# In[ ]:


dataStudentLoc = dataStudent['students_location']
dataStudentDat = dataStudent['students_date_joined']
dataTeacherLoc = dataTeacher['professionals_location']
dataTeacherInd = dataTeacher['professionals_industry']
dataTeacherHea = dataTeacher['professionals_headline']
dataTeacherDat = dataTeacher['professionals_date_joined']
dataStudentDat


# ### Missing value processing

# In[ ]:


dataStudentLoc = dataStudentLoc.dropna(how='any')
dataTeacherLoc = dataTeacherLoc.dropna(how='any')
dataTeacherInd = dataTeacherInd.dropna(how='any')
dataTeacherHea = dataTeacherHea.dropna(how='any')
dataStudentLoc


# ### Define simple functions and complete classification statistics to synthesize new arrays

# In[ ]:


def sorted(dataset):
    datasetNum = dataset.value_counts()
    datasetLab = dataset.unique()
    
    return np.column_stack((datasetLab,datasetNum))
    
# dataStudentLocNum = dataStudentLoc.value_counts()
# dataStudentLocLab = dataStudentLoc.unique()
# studentLoc = np.column_stack((dataStudentLocLab,dataStudentLocNum))
studentLoc = sorted(dataStudentLoc)
teacherLoc = sorted(dataTeacherLoc)
teacherInd = sorted(dataTeacherInd)
teacherHea = sorted(dataTeacherHea)
teacherInd


# ### Slice the date, accurate to the year, you can also choose the month. At the same time, you can also process it by converting it to a date object in pandas.

# In[ ]:


def cutDate(dataset):
    for i in range(len(dataset)):
        dataset[i] = dataset[i][:4]
        
    return dataset

studentDat = cutDate(dataStudentDat)
teacherIDat = cutDate(dataTeacherDat)
studentDat


# ### Sorting and categorizing years by collections

# In[ ]:


from collections import Counter
studentDat = Counter(studentDat)
teacherIDat = Counter(teacherIDat)
teacherIDat


# # Drawing graphics

# ### Separate X, Y from Counter

# ### Student's address distribution top ten

# In[ ]:


studentx = studentLoc[:,1]
studenty = studentLoc[:,0]
studenty[:9]


# In[ ]:


sns.barplot(studentx[:9],studenty[:9])
plt.xlabel("Number",fontsize=15)
plt.ylabel("Loc",fontsize=15)
plt.title("Student's address distribution top ten")
plt.show()


# ### Teacher's address distribution top ten

# In[ ]:


teacherx = teacherLoc[:,1]
teachery = teacherLoc[:,0]
sns.barplot(teacherx[:9],teachery[:9])
plt.title("Teacher's address distribution top ten")
plt.xlabel("Number",fontsize=15)
plt.ylabel("Loc",fontsize=15)
plt.show()


# ### Line chart showing the increase in the number of students and teachers per year

# In[ ]:


studentDatX = studentDat.keys()
studentDatY = studentDat.values()
teacherDatX = teacherIDat.keys()
teacherDatY = teacherIDat.values()
plt.plot(studentDatX,studentDatY,'r',label='student')
plt.plot(teacherDatX,teacherDatY,'b',label='teacher')
plt.title(" increase in the number of students and teachers per year")
plt.xlabel('Years')
plt.ylabel('Number')
plt.legend()
plt.show()


# ### Pie chart of professionals_industry

# In[ ]:


teacherInd


# In[ ]:


teacherIndlab = teacherInd[:,0]
teacherIndnum = teacherInd[:,1]
explode = (0.02,0.015,0.012,0.010,0.01,0.1,0.005,0,0)
plt.pie(teacherIndnum[:9],labels=teacherIndlab[:9],explode=explode,autopct='%1.1f%%',shadow=True)
plt.axis('equal')
plt.legend()
 
plt.show()
# teacherIndlab


# In[ ]:


teacherHealab = teacherHea[:,0]
teacherHeanum = teacherHea[:,1]
explode = (0.02,0.015,0.012,0.010,0.01,0.1,0.005,0,0)
plt.pie(teacherHeanum[:9],labels=teacherHealab[:9],explode=explode,autopct='%1.1f%%',shadow=True)
plt.axis('equal')
plt.legend()
 
plt.show()
# teacherHealab


# 
#     This is my first time posting a work on kaggle. This work is to help you learn machine learning by extracting various data and merging new array objects. After that, I will continue to explore the content of the remaining data sets, and at the end look for interesting questions and build model predictions or classifications to solve the problem.
#     Here, I am very willing to communicate with you and build friendships and even team up.
#     thank you all.
