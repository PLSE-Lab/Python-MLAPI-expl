#!/usr/bin/env python
# coding: utf-8

# *Hello there!, this is my first kernel on kaggle and also the first time of me testing my skills on a real world dataset.I have learned and understood a lot by referring to some kernels and following their way of solving a problem.I'm grateful for what I've learnt and ready to start solving problems.
# So, take a look at my first kernel and let me know how did you find it,what else I can do to improve my analysing skills and some suggestions to make it better.
# 
# Thanks!*

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#importing the dataset
dataset=pd.read_csv('../input/StudentsPerformance.csv')


# In[ ]:


#Printing the top few rows to read column names
dataset.head()


# In[ ]:


#Plotting test preparation scores against the maths,reading and writing scores to see which gender perfroms better
plt.figure(figsize=(18,8))
plt.subplot(1,4,1)
sns.barplot(x='test preparation course',y='math score',data=dataset,hue='gender',palette="Blues_d")
plt.title('Maths Scores')

plt.subplot(1, 4, 2)
sns.barplot(x='test preparation course',y='reading score',data=dataset,hue='gender',palette="Blues_d")
plt.title('reading scores')

plt.subplot(1, 4, 3)
sns.barplot(x='test preparation course',y='writing score',data=dataset,hue='gender',palette="Blues_d")
plt.title('writing scores')
plt.show()


# From the above output plots we infer that the maths scores of the male students are better,
# the reading scores of female students are even better when the test preparation course is completed.The female students perfrom extremely well in reading when the test course is completed.
# 
# 

# In[ ]:


#Plotting lunch against the maths,reading and writing scores to see which gender perfroms better
plt.figure(figsize=(18,8))
plt.subplot(1,4,1)
sns.barplot(x='lunch',y='math score',data=dataset,hue='gender',palette="Blues_d")
plt.title('Maths Scores')

plt.subplot(1, 4, 2)
sns.barplot(x='lunch',y='reading score',data=dataset,hue='gender',palette="Blues_d")
plt.title('reading scores')

plt.subplot(1, 4, 3)
sns.barplot(x='lunch',y='writing score',data=dataset,hue='gender',palette="Blues_d")
plt.title('writing scores')
plt.show()


# From the above output plots we infer that the students score better only when they are offered a standard lunch

# In[ ]:


#Plotting race against the maths,reading and writing scores to see which gender perfroms better
plt.figure(figsize=(18,8))
plt.subplot(1,4,1)
sns.barplot(x='race/ethnicity',y='math score',data=dataset,hue='gender',palette="Blues_d")
plt.title('Maths Scores')

plt.subplot(1, 4, 2)
sns.barplot(x='race/ethnicity',y='reading score',data=dataset,hue='gender',palette="Blues_d")
plt.title('reading scores')

plt.subplot(1, 4, 3)
sns.barplot(x='race/ethnicity',y='writing score',data=dataset,hue='gender',palette="Blues_d")
plt.title('writing scores')
plt.show()


# In the above plot we see that the group E scores well in all the three subjects.

# In[ ]:


#plotting a graph with parental level of education against count to see how important is parent's education
fig,ax=plt.subplots()
sns.countplot(x='parental level of education',data=dataset)
plt.tight_layout()
fig.autofmt_xdate()


# From this output plot we infer that there are very less parents with a master degree and most of them have a college study and associate's degree

# In[ ]:


#calculating the average
dataset['Total score']=dataset['math score']+dataset['reading score']+dataset['writing score']
print("Average score is   : {}".format(np.mean(dataset['Total score'])/3))
#on looking at the average score we decide the pass mark,let pass mark be 50
passing_score=50


# In[ ]:


#maths passing score
dataset['Maths_pass_students'] = np.where(dataset['math score']<passing_score, 'F', 'P')
dataset.Maths_pass_students.value_counts()
#reading passing score
dataset['reading_pass_students'] = np.where(dataset['reading score']<passing_score, 'F', 'P')
dataset.reading_pass_students.value_counts()
#writing passing score
dataset['writing_pass_students'] = np.where(dataset['writing score']<passing_score, 'F', 'P')
dataset.writing_pass_students.value_counts()


# In[ ]:


#checking overall passed students
dataset['OverAllPassStudents'] = dataset.apply(lambda x : 'F' if x['Maths_pass_students'] == 'F' or x['reading_pass_students'] == 'F' or x['writing_pass_students'] == 'F' else 'P', axis =1)

dataset.OverAllPassStudents.value_counts()


# In[ ]:


#Plotting a graph to see which group has more number of pass students
fig,ax=plt.subplots()
sns.countplot(x='race/ethnicity', data = dataset, hue='OverAllPassStudents', palette='Blues_d')
plt.tight_layout()
fig.autofmt_xdate()


# The group C students have passed the most and are well performing.

# In[ ]:


#finding percentage and assigning grades
dataset['Percentage'] = dataset['Total score']/3


# Assigning the grades let above 80 be A Grade,70 to 80 = B Grade,60 to 70 = C Grade,50 to 60 = D Grade,40 to 50 = E Grade
# and let below 40 be F Grade.

# In[ ]:


def GetGrade(Percentage, OverAllPassStudents):
    if ( OverAllPassStudents == 'F'):
        return 'F'    
    if ( Percentage >= 80 ):
        return 'A'
    if ( Percentage >= 70):
        return 'B'
    if ( Percentage >= 60):
        return 'C'
    if ( Percentage >= 50):
        return 'D'
    if ( Percentage >= 40):
        return 'E'
    else: 
        return 'F'

dataset['Grade'] = dataset.apply(lambda x : GetGrade(x['Percentage'], x['OverAllPassStudents']), axis=1)

dataset.Grade.value_counts()


# In[ ]:


#plotting the grades assigned
sns.countplot(x="Grade", data = dataset, order=['A','B','C','D','E','F'],  palette="muted")
plt.show()


# In[ ]:


#Seeing which group gets good grades
fig,ax=plt.subplots()
sns.countplot(x='race/ethnicity', data = dataset, hue='Grade', palette='Blues_d')
plt.tight_layout()
fig.autofmt_xdate()


# * The A grade is scored the most by Group C followed by Group D
# * The B Grade is scored the most by Group C followed by Group D and least by Students who fall into Group A Category
# * The C Grade is scored the most by Group C students followed by Group D and Group B
# * The D grade is scored the most by Group C and least by Group A
# * The F grade is scored the most by Group C and least by Group E
# * The most number of passing students are from Group C and the most number of failing students are also from Group C.

# In[ ]:




