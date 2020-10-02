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

import os
import matplotlib.pyplot as plt
import seaborn as sns

# Any results you write to the current directory are saved as output.


# In[ ]:


students = pd.read_csv('../input/StudentsPerformance.csv')


# In[ ]:


students.head(10)


# In[ ]:


students.describe()


# In[ ]:


students.info()


# In[ ]:


print("---gender---")
print(students.gender.value_counts())
print("--Number of parental level of education--")
print(students['parental level of education'].value_counts())


# In[ ]:


print("--lunch-type---")
print(students.lunch.value_counts())
print("<<<Test preparation course>>>")
print(students['test preparation course'].value_counts())


# In[ ]:


print(students['race/ethnicity'].value_counts().plot(kind = 'bar'))
sns.set()
plt.ylabel('Total')
plt.show()


# In[ ]:


print("-------Female")
print(students['race/ethnicity'][students.gender == 'female'].value_counts(normalize = True))
print("-------Male")
print(students['race/ethnicity'][students.gender == 'male'].value_counts(normalize = True))


# This is the maximum and minimum marks obtained by the Male and female in Math,reading and writing.

# In[ ]:


print(students.groupby('gender')[['math score','reading score','writing score']].agg(['max','min']))


# Females are obtaining less marks in maths than males,but reading and writing scores of females is far better than males.

# In[ ]:


print(students.groupby('gender')['math score','reading score','writing score'].mean())


# 8.6% in parental level of education of males candidates have acquired master degree which is 5% greater than females.May be this can be a factor of high marks obtained by the males in maths,but there is no solid proof to justify this equation.

# In[ ]:


female = students[(students.gender == 'female') & (students['race/ethnicity'] == 'group C')]
male = students[(students.gender == 'male') & (students['race/ethnicity'] == 'group C')]
print("----females----")
print(female['parental level of education'].value_counts(normalize = True))
print("----Males---")
print(male['parental level of education'].value_counts(normalize = True))


# In[ ]:


female1 = students[students.gender == 'female']
male1 = students[students.gender == 'male']

print("--female--")
print(female1['race/ethnicity'].value_counts())
print("--male--")
print(male1['race/ethnicity'].value_counts())


# males are having more number of standard lunch than females.It might be a reason for greater marks in maths.

# In[ ]:


print("---female--")
print(female1['lunch'].value_counts(normalize = True))
print("---male--")
print(male1.lunch.value_counts(normalize = True))


# Here males are in more percentage in completed category of test preparation course as compared to females.May be this can be reason for males to achieve more marks in maths than females

# In[ ]:


print("---female--")
print(female1['test preparation course'].value_counts(normalize = True))
print("---male--")
print(male1['test preparation course'].value_counts(normalize = True))


# Both male and female students obtained more marks when they took standard lunch and completed their test preparation course.

# In[ ]:


print("--female--")
print(female1.groupby(['lunch','test preparation course'])['math score'].max().unstack(level = 'lunch'))
print("--male--")
print(male1.groupby(['lunch','test preparation course'])['math score'].max().unstack(level = 'lunch'))


# In[ ]:


print(female1.groupby('race/ethnicity')['math score'].agg(['min','max']))
print(male1.groupby('race/ethnicity')['math score'].agg(['min','max']))


# Both male and female students obtained more marks in maths when they took standard lunch.

# In[ ]:


print(female1.groupby('lunch')['math score'].agg(['min','max']))
print(male1.groupby('lunch')['math score'].agg(['min','max']))


# I am comparing group C and E because students were obtained minimum marks in these category.

# In[ ]:


print("---Female---")
groupE = students[(students.gender == 'female') & (students['race/ethnicity'] == 'group E')]
groupC = students[(students.gender == 'female') & (students['race/ethnicity'] == 'group C')]
print(groupE.lunch.value_counts(normalize=True))
print(groupC.lunch.value_counts(normalize=True))
print('---courses---')
print(groupE['test preparation course'].value_counts(normalize=True))
print(groupC['test preparation course'].value_counts(normalize=True))


# Marks of male students in group E is greater than group c because more number of students in group E showed interest in completing their test preparation course.There are also more number of standard lunch in lunch category.

# In[ ]:


print("----Male---")
groupE = students[(students.gender == 'male') & (students['race/ethnicity'] == 'group E')]
groupC = students[(students.gender == 'male') & (students['race/ethnicity'] == 'group C')]
print(groupE.lunch.value_counts(normalize=True))
print(groupC.lunch.value_counts(normalize=True))
print("---Courses on groupE---")
print(groupE['test preparation course'].value_counts(normalize=True))
print("---Courses on groupC---")
print(groupC['test preparation course'].value_counts(normalize=True))


# Some Data visualisation..

# In[ ]:


students.head()


# In[ ]:


plt.subplot(1,2,1)
sns.violinplot(y = 'writing score', data = students)
plt.subplot(1,2,2)
sns.violinplot(y = 'reading score' ,data = students)
plt.tight_layout()
plt.show()


# females has more number of 100 marks in both reading and writing than males.

# In[ ]:


print(students.groupby('gender')['reading score','writing score'].agg(['min','max']))


# In[ ]:


reading_score = students[(students['reading score'] == 100)]
writing_score = students[(students['writing score'] == 100)]
print(reading_score.groupby('gender')['reading score'].value_counts().plot(kind = 'bar'))
plt.show()
print(writing_score.groupby('gender')['writing score'].value_counts().plot(kind = 'bar'))
plt.show()


# More female students are getting 100 marks in all the race/ethnicity in both reading and writing.

# In[ ]:


sns.countplot(x = 'race/ethnicity' , data = reading_score,hue = 'gender',palette='dark')
plt.show()


# In[ ]:


sns.countplot(x = 'race/ethnicity' , data = writing_score , hue = 'gender' , palette= 'dark')
plt.show()


# Both male and female students showed increase in their marks when they get standard lunch and completed their test preparation course.

# In[ ]:


male1.groupby(['lunch','test preparation course'])['writing score','reading score','math score'].mean()


# In[ ]:


female1.groupby(['lunch','test preparation course'])['writing score','reading score','math score'].mean()


# Well,male students are performing good in math because of more standard and completed lunch than female students and we can conclude that for achieving good marks in maths test preparation course and lunch is two main factors.

# In[ ]:




