#!/usr/bin/env python
# coding: utf-8

# <h2><font color=red>If you like my work please consider upvoting it. Thank you.</font></h2>

# ## Importing libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Reading CSV

# In[ ]:


students = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')


# ## Peeking into data

# In[ ]:


students.columns


# In[ ]:


students.head()


# In[ ]:


students.shape


# In[ ]:


students.isnull().sum()


# - There are 1000 rows and 8 columns
# - Column headers are 'gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 'math score', 'reading score' and 'writing score'
# - There is no null columns

# ## Distribution of scores

# ### Maths score

# In[ ]:


sns.distplot(students['math score'], bins=20, color='orange')
plt.grid()


# - Most students scored between 60 - 75%. 

# ### Reading score

# In[ ]:


sns.distplot(students['reading score'], bins=20, color='orange')
plt.grid()


# - Most students scored between 60 - 80%

# ### Writing score

# In[ ]:


sns.distplot(students['writing score'], bins=20, color='orange')
plt.grid()


# - Students scored between 65 - 80%

# ## Distribution of catagories

# ### Gender count

# In[ ]:


sns.catplot(y="gender",  kind="count", height=6, aspect=2, data=students);


# In[ ]:


students['gender'].value_counts(normalize=True) * 100


# Female students are slightly more in count when compared to male students.

# ### Race/ethnicity count

# In[ ]:


sns.catplot(y="race/ethnicity",  kind="count", height=6, aspect=2, data=students);


# In[ ]:


students['race/ethnicity'].value_counts(normalize=True) * 100


# - More students are from group C. 31.9% of total sudents belongs to group C.
# - 8.9% students belongs to group A and it is the smallest group.

# ### Parent level of education

# In[ ]:


sns.catplot(y="parental level of education",  kind="count", height=6, aspect=2, data=students);


# In[ ]:


students['parental level of education'].value_counts(normalize=True) * 100


# - Parents of most students have some college education. A total of 62.4% have some college education.
# - Only 5.9% parents have masters degree.
# - Most parents have some college degree or associate degree

# ### Lunch count

# In[ ]:


sns.catplot(y="lunch",  kind="count", height=6, aspect=2, data=students);


# In[ ]:


students['lunch'].value_counts(normalize=True) * 100


# - 64.5% students are eating standard food.
# - 35.5% students gets free/reduced food.

# ### Test preparation course

# In[ ]:


sns.catplot(y="test preparation course",  kind="count", height=6, aspect=2, data=students);


# In[ ]:


students['test preparation course'].value_counts(normalize=True) * 100


# - Most of the students have not completed the test preperation course.
# - Only 35.8% people have completed the test preparation course

# ## Influence of test preparation score on exam score

# ### math score

# In[ ]:


sns.catplot(x="test preparation course", y="math score", data=students, height=6, aspect=2, kind='swarm', hue='gender');
plt.grid()


# - All the students that completed the course have scored above 20.
# - 4 students who didnt completed the course have scored below 20
# - All the students who scored less than 20 are females

# In[ ]:


sns.catplot(x="test preparation course", y="math score", kind='violin', hue='gender', split='true', data=students, height=6, aspect=2);
plt.grid()


# - Average marks are high for students who have finished the test preparation course
# - Boys have better average performance than girls

# ### reading score

# In[ ]:


sns.catplot(x="test preparation course", y="reading score", data=students, height=6, aspect=2, kind='swarm', hue='gender');
plt.grid()


# - Here we can see an interesing difference between students who finished the course and those who didnt finish.
# - Almost all the students who finished the course has scored more than 40 for the reading test
# - Among the students who scored full marks, 80% are girls.

# In[ ]:


sns.catplot(x="test preparation course", y="reading score", kind='violin', hue='gender', split='true', data=students, height=6, aspect=2);
plt.grid()


# - Average marks are high for students who have finished the test preparation course

# ### writing score

# In[ ]:


sns.catplot(x="test preparation course", y="writing score", data=students, height=6, aspect=2, kind='swarm', hue='gender');
plt.grid()


# - For writing test also students who completed the course have scored more than 40% marks
# - Among the students who scored full marks, 93% are girls

# In[ ]:


sns.catplot(x="test preparation course", y="writing score", kind='violin', hue='gender', split='true', data=students, height=6, aspect=2);
plt.grid()


# - Average marks are high for students who have finished the test preparation course

# ## Influence of parents education on score

# ### Maths score

# In[ ]:


sns.catplot(y="parental level of education", x="math score", data=students, height=6, aspect=2, kind='swarm', hue='gender');
plt.grid()


# - Childrens of parents having masters degree have shown good performance for maths but havent scored full marks
# - Childrens of parents having bachelors's/associate's degree also shown good performance. However few students have less than 40 marks 
# - In the group of students who have scored less than 40%, a good percentage comes from parents having some college, some high school and high school.
# - Childrens whose parents have high school or some high school education have scored less that 20% as well.
# - We can see students scored more than 80% in all groups irrespective of their parents education
# - One student have score no marks whose parent has some high school education
# - Parents of the students who have scored full marks have bachelors, associate or come college degree.
# - Female students tend to score less marks in all category

# In[ ]:


sns.catplot(y="parental level of education", x="math score", kind='violin', hue='gender', split='true', data=students, height=6, aspect=2);
plt.grid()


# - Average marks are highest for students whose parents have master's degree and is lowest for students whose parents have some high school education

# ### reading score

# In[ ]:


# fig, ax = plt.subplots()
# fig.set_size_inches(11.7, 8.27)
sns.catplot(y="parental level of education", x="reading score", data=students, height=5, aspect=2, kind='swarm', hue='gender');
plt.grid()


# - Childrens of parents having masters and bacholers degree have shown good performance for reading all of them have scored above 40%
# - Childrens of parents having associate's degree also shown good performance. However few students have less than 40% marks 
# - In the group of students who have scored less than 40%, a good percentage comes from parents having some college, some high school and high school.
# - Childrens whose parents have high school or some high school education have scored less that 20% as well.
# - We can see students scored more than 80% in all groups irrespective of their parents education
# - Students from all group have scored full marks.
# - The parents of the student who have scored lowest marks have some high school education
# - Most of the students who have scored full marks are females

# In[ ]:


sns.catplot(y="parental level of education", x="reading score", kind='violin', hue='gender', split='true', data=students, height=5, aspect=2);
plt.grid()


# - Average marks are highest for students whose parents have master's degree and is lowest for students whose parents have some high school/ high school education

# ### writing score

# In[ ]:


sns.catplot(y="parental level of education", x="writing score", data=students, kind='swarm', height=6, aspect=2, hue='gender');
plt.grid()


# - Childrens of parents having masters degree have shown good performance for maths
# - Childrens of parents having bachelors's/associate's degree also shown good performance. However few students have less than 40 marks 
# - In the group of students who have scored less than 40%, a good percentage comes from parents having some college, some high school and high school.
# - Childrens whose parents have high school or some high school education have scored less that 20% as well.
# - We can see students scored more than 80% in all groups irrespective of their parents education
# - Most of the students who have scored full marks are females

# In[ ]:


sns.catplot(y="parental level of education", x="writing score", kind='violin', hue='gender', split='true', data=students, height=6, aspect=2);
plt.grid()


# - Average marks are highest for students whose parents have master's degree and is lowest for students whose parents have some high school/ high school education

# ## Relationship of different scores

# In[ ]:


sns.pairplot(students, hue="gender", palette="Set2", diag_kind="kde", height=5)


# In[ ]:




