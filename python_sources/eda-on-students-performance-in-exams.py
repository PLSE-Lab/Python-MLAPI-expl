#!/usr/bin/env python
# coding: utf-8

# ### Exploratory Data Analysis

# ![PIC](https://miro.medium.com/max/1500/0*hUeopQVbu6SL50Oc.png)

# #### Why do we need to perform Exploratory Data Analysis?
# 
# 1. To Maximise the insight into dataset.
# 2. To understand the connection between the variables and to uncover the underlying structure
# 3. To extract the import Variables
# 4. To detect anomalies
# 5. To test the underlying assumptions.

# #### Objective of this kernel:
# 
# * To understand the how the student's performance (test scores) is affected by the other variables (Gender, Ethnicity, Parental level of education, Lunch, Test preparation course)
# 

# Lets import the required libraries

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


# Reading the data set

# In[ ]:


df=pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')


# In[ ]:


df.info()


# Here, you can see all the column names, total values and type of the values.

# #### We have 2 types of variables.
# 
# 1. Numerical variables : which contains number as values
# 2. Categorical variables : which contains descriptions of groups or things.
# 
# In this Data set,
# 
# Numerical Variables are  Math score, Reading score and Writing score.
# 
# Categorical Variables are Gender, Race/ethnicity, Parental level of education, Lunch and Test preparation course.[](http://)

# In[ ]:


df.describe()


# You can see the descriptive statistics of numerical variables such as total count, mean, standard deviation, minimum and maximum values and three quantiles of the data (25%,50%,75%).

# In[ ]:


df.shape


# It shows the number of rows and columns.

# In[ ]:


df.isnull().sum() #checks if there are any missing values


# So there are no missing values in the data

# ### Lets start with plotting graphs

# We want to analyse the scores of the students.
# 
# * So lets see the distribution of Math, Reading and Writting scores

# In[ ]:


plt.rcParams['figure.figsize'] = (20, 10)
sns.countplot(df['math score'], palette = 'dark')
plt.title('Math Score',fontsize = 20)
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (20, 10)
sns.countplot(df['reading score'], palette = 'Set3')
plt.title('Reading Score',fontsize = 20)
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (20, 10)
sns.countplot(df['writing score'], palette = 'prism')
plt.title('Writing Score',fontsize = 20)
plt.show()


# In[ ]:


plt.figure(figsize=(15,5))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                      wspace=0.5, hspace=0.2)
plt.subplot(141)
plt.title('Math Scores')
sns.violinplot(y='math score',data=df,color='m',linewidth=2)
plt.subplot(142)
plt.title('Reading Scores')
sns.violinplot(y='reading score',data=df,color='g',linewidth=2)
plt.subplot(143)
plt.title('Writing Scores')
sns.violinplot(y='writing score',data=df,color='r',linewidth=2)
plt.show()


# From the above plots, we can see that the maximum number of students have scored 60-80 in all three subjects i.e., math, reading and writing.

# Lets see the proportion of the remaining variables

# In[ ]:


plt.figure(figsize=(20,10))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                      wspace=0.5, hspace=0.2)
plt.subplot(141)
plt.title('Gender',fontsize = 20)
df['gender'].value_counts().plot.pie(autopct="%1.1f%%")

plt.subplot(142)
plt.title('Ethinicity',fontsize = 20)
df['race/ethnicity'].value_counts().plot.pie(autopct="%1.1f%%")

plt.subplot(143)
plt.title('Lunch',fontsize = 20)
df['lunch'].value_counts().plot.pie(autopct="%1.1f%%")

plt.subplot(144)
plt.title('Parentel level of Education',fontsize = 20)
df['parental level of education'].value_counts().plot.pie(autopct="%1.1f%%")
plt.show()


# #### Observations:
# * The proportion of male and female are almost same
# * Highest number of students belong to Group C ethinicity followed by Group D
# * Highest proportion of the students have standard lunch
# * Highest proportion of parentel level of Education is 'Some college', 'associate's degreee' and 'high school'

# Lets look at the scores of male and female students seperately in each subject.

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                      wspace=0.5, hspace=0.2)
plt.subplot(131)
plt.title('Math Scores')
sns.barplot(x="gender", y="math score", data=df)
plt.subplot(132)
plt.title('Reading Scores')
sns.barplot(x="gender", y="reading score", data=df)
plt.subplot(133)
plt.title('Writing Scores')
sns.barplot(x="gender", y="writing score", data=df)
plt.show()


# We can see that male students scored higher in Maths where as female students scored higher in Reading and writing

# Lets look at the scores who completed Test preperation course

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                      wspace=0.5, hspace=0.2)
plt.subplot(131)
plt.title('Math Scores')
sns.barplot(hue="gender", y="math score", x="test preparation course", data=df)
plt.subplot(132)
plt.title('Reading Scores')
sns.barplot(hue="gender", y="reading score", x="test preparation course", data=df)
plt.subplot(133)
plt.title('Writing Scores')
sns.barplot(hue="gender", y="writing score", x="test preparation course", data=df)
plt.show()


# So the students (male and female) who completed the test preparation course scored higher in all three subjects.

# Lets look at the scores of the students of different group who completed test preperation course.

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                      wspace=0.5, hspace=0.2)
plt.subplot(131)
plt.title('Math Scores')
sns.barplot(x="race/ethnicity", y="math score", hue="test preparation course", data=df)
plt.subplot(132)
plt.title('Reading Scores')
sns.barplot(hue="test preparation course", y="reading score", x="race/ethnicity", data=df)
plt.subplot(133)
plt.title('Writing Scores')
sns.barplot(hue="test preparation course", y="writing score", x= 'race/ethnicity',data=df)

plt.show()


# Highest number of Students who belongs to Group E has completed the test preperation course in Math and Reading and scored highest. 
# 
# Highest number of Students who belongs to Group D and E has completed the test preperation course in Writing and scored highest. 

# Now lets analyze the relation between Test preperation course and other variables

# In[ ]:


plt.figure(figsize=(30,15))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                      wspace=0.5, hspace=0.2)
plt.subplot(251)
plt.title('Test Preparation course Vs Gender',fontsize = 15)
sns.countplot(hue="test preparation course", x="gender", data=df)

plt.subplot(254)
plt.title('Test Preparation course Vs Parental Level Of Education',fontsize = 15)
sns.countplot(hue="test preparation course", y="parental level of education", data=df)

plt.subplot(253)
plt.title('Test Preparation course Vs Lunch',fontsize = 15)
sns.countplot(hue="test preparation course", x="lunch", data=df)

plt.subplot(252)
plt.title('Test Preparation course Vs Ethnicity',fontsize = 15)
sns.countplot(hue="test preparation course", y="race/ethnicity", data=df)

plt.show()


# #### Observations:
# * Most of the students have not completed the test preparation course.
# * Highest number Students who belong to group C ethinicity have completed the test preparation course.
# * Standard lunch students have completed the test preparation course
# * Students whos parental level of education is 'some college, 'associate's degree', and high school have completed the test preparation course.

# We can also say that the students who belongs to Group E ethincity has scored more marks in all three subjectes even though they have not completed the test preparation course.

# Now, lets see the relation between the remaining variables

# In[ ]:


plt.title('Gender Vs Ethnicity',fontsize = 20)
sns.countplot(x="gender", hue="race/ethnicity", data=df)
plt.show()


# In[ ]:


pr=pd.crosstab(df['race/ethnicity'],df['parental level of education'],normalize=0)

pr.plot.bar(stacked=True)
plt.title('Ethinicity Vs Parental Level of Education',fontsize = 20)
plt.show()


# In[ ]:


plt.figure(figsize=(40,10))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                      wspace=0.5, hspace=0.2)
plt.subplot(251)
plt.title('Parental education and Gender',fontsize=15)
sns.countplot(hue="parental level of education", x="gender", data=df)
plt.subplot(252)
plt.title('Parental education and Lunch',fontsize=15)
sns.countplot(hue="parental level of education", x="lunch", data=df)

plt.show()


# In[ ]:


plt.figure(figsize=(40,10))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                      wspace=0.5, hspace=0.2)
plt.subplot(251)
plt.title('Lunch and Gender',fontsize=15)
sns.countplot(x="lunch", hue="gender", data=df)
plt.subplot(252)
plt.title('Ethinicity and Lunch',fontsize=15)
sns.countplot(x="race/ethnicity", hue="lunch", data=df)
plt.show()


# To analyse the data in more deeper way, lets few new columns: Total marks, Percentage and Grades.

# In[ ]:


df['total marks']=df['math score']+df['reading score']+df['writing score']


# In[ ]:


df['percentage']=df['total marks']/300*100


# ##### Lets assign grades.
# 
# Criteria of the grades are as follows:
# 
# * 85-100 : Grade A
# * 70-84 : Grade B
# * 55-69 : Grade C
# * 35-54 : Grade D
# * 0-35 : Grade E

# In[ ]:


#Assigning the grades

def determine_grade(scores):
    if scores >= 85 and scores <= 100:
        return 'Grade A'
    elif scores >= 70 and scores < 85:
        return 'Grade B'
    elif scores >= 55 and scores < 70:
        return 'Grade C'
    elif scores >= 35 and scores < 55:
        return 'Grade D'
    elif scores >= 0 and scores < 35:
        return 'Grade E'
    
df['grades']=df['percentage'].apply(determine_grade)


# In[ ]:


df.info()


# Now, total marks, percentage and grades columns are created.

# In[ ]:


df['grades'].value_counts().plot.pie(autopct="%1.1f%%")
plt.show()


# Most of the students got Grade B and Grade C.

# In[ ]:


plt.figure(figsize=(30,10))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                      wspace=0.5, hspace=0.2)
plt.subplot(251)
plt.title('Grades and Gender')
sns.countplot(hue="gender", x="grades", data=df)

plt.subplot(252)
plt.title('Grades and Lunch')
sns.countplot(hue="lunch", x="grades", data=df)

plt.subplot(253)
plt.title('Grades and Test preperation Course')
sns.countplot(hue="test preparation course", x="grades", data=df)

plt.show()


# In[ ]:


plt.title('Grades and Parental level of Education',fontsize=20)
sns.countplot(x="parental level of education", hue="grades", data=df)
plt.show()


# In[ ]:


plt.title('Grades and Ethinicity',fontsize=20)
sns.countplot(x="race/ethnicity", hue="grades", data=df)


gr=pd.crosstab(df['grades'],df['race/ethnicity'],normalize=0) #normalized values 
gr.plot.bar(stacked=True)
plt.title('Grades and Ethinicity',fontsize=20)
plt.show()

