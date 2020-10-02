#!/usr/bin/env python
# coding: utf-8

# In this kernel I used Pie chart, Bar chart, Scatter plot, Regression Plot(regplot), heatmap(for task) and some pandas manipulations. 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns # visualization

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# some configs for seaborn
sns.set(style="whitegrid")
sns.set_palette("husl")


# -Let's start with importing dataset and changing some column names for more simple usage.

# In[ ]:


students=pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")
students.rename(columns={"race/ethnicity": "ethnic",
                         "parental level of education": "parents_education",
                         "test preparation course":"test_prep_score",
                         "math score":"math",
                         "reading score":"reading",
                         "writing score":"writing"}, inplace=True)


# In[ ]:


students.info()


# In[ ]:


# is there Na observations?
students.isna().sum()


# In[ ]:


students.shape


# In[ ]:


students.columns


# In[ ]:


students.describe()


# -I want to look parent's education level and learn how many of them in this dataset:

# In[ ]:


students["parents_education"].value_counts()


# In[ ]:


uniq_degree=students["parents_education"].unique()


# In[ ]:


plt.figure(dpi=100)
plt.pie(students["parents_education"].value_counts(),labels=uniq_degree,autopct="%1.2f%%")
plt.title("Parental level of education")


# -What's the number of Females and Males in this dataset?

# In[ ]:


students["gender"].value_counts()


# In[ ]:


plt.figure(dpi=100)
plt.pie(students["gender"].value_counts(),labels=["Female","Male"],autopct="%1.2f%%")
plt.title("Students gender ratio")


# -What about ethnicity?

# In[ ]:


students["ethnic"].unique()
plt.figure(dpi=100)
plt.pie(students["ethnic"].value_counts(),labels=["Group A","Group B","Group C","Group D","Group E"],autopct="%1.2f%%")
plt.title("Students gender ratio")


# # MATH

# In[ ]:


ax=sns.barplot(y="math",x="gender",data=students)
ax.set(xlabel="Gender(F/M)",ylabel="Math Score Mean",title="Mean of Math Scores by Gender")


# In[ ]:


print("Mean of math score (female)  = " + str(students[students["gender"]=="female"]["math"].mean()))
print("Mean of math score (male)    = " + str(students[students["gender"]=="male"]["math"].mean()))


# In[ ]:


ax=sns.barplot(x="ethnic",y="math",data=students)
ax.set(xlabel="Ethnicity",ylabel="Math Score Mean",title="Mean of Math Scores by Ethnicity")


# In[ ]:


print(students.groupby("ethnic")["math"].mean())


# -I want to compare math and reading scores with scatter plot. I think scatter plot is the most efficient when looking a relation between two variables.

# In[ ]:


ax=sns.scatterplot(x="math",y="reading",data=students)
ax.set(xlabel="Math Scores",ylabel="Reading Scores",title="Math and Reading Scores")


# In[ ]:


ax = sns.regplot(x="math", y="reading", data=students,color="g",line_kws={'color':'blue'})
ax.set(xlabel="Math Scores",ylabel="Reading Scores",title="Math and Reading Scores with Regression Line")


# > As you can see from the regression plot: There is a relation between Math and Reading scores. 

# -I want to compare math and writing scores with scatter plot

# In[ ]:


ax=sns.scatterplot(x="math",y="writing",data=students)
ax.set(xlabel="Math Scores",ylabel="Writing Scores",title="Math and Writing Scores")


# In[ ]:


ax = sns.regplot(x="math", y="writing", data=students,color="g",line_kws={'color':'blue'})
ax.set(xlabel="Math Scores",ylabel="Reading Scores",title="Math and Writing Scores with Regression Line")


# > We can see that there is a relation between math and reading scores.

# # READING

# In[ ]:


ax=sns.barplot(y="reading",x="gender",data=students)
ax.set(xlabel="Gender(F/M)",ylabel="Reding Score Mean",title="Mean of Reading Scores by Gender")


# In[ ]:


print("Mean of reading score (female)  = " + str(students[students["gender"]=="female"]["reading"].mean()))
print("Mean of reading score (male)    = " + str(students[students["gender"]=="male"]["reading"].mean()))


# In[ ]:


ax=sns.barplot(x="ethnic",y="reading",data=students)
ax.set(xlabel="Ethnicity",ylabel="Reading Score Mean",title="Mean of Reading Scores by Ethnicity")


# In[ ]:


print(students.groupby("ethnic")["reading"].mean())


# -I want to compare reading and math scores with scatter plot

# In[ ]:


ax=sns.scatterplot(x="reading",y="math",data=students,color="r")
ax.set(xlabel="Reading Scores",ylabel="Math Scores",title="Reading and Math Scores")


# -I want to compare reading and writing scores with scatter plot

# In[ ]:


ax=sns.scatterplot(x="reading",y="writing",data=students,color="r")
ax.set(xlabel="Reading Scores",ylabel="Writing Scores",title="Reading and Writing Scores")


# In[ ]:


ax = sns.regplot(x="reading", y="writing", data=students,color="darkblue",line_kws={'color':'red'})
ax.set(xlabel="Reading Scores",ylabel="Writing Scores",title="Reading and Writing Scores with Regression Line")


#      We can see there is a relation between Reading and Writing scores. We can see the relation is possitive but not the actual degree.
#      We should calculate the correlation to learn the variables relation strength

# # WRITING

# In[ ]:


ax=sns.barplot(y="writing",x="gender",data=students)
ax.set(xlabel="Gender(F/M)",ylabel="Writing Score Mean",title="Mean of Writing Scores by Gender")


# In[ ]:


print("Mean of writing score (female)  = " + str(students[students["gender"]=="female"]["writing"].mean()))
print("Mean of writing score (male)    = " + str(students[students["gender"]=="male"]["writing"].mean()))


# In[ ]:


ax=sns.barplot(x="ethnic",y="math",data=students)
ax.set(xlabel="Ethnicity",ylabel="Writing Score Mean",title="Mean of Writing Scores by Ethnicity")


# In[ ]:


print(students.groupby("ethnic")["writing"].mean())


# -I want to compare writing and math scores with scatter plot

# In[ ]:


ax=sns.scatterplot(x="writing",y="math",data=students,color="b")
ax.set(xlabel="Writing Scores",ylabel="Math Scores",title="Writing and Math Scores")


# -I want to compare writing and reading scores with scatter plot

# In[ ]:


ax=sns.scatterplot(x="writing",y="reading",data=students,color="b")
ax.set(xlabel="Writing Scores",ylabel="Writing Scores",title="Writing and Reading Scores")


# # TASK

# We can calculate the correlation with .corr method

# In[ ]:


correlations=students.corr()
print(correlations)


# > The correlations table says:
#  1. Math and Reading scores have a positive relation. And the relation's degree is 0.817 which means %81.7
#  2. Math and Writing scores have a positive relation. And the relation's degree is 0.802 which means %80.2
#  3. Reading and Writing scores have a positive strong relation. And the relation's degree is 0.954 which means %95.4
# 

# *To see more clearly, we can use heatmap of correlation.*
# 

# In[ ]:


plt.figure(dpi=100)
plt.title('Correlation Analysis of Math/Reading/Writing Scores')
sns.heatmap(correlations,annot=True,lw=1,linecolor='black',cmap='terrain')
plt.yticks(rotation=0)


# > Actually, we can see from regression plots that the correlation results are possitive. But calculating makes it better for see what the actual degree is
# 
