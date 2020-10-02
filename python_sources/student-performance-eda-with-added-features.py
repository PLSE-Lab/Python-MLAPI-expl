#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')


# In[ ]:


df = pd.read_csv('../input/StudentsPerformance.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


# Gender distribution
plt.figure(figsize=(8,7))
plt.title('Gender Distribution')
sns.countplot(df['gender'], palette='rainbow')


# In[ ]:


# race/ethinicity distribution
plt.figure(figsize=(8,7))
plt.title('Race/Ethinicity distribution')
sns.countplot(df['race/ethnicity'], palette='rainbow', hue=df['gender'])


# In[ ]:


#parental level of education
plt.figure(figsize=(12,6))
plt.title('Parental Level of Education')
sns.countplot(df['parental level of education'], palette='rainbow', hue=df['gender'])


# 

# In[ ]:


# test prep course
plt.figure(figsize=(8,7))
plt.title('Test Preparation course')
sns.countplot(df['test preparation course'], palette='rainbow', hue=df['gender'])


# In[ ]:


# students with highest score in math
math_df = df[df['math score']==df['math score'].max()]
math_df


# In[ ]:


# number of student with math score = 100
len(math_df)


# In[ ]:


# gender distribution of students with high math score
plt.figure(figsize=(8,7))
plt.title('Gender Distribution with math score = 100')
sns.countplot(math_df['gender'], palette='rainbow')


# In[ ]:


# race/ethinicity distribution of students with high math score
plt.figure(figsize=(8,7))
plt.title('race/ethinicity Distribution with math score = 100')
sns.countplot(math_df['race/ethnicity'], palette='rainbow')


# 

# In[ ]:


# parent level education distribution of students with high math score
plt.figure(figsize=(8,7))
plt.title('parent level education Distribution with math score = 100')
sns.countplot(math_df['parental level of education'], palette='rainbow')


# 
#     Total number of students with high math score = 7
#     4 Male 3 Female
#     group E students are good at math
#     students whose parental level of education is 'some college' have higher math score
# 

# In[ ]:


# students with highest score in reading and writing
rw_df = df[(df['reading score']==df['reading score'].max()) & (df['writing score']==df['writing score'].max())]
rw_df


# In[ ]:


# number of students who have high scores in reading and writing
len(rw_df)


# In[ ]:


# gender distribution of students with high reading and writing score
plt.figure(figsize=(8,7))
plt.title('Gender Distribution with reading and writing score = 100')
sns.countplot(rw_df['gender'], palette='rainbow')


# Female students have high scores in reading and writing

# In[ ]:


# race distribution of students with high reading and writing score
plt.figure(figsize=(8,7))
plt.title('race Distribution with reading and writing score = 100')
sns.countplot(rw_df['race/ethnicity'], palette='rainbow')


# Group D and Group E students have high scores in reading and writing

# In[ ]:


# parental level of education of students with high reading and writing score
plt.figure(figsize=(8,7))
plt.title('Parental level of education Distribution with reading and writing score = 100')
sns.countplot(rw_df['parental level of education'], palette='rainbow')


# Students with parental level of education as 'bachelor's degree' have high scores in reading and writing

# # Some Data Imputations

# Creating a percentage data column

# In[ ]:


def calculate_percentage(math, read, write):
    total = math + read + write
    return round((total/300)*100)


# In[ ]:


df['percentage']=df[['math score','reading score', 'writing score']].apply(lambda row : calculate_percentage(row['math score'],row['reading score'], row['writing score']), axis=1)
df.head(3)


# In[ ]:


df['percentage'].describe()


# # Grading the students

# create a grading system in which student are assigned particular grades as follows: 
# < 39 -> F (Failed)
# 40-50 -> E 
# 51-60 -> D
# 61-70 -> C
# 71-80 -> B
# 81-90 -> A
# 91-100 -> A+

# In[ ]:


def define_grades(perc):
    if perc <= 39:
        return 'F'
    elif perc >= 40 and perc <= 50:
        return 'E'
    elif perc >= 51 and perc <= 60:
        return 'D'
    elif perc >= 61 and perc <= 70:
        return 'C'
    elif perc >= 71 and perc <= 80:
        return 'B'
    elif perc >= 81 and perc <= 90:
        return 'A'
    elif perc >= 91 and perc <= 100:
        return 'A+'
    else:
        return ''


# In[ ]:


df['grades'] = df['percentage'].apply(lambda x : define_grades(x))
df.head()


# In[ ]:


plt.figure(figsize=(8,7))
plt.title('Grades of students')
sns.countplot(df['grades'], palette='Set1')


# In[ ]:


plt.figure(figsize=(8,7))
plt.title('Grades of students with gender distribution')
sns.countplot(df['grades'], hue=df['gender'], palette='Set1')


# In[ ]:


plt.figure(figsize=(12,7))
plt.title('Grades of students with race distribution')
sns.countplot(df['grades'], hue=df['race/ethnicity'], palette='Set1')


# In[ ]:


plt.figure(figsize=(15,7))
plt.title('Grades of students with parental lvl of education distribution')
sns.countplot(df['grades'], hue=df['parental level of education'], palette='Set1')


# In[ ]:


plt.figure(figsize=(15,7))
plt.title('Grades of students with test preparation course distribution')
sns.countplot(df['grades'], hue=df['test preparation course'], palette='Set1')


# Students who have completed the test prep course have good grades

# In[ ]:




