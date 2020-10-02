#!/usr/bin/env python
# coding: utf-8

# ## Aim -
# 
# ### To understand the influence of various factors like gender, parents' qualification, social and economic situation etc.
# 

# In[ ]:


#importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


# let (to be used later)
passmark=40


# In[ ]:


## Reading the dataset
df=pd.read_csv('../input/StudentsPerformance.csv')


# In[ ]:


## getting the top of dataset
df.head()


# In[ ]:


## getting the features of dataset
df.columns


# In[ ]:


## finding the mean,variance,count etc of data
df.describe()


# **Observation** -
# 
# **Mean Score**
# 1. Maths = 66
# 2. Reading = 69
# 3. Writing = 68
# 
# **Maximum Score**
# 1. 100 for all subjects.
# 
# **Minimum Score**
#  1. Maths=0
#  2. Reading = 17
#  3.  Writing = 10
# 

# In[ ]:


#DISRTRIBUTION OF MARKS

sns.distplot(df['math score'])


# **Observation**-
# 
# Maximum students have attain  marks between 60-80 in Maths.
# 

# In[ ]:


sns.distplot(df['reading score'])


# **Observation**- 
# 
# Maximum students have attain marks between 60-80 in Reading.

# In[ ]:


sns.distplot(df['writing score'])


# **Observation**
# 
# Maximum students have attain marks between 60-75 in Writing.

# ### Anlysis on the basis of gender

# In[ ]:


## Plot the bar plot to visualize the effect of gender on the scores of students
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
sns.barplot(x = 'gender', y = 'reading score', data = df)

plt.subplot(1,3,2)
sns.barplot(x = 'gender', y = 'writing score', data = df)

plt.subplot(1,3,3)
sns.barplot(x = 'gender', y = 'math score', data = df)

plt.tight_layout()


# **Observation** -
# * Girls have attain more marks in reading and writing than boys.
# *  Boys have attain more marks in maths than girls.
# 
# 

# **Bar Plot on the basis of Ethnicity**
# 

# In[ ]:


## Plotting bar plot to visualize the effect of race on marks of students

plt.figure(figsize=(15,8))
plt.subplot(1,3,1)
sns.barplot(x='race/ethnicity',y='math score',data=df)

plt.subplot(1,3,2)
sns.barplot(x='race/ethnicity',y='reading score',data=df)

plt.subplot(1,3,3)
sns.barplot (x='race/ethnicity',y='writing score',data=df)

plt.show()


# **Observation** -
# Group E students have performed well lin all subjects.
# Group A students have performed worst in all subjects.
# 
# 

# **Bar plot on the basis of  parental level of education** 
# 
# 

# In[ ]:


## plotting bar plot to visualize the effect of parents educational qualification on the marks of students
plt.figure(figsize=(13,5))
plt.subplot(1,3,1)
sns.barplot(x='parental level of education', y= 'math score', data = df )
plt.xticks(rotation = 90)

plt.subplot(1,3,2)
sns.barplot(x='parental level of education', y= 'writing score', data=df)
plt.xticks(rotation = 90)

plt.subplot(1,3,3)
sns.barplot(x='parental level of education',y='reading score', data=df)
plt.xticks(rotation = 90)


# ### Observation- Students whose parents have 'Master Degree' have performed better in all subjects.
# ### Students whose parents have 'high school' degree have performed worst in exam.

# **Bar plot on the basis of lunch**
# 

# In[ ]:


## plotting bar plot to visualize the effect of lunch on marks of students
plt.figure(figsize=(13,8))
plt.subplot(1,3,1)
sns.barplot(x='lunch', y= 'math score', data=df)

plt.subplot(1,3,2)
sns.barplot(x='lunch',y='reading score',data=df)

plt.subplot(1,3,3)
sns.barplot(x='lunch',y='writing score', data=df)


# ### Observation- Students who take standard lunch have performed well in all subjects.
# ### Students who take reduced or free lunch have not performed well.

# **Bar plot on the basis of test preparation course**
# 

# In[ ]:


### Plotting bar plot to visualize the effect of test prepration course
plt.figure(figsize=(13,8))
plt.subplot(1,3,1)
sns.barplot(x='test preparation course',y='math score' ,data=df)

plt.subplot(1,3,2)
sns.barplot(x='test preparation course',y='reading score',data=df)

plt.subplot(1,3,3)
sns.barplot(x='test preparation course',y='writing score', data=df)


# ### Observation- Students who have completed test preparation course have performed well in all subject compared to those who did not.
# 

# **Getting the Pass Status of Students**

# In[ ]:


## How many students have passed in Maths?
df['Math_PassStatus'] = np.where(df['math score']<passmark, 'F', 'P')
df.Math_PassStatus.value_counts()


# In[ ]:


## How many students have passed in Reading?
df['Reading_PassStatus'] = np.where(df['reading score']<passmark, 'F', 'P')
df.Reading_PassStatus.value_counts()


# In[ ]:


## How many Students have passed in Writing?
df['Writing_PassStatus'] = np.where(df['writing score']<passmark,'F','P')
df.Writing_PassStatus.value_counts()


# In[ ]:


## Students who have passed in all subjects.
df['OverAll_PassStatus'] = df.apply(lambda x : 'F' if x['Math_PassStatus'] == 'F' or 
                                    x['Reading_PassStatus'] == 'F' or x['Writing_PassStatus'] == 'F' else 'P', axis =1)

df.OverAll_PassStatus.value_counts()


# ### Observation- Maximum students have failed in Maths.
# ### The number of students who have passed in all subjects are 949.
# ### Number of students who failed are 51.

# In[ ]:


## Finding the percentage of marks
df['Total_Marks'] = df['math score']+df['writing score']+df['reading score']
df['Percentage']= df['Total_Marks']/3


# **Using Mean And Variance finding the effect of various features given in data.**

# In[ ]:


### Effect of gender on marks
gender_effect = df.groupby('gender')['Percentage']                         .aggregate(['mean', 'var', 'count'])                         .replace(np.NaN, 0)                         .sort_values(['mean', 'var'], ascending=[False, False])
gender_effect.head()


# #### Observation - Girls have perfomed well ,compared to boys.

# In[ ]:


### Effect of race of students
mean_marks_race = df.groupby('race/ethnicity')['Percentage']                     .aggregate(['mean','var', 'count'])                     .replace(np.NaN, 0)                     .sort_values(['mean','var'],ascending=[False,False])
mean_marks_race.head()


# ### Observation- Group E students have performed well in exam.
# ### Group A students have performed worst.

# In[ ]:


## Effect of parents qualification on marks
effect_parental_degree = df.groupby('parental level of education')['Percentage']                                .aggregate(['mean','var','count'])                                .replace(np.NaN,0)                                .sort_values(['mean','var'],ascending=[False,False]) 

effect_parental_degree.head()


# ### Observation - The students whose parents has Master's degree have performed well.
# ### The students whose parents has High school Degree have performed worst.

# In[ ]:


### Effect of type of lunch on marks 
effect_lunch = df.groupby('lunch')['Percentage']                      .aggregate(['mean','var','count'])                      .replace(np.NaN,0)                      .sort_values(['mean','var'],ascending=[False,False])
effect_lunch.head()


# ### Observation- The students who take standard lunch have performed well in exam.

# In[ ]:


### Effect of test preparation on marks
effect_testprep= df.groupby('test preparation course')['Percentage']                        .aggregate(['mean','var','count'])                        .replace(np.NaN,0)                        .sort_values(['mean','var'],ascending=[False,False])
effect_testprep.head()


# ### Observation - Students who have completed the test preparation course have performed well in the exam.

# In[ ]:




