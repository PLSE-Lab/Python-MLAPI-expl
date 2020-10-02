#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/coursera-course-dataset/coursea_data.csv')


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


# checking for null values
df.isnull().sum()


# Great No null values to work with!

# In[ ]:


df.head()


# **VISUALIZATION**

# ***Let's see top 10 organizations providing courses***

# In[ ]:


plt.figure(figsize=(16,7))
sns.countplot(x = 'course_organization' , data=df , order=df['course_organization'].value_counts().iloc[:10].index)
plt.xticks(rotation=70)
plt.show()


# University of Pennsylvania has the higest number of courses

# In[ ]:


plt.figure(figsize = (8,8))
df.course_Certificate_type.value_counts().plot(kind='pie',shadow=True, explode=(0,0, 0), startangle=90,autopct='%1.1f%%')
plt.title('Status')
plt.show()


# **major share is of courses but you need to see that a specialization contains n number of courses**

# In[ ]:


plt.figure(figsize=(16,7))
sns.countplot(x = 'course_organization' , data=df , order=df['course_organization'].value_counts().iloc[:10].index , hue='course_Certificate_type')
plt.xticks(rotation=70)
plt.show()


# ***Let's see top 20 courses and specializations as per number of enrolled students ***

# In[ ]:


# mapping m and k as millions and thousands
mp = {'k':' * 10**3', 'k':' * 10**6'}
df.course_students_enrolled = (df.course_students_enrolled.replace(mp.keys(), mp.values(), regex=True).str.replace(r'[^\d\.\*]+',''))
df['course_students_enrolled'] = df['course_students_enrolled'].astype('float')


# In[ ]:


df_c = df[df['course_Certificate_type'] == 'COURSE']
df_s = df[df['course_Certificate_type'] == 'SPECIALIZATION']

ass=df_c.sort_values('course_students_enrolled' , ascending=False)
ass2 = df_s.sort_values('course_students_enrolled' , ascending=False)
plt.figure(figsize = (16,16))
plt.subplot(2,1,1)
sns.barplot(x= 'course_students_enrolled' , y = 'course_title' , data = ass[:20] )
plt.xticks(rotation=50)
plt.title('COURSES')
plt.tight_layout(pad = 0.6)

plt.subplot(2,1,2)
sns.barplot(x= 'course_students_enrolled' , y = 'course_title' , data = ass2[:20] )
plt.ylabel('specialization_title')
plt.xticks(rotation=50)
plt.title('SPECIALIZATION')
plt.tight_layout(pad = 0.6)


# **Observation**
# * English for Career development is on higest demand in courses
# * Data Science is the most in demand specialization and in top 20, 5 are data related specialization. That's quite RIGHT!

# ***Let's see top 20 instutues that provide courses and specialization***

# In[ ]:


plt.figure(figsize = (16,16))
plt.subplot(2,1,1)
sns.countplot(y= 'course_organization'  , data = df_c , order = df_c['course_organization'].value_counts().iloc[:20].index  )
plt.xticks(rotation=0)
plt.title('COURSES')
plt.ylabel('University')
plt.tight_layout(pad = 0.6)

plt.subplot(2,1,2)
sns.countplot(y= 'course_organization'  , data = df_s , order = df_s['course_organization'].value_counts().iloc[:20].index )
plt.ylabel('University')
plt.xticks(rotation=0)
plt.title('SPECIALIZATION')
plt.tight_layout(pad = 0.6)


# **Observations**
# * University of Pennsylvania and University of Michigan top the chart in courses
# * University of Pennsylvania and Google cloud are at power in number of spezializations offered

# **Distribution of difficulty among Specializations and Courses**

# In[ ]:


plt.figure(figsize = (8,8))
plt.subplot(2,1,1)
df_c.course_difficulty.value_counts().plot(kind='pie',shadow=True, explode=(0,0, 0,0.1), startangle=90,autopct='%1.1f%%')
plt.title('Difficulty in Courses')
plt.ylabel("")
plt.tight_layout(pad = 10)

plt.subplot(2,1,2)
df_s.course_difficulty.value_counts().plot(kind='pie',shadow=True, explode=(0,0.10, 0), startangle=90,autopct='%1.1f%%')
plt.title('Difficulty in Specialization')
plt.ylabel("")
plt.tight_layout(pad = 0.8)


# **Distribution of Reviews in Courses and Specializations**

# In[ ]:


plt.figure(figsize = (8,8))
plt.subplot(2,1,1)
sns.distplot(df_c['course_rating'] , bins = 10)
plt.title('Distribution of rating in Courses')
plt.tight_layout(pad=3.0)

plt.subplot(2,1,2)
sns.distplot(df_s['course_rating'] , bins = 10)
plt.title('Distribution of rating in Specialization')


# *Both are bell shaped curve indication normal distribution of rating in both courses and specialization*

# **IF YOU LIKE THE NOTEBOOK , PLEASE UPVOTE.**

# 

# In[ ]:




