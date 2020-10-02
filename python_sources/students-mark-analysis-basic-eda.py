#!/usr/bin/env python
# coding: utf-8

# **IMPORTING LIBRARIES**

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data=pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')


# In[ ]:


data.head()


# **Let us rename the columns for better understanding.**

# In[ ]:


data.columns = ['Gender', 'Race', 'Parent Degree', 'Lunch', 'Course', 'Math Score', 'Reading Score', 'Writing Score'] 


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# **Let's create a new column named average score to calculate percentages.**

# In[ ]:


data['Total Score']=(data['Math Score'] + data['Reading Score'] + data['Writing Score'])/3


# In[ ]:


data.head()


# In[ ]:


data.groupby(['Gender']).mean()


# **We can see that the female candidates scored higher than the male candidates.**

# **But the male candidates scored better for maths compared to female.**

# In[ ]:


sns.barplot(x='Total Score',y='Gender',data=data).set_title('GENDER VS TOTAL SCORE')


# In[ ]:


sns.barplot(x='Math Score',y='Gender',data=data).set_title('GENDER VS MATH SCORE')


# In[ ]:


data['Math Score'].mode()


# In[ ]:


data['Math Score'].mean()


# In[ ]:


plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
sns.boxplot(x = 'Gender', y = 'Math Score', data = data)

plt.subplot(1,3,2)
sns.boxplot(x = 'Gender', y = 'Reading Score', data = data)

plt.subplot(1,3,3)
sns.boxplot(x = 'Gender', y = 'Writing Score', data = data)


# **This boxplot clearly shows about the marks scored by male and female in each section.**

# In[ ]:


sns.barplot(x='Lunch',y='Total Score',data=data)


# **So this clearly shows that the total score was increased when the student had a standard lunch compared to a 
# reduced lunch.**

# In[ ]:


plt.figure(figsize=(13,4))

plt.subplot(1,3,1)
sns.barplot(x = "Parent Degree" , y="Reading Score" , data=data)
plt.xticks(rotation = 90)
plt.title("Reading Scores")

plt.subplot(1,3,2)
sns.barplot(x = "Parent Degree" , y="Writing Score" , data=data)
plt.xticks(rotation=90)
plt.title("Writing Scores")

plt.subplot(1,3,3)
sns.barplot(x = "Parent Degree" , y="Math Score" , data=data)
plt.xticks(rotation=90)
plt.title("Math Scores")

plt.tight_layout()
plt.show()


# **Student's whose parents have master's degree have the highest score in all categories.**

# **Student's whose parents have bachelor's degree or associate degree performed reasonably well in all categories.**

# In[ ]:


sns.barplot(x='Race',y='Total Score',data=data)


# **From this,we can infer that Group E scored the highest scores compared to other groups.**

# In[ ]:


df1=data.sort_values('Total Score',ascending=False)


# In[ ]:


df1.head()


# In[ ]:


fig,ax=plt.subplots()
sns.countplot(x='Parent Degree',data=data,palette='spring')
plt.tight_layout()
fig.autofmt_xdate()


# **The above plot shows most of the parents went to some college or had associate's degree and there are very less people 
# who had higher studies.**

# In[ ]:




