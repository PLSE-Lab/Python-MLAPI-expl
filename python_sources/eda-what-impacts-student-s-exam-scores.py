#!/usr/bin/env python
# coding: utf-8

# Lets perform **Exploratory Data Analysis (EDA)** to find out what impacts the **Student's Exam Scores**

# In[ ]:


#Import python libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')


# In[ ]:


#Import Students Performance Data
df=pd.read_csv('../input/StudentsPerformance.csv')
df.head()


# In[ ]:


#Calculate Total Score and insert it as a column
df['Total score']=df['math score']+df['reading score']+df['writing score']
df.head()


# In[ ]:


df.info()


# In[ ]:


sns.pairplot(data=df,hue='gender')


# **Impact of Gender on the Scores**
# * Auto-correlation
# * * Math score : Males performed better than females, however the topper is female (female peak is higher than male)
# * * Reading score : Females performed better then males
# * * Writing score : Females performed better then males
# * Cross-correlation
# * * Reading vs Maths : Males scored better in Maths (Males are towards right)
# * * Writing vs Maths : Males scored better in Maths (Males are towards right)
# * * Reading vs Writing : The difference is not distinct

# In[ ]:


sns.pairplot(data=df,hue='lunch')


# **Impact of Lunch** :  It can be seen from above plot that the students who got **standard** lunch scored more as compared to the students who got **free/reduced** lunch

# In[ ]:


sns.pairplot(data=df,hue='race/ethnicity')


# **Impact of Race/Ethnicity** : It can be seen from above plots that "Group D" has performed better as compared to other Groups

# In[ ]:


sns.pairplot(data=df,hue='parental level of education')


# **Impact of Parental Education** :
# It can be seen from above plot that the students who parents have completed "Masters" have got less marks

# In[ ]:


sns.pairplot(data=df,hue='test preparation course')


# **Impact of Completion of Test Preparation Course** :
# It can be seen from above plot that the students who have **Completed the test preparation course** have score significantly better as compard to the students who have **NOT completed the test preparation couse**

# In[ ]:


toppers = df[(df['math score'] > 90) & (df['reading score'] > 90) & (df['writing score']>90)].sort_values(by=['Total score'],ascending=False)
toppers.head()


# The first two toppers are either **geniuses** or they did some **malpractice**  as their test preparation course was ** none**.

# In[ ]:


fig,ax=plt.subplots()
sns.countplot(x='parental level of education',data=df)
plt.tight_layout()
fig.autofmt_xdate()


# Thw above plot shows most of the parents went to some college or had associate's degree and there are very less people who had higher studies.

# In[ ]:


fig,ax=plt.subplots()
sns.barplot(x=df['parental level of education'],y='Total score',data=df)
fig.autofmt_xdate()


# From the above plot its clear that ** if the parental education is better their children tend to score better in all areas **(math, reading, writing).
