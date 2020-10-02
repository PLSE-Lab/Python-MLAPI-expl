#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt 


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.info()


# ## Which factor influenced a candidate in getting placed?

# In[ ]:


data.status.value_counts()


# ## Gender

# In[ ]:


# Let's see the apperance of boys and girls in our dataset
data.gender.value_counts()


# In[ ]:


sns.set(style='whitegrid')
plt.figure(figsize=(11,9))
sns.countplot(x='status',hue='gender',data=data,palette='pink_r')


# In[ ]:


gender_placement=data.groupby(['gender','status']).workex.count().reset_index(name='count')
gender_placement['percent']=round(gender_placement['count']/gender_placement.groupby('gender')['count'].transform('sum')*100,2)
gender_placement


# In[ ]:


data.query('gender=="M" ' ).salary.describe()


# In[ ]:


data.query('gender=="F"').salary.describe()


# In[ ]:


plt.figure(figsize=(13,9))
plt.subplot(211)
sns.distplot(data.query('gender=="M"').salary,label='Male salary',norm_hist=True,kde=False)
plt.legend()
plt.title('Distribution of salary for each gender',size=25)
plt.subplot(212)
sns.distplot(data.query('gender=="F"').salary,label='female salary',color='pink',norm_hist=True,kde=False,bins=20)
plt.legend()


# In[ ]:


plt.figure(figsize=(10,7))
sns.boxplot(x='gender',y='salary',data=data,palette='pink_r')
plt.title('Distribution of salary for each gender',size=25)


# # Results

# 
# * As you can tell boys are getting more placements  than girls
# * the higher salary offred to a boy is 940000
# * the higher salary offred to a girl is 650000
# 
# 

# ## Secondery Education 10th Grade

# In[ ]:


data.ssc_p.describe()


# In[ ]:


sns.set(style='whitegrid')
plt.figure(figsize=(11,9))
sns.boxplot(x='status',y='ssc_p',data=data,palette='afmhot_r')
plt.title('Placement VS Percentage-10th Grade',size=25)


# In[ ]:


plt.figure(figsize=(11,9))
sns.countplot(x='status',hue='ssc_b',data=data,palette='mako')


# In[ ]:


sscb_placement=data.groupby(['ssc_b','status']).ssc_p.count().reset_index(name='count')
sscb_placement['percent']=round(sscb_placement['count']/sscb_placement.groupby('ssc_b')['count'].transform('sum')*100,2)
sscb_placement


# In[ ]:


data.query('ssc_b=="Central" ' ).salary.describe()


# In[ ]:


data.query('ssc_b=="Others" ' ).salary.describe()


# In[ ]:


plt.figure(figsize=(13,9))
plt.subplot(211)
sns.distplot(data.query('ssc_b=="Others" ').salary,label='Others board',norm_hist=True,kde=False,bins=20)
plt.legend()
plt.title('Distribution of salary for Board Education',size=25)
plt.subplot(212)
sns.distplot(data.query('ssc_b=="Central" ').salary,label='Central board',color='c',norm_hist=True,kde=False,bins=20)
plt.legend()


# In[ ]:


plt.figure(figsize=(11,9))
sns.violinplot(x='ssc_b',y='salary',data=data,palette='mako',inner='quartil')


# # Resulats
# 

# * We can see that the Range of percentage in 10th Grade is hight for  persons who had placed with medina near to 73
# * It's seems like  Board of education note affect to much the placement but i think it's better to chose others board because the range of salary of others is superior than the central bord

# ## Higher Secondery Education 12th Grade

# **let's see if students who did well in 10th grid ,did the same in 12th*

# In[ ]:


from scipy.stats import spearmanr
sns.jointplot(x='ssc_p',y='hsc_p',data=data,stat_func=spearmanr,kind='reg',height=10)


# > we can say that student how did well in 10th did well also in 12th grade even though the correlation is not to strog 

# In[ ]:


plt.figure(figsize=(11,9))
sns.boxplot(x='status',y='hsc_p',data=data,palette='afmhot_r')
plt.title('Placement VS Percentage-12th Grade',size=25)


# As we can see student who get palced have a high Higher Secondary Education percentage 

# In[ ]:


plt.figure(figsize=(11,9))
sns.countplot(x='hsc_s',hue='status',data=data,palette='seismic')


# Specialization in Higher Secondary Education doesn't affect the placement  but what about salaries ?

# In[ ]:


plt.figure(figsize=(11,9))
sns.violinplot(x='status',y='degree_p',data=data,palette='rocket',inner='quartilles')
plt.title('Placement VS Degree Percentage',size=25)


# the 3rd quartille for not placed student seems equal to 1rt quatille for student who get placed. the degree percentage affect the placement, students should have  a good degree.

# In[ ]:


sns.jointplot(x='salary',y='degree_p',data=data,stat_func=spearmanr,kind='reg',height=10,color='c')


# but the degree is not important to get a high salary

# In[ ]:


plt.figure(figsize=(11,9))
sns.countplot(x='status',hue='degree_t',data=data,palette='mako')


# In[ ]:


sscb_placement=data.groupby(['degree_t','status']).ssc_p.count().reset_index(name='count')
sscb_placement['percent']=round(sscb_placement['count']/sscb_placement.groupby('degree_t')['count'].transform('sum')*100,2)
sscb_placement

Students who have Comm&Mgmt or Sci&Tech as type of degree have a high chance to get palcement, let's see the salary
# In[ ]:


plt.figure(figsize=(11,9))
sns.boxplot(x='degree_t',y='salary',data=data,palette='seismic')


# Sci&Tech is the best choice when we talk about salaries 

# ### Is Work Experience important for Placment ?!

# In[ ]:


plt.figure(figsize=(11,9))
sns.countplot(x='status',hue='workex',data=data,palette='mako')


# In[ ]:


Exwork=data.groupby(['workex','status']).ssc_p.count().reset_index(name='count')
Exwork['percent']=round(Exwork['count']/Exwork.groupby('workex')['count'].transform('sum')*100,2)
Exwork


# Work Experience is very import as we can tell  86% of students who have a ex work get a placement 

# ## thank you for reading i hope that i could help a little bit

# In[ ]:




