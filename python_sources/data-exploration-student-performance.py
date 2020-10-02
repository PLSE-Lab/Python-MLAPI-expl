#!/usr/bin/env python
# coding: utf-8

# 
# **A SIMPLE DATA EXPLORATION.
# **
# Using a dataset on kaggle I tried to do a little data exploration by using the most common plot

# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
df = pd.read_csv('../input/StudentsPerformance.csv')
# Any results you write to the current directory are saved as output.


# In[ ]:





# **SUMMARY OF DATASET**

# In[21]:


df.head(10) #summary


# In[22]:


df.columns #names of columns


# **Gender Count**

# In[23]:


plt.figure(figsize=(12,6))
df.gender.value_counts()
sns.countplot(x="gender", data=df, palette="bwr")
plt.show()


# **PARENTAL LEVEL OF EDUCATION
# **

# In[24]:


df.gender.value_counts()
plt.figure(figsize=(12,6))
sns.countplot(x="parental level of education", data=df, palette="bwr")
plt.show()


# **# 'Completed Pre-test divided by gender'**

# In[25]:


pd.crosstab(df.gender,df['test preparation course']).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])
plt.title('Completed Pre-test divided by gender')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Completed", "Not Completed"])
plt.ylabel('Frequency')
plt.show()


# **# TEST PREPARATION COURSE
# **

# In[26]:


df['test preparation course'].value_counts()
plt.figure(figsize=(12,6))

sns.countplot(x="test preparation course", data=df, palette="bwr")
plt.show()


# Distribution by ethnicity

# In[27]:


plt.figure(figsize=(8,6))
df['race/ethnicity'].value_counts().head(10).plot.bar()


# In[ ]:





# **Treating the categorical variables**
# One hot econding

# In[28]:


df_treat = pd.get_dummies(df)
df_treat.dtypes.value_counts()
df_treat.head(10)


# **# DISTRIBUTION OF TESTS SCORING
# **

# In[29]:


fig = plt.figure(figsize=(8,10))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
sns.distplot(df['math score'],ax=ax1,color="y")
sns.distplot(df['writing score'],ax=ax2,color="r")
sns.distplot(df['reading score'],ax=ax3)


# **SCATTER PLOT**

# In[30]:


g = sns.FacetGrid(df, hue='gender', size = 7)
g.map(plt.scatter, 'math score','reading score', edgecolor="w")
g.add_legend()


# Of course the girl are better in math than the men!!!!!!!!

# In[31]:


g = sns.FacetGrid(df, hue='test preparation course', size = 7)
g.map(plt.scatter, 'math score','reading score', edgecolor="w")
g.add_legend()


# Distribution of tests scoring by gender

# In[32]:


p = sns.FacetGrid(data = df, hue = 'gender', size = 5, legend_out=True)
p = p.map(sns.kdeplot, 'math score')
plt.legend()
p = sns.FacetGrid(data = df, hue = 'gender', size = 5, legend_out=True)
p = p.map(sns.kdeplot, 'reading score')
plt.legend()
p = sns.FacetGrid(data = df, hue = 'gender', size = 5, legend_out=True)
p = p.map(sns.kdeplot, 'writing score')

plt.legend()


# The distribution plot confirm: girls are better than mens as you can see!!
