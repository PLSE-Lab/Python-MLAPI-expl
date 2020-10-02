#!/usr/bin/env python
# coding: utf-8

# > # HR Data Exploration 

# Import Library

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Reading the data from file

# In[ ]:


df = pd.read_csv('../input/core_dataset.csv')
df.head(50) #first seven data points


# Data Exploration

# In[ ]:


df.describe()


# In[ ]:


df.shape #(rows,columns)


# In[ ]:


df.columns


# Gender 

# In[ ]:


df['Sex']


# In[ ]:


df['Sex'].unique()


# In[ ]:


df['Sex'].replace('male','Male',inplace=True)


# In[ ]:


df['Sex'].unique()


# In[ ]:


df.dropna(subset=['Sex'],inplace=True) # remove rows with nan values for Sex
df.shape


# In[ ]:


df['Sex'].value_counts() #Lets plot this
df['Sex'].value_counts().plot(kind='bar')


# In[ ]:


import matplotlib.pyplot as plt

df['Sex'].value_counts().plot(kind='bar')


# In[ ]:


# Gender diversity across departmets
import seaborn as sns
plt.figure(figsize=(16,9))
ax=sns.countplot(x=df['Department'],hue=df['Sex'])


# Conclusions from graph :
# 
# No males in executive office and no females in software engineering department.
# Gender diversity is not maintained in production department and software engineering.(No.of females is nearly double the number of males)

# Other Graphs

# In[ ]:


df['RaceDesc'].value_counts().plot(kind='pie')


# In[ ]:


df['CitizenDesc'].unique()


# In[ ]:


df['CitizenDesc'].value_counts().plot(kind='bar')


# In[ ]:


df['Age'].hist()


# In[ ]:


df['Age'].hist(bins=40)


# In[ ]:


df['Position'].unique()


# In[ ]:


plt.figure(figsize=(16,9))
df['Position'].value_counts().plot(kind='bar')


# Is there any relationship between pay rate and age?

# In[ ]:


df['Pay Rate'].describe()


# In[ ]:


df['Age'].describe()


# In[ ]:


df.plot(x='Age',y='Pay Rate',kind='scatter')
# Looks like thery are not related! 


# **Which manager has the best performers?**
# 

# In[ ]:


df['Manager Name'].unique()


# In[ ]:


df['Performance Score']


# In[ ]:


plt.figure(figsize=(20,20))
sns.countplot(y=df['Manager Name'], hue=df['Performance Score'])


# * Davind Stanley and Kelly Spirea have highest number of employees who fully meet the expectation.
# * Simon and Brannon have a highest number of exceptional employess!
# * Employees working with Michael need to improve their performance.

# **Which department pays more?**

# In[ ]:


df.groupby('Department')['Pay Rate'].sum().plot(kind='bar')
#Production department pays more!


# Which position gives away more money? This doesn't mean that all employees in this position get maximum pay. The number of employees could be more for this dept. Note that we are taking sum of pay rate for each department.

# In[ ]:


plt.figure(figsize=(16,9))
df.groupby('Position')['Pay Rate'].sum().plot(kind='bar')


# Who gets the highest salary ? ;)

# In[ ]:


id_of_person_with_highgest_pay = df['Pay Rate'].idxmax()
df.loc[id_of_person_with_highgest_pay]


df.loc[df['Pay Rate'].idxmax()]


# In[ ]:


HispLat_map ={'No': 0, 'Yes': 1, 'no': 0, 'yes': 1}
df['Hispanic/Latino'] = df['Hispanic/Latino'].replace(HispLat_map)
df['Hispanic/Latino']


# In[ ]:


sns.violinplot('Hispanic/Latino', 'Pay Rate', data = df)


# # Class Exercise

# 1. Plot a Bar Chart of the count of Marital Status**
# 2. Plot a bar chart of Employment status group by Martial Status
# 3. Explore the data of another data set - HRDataset_v9
# 4. With the new dataset, plot a scatter graph Days Employed vs Pay Rate
# 5. Plot a violin Chart of Citizenship and Pay Rate
# 6. Plot a bar chart of Race Description group by Citizenship

# In[ ]:


df.columns


# In[ ]:


df['MaritalDesc'].value_counts().plot(kind='bar')


# In[ ]:


df['MaritalDesc'].value_counts().plot(kind='bar')


# In[ ]:


plt.figure(figsize=(16,9))
ax=sns.countplot(x=df['Employment Status'],hue=df['MaritalDesc'])


# In[ ]:


df2 = pd.read_csv('../input/HRDataset_v9.csv')
df2.head(50) #first seven data points


# In[ ]:


df2.columns


# In[ ]:


df2.plot(x='Days Employed',y='Pay Rate',kind='scatter')


# In[ ]:


sns.violinplot('CitizenDesc', 'Pay Rate', data = df2)


# In[ ]:


plt.figure(figsize=(16,9))
ax=sns.countplot(x=df2['CitizenDesc'],hue=df2['RaceDesc'])


# In[ ]:




