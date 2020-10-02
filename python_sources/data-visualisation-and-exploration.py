#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/core_dataset.csv')
df.head(7) #first seven data points


# In[ ]:


df.shape #(rows,columns)


# **Gender diversity**

# In[ ]:


df['Sex'].unique()


# In[ ]:


df['Sex'].replace('male','Male',inplace=True)


# In[ ]:


df.dropna(subset=['Sex'],inplace=True) # remove rows with nan values for Sex
df.shape


# In[ ]:


df['Sex'].value_counts() #Lets plot this


# In[ ]:


import matplotlib.pyplot as plt

df['Sex'].value_counts().plot(kind='bar')
# More female employees!


# In[ ]:


# Gender diversity across departmets
import seaborn as sns
plt.figure(figsize=(16,9))
ax=sns.countplot(x=df['Department'],hue=df['Sex'])


# Conclusions from graph :
# *  No males in executive office and no females in software engineering department.
# *  Gender diversity is not maintained in production department and software engineering.(No.of females is nearly double the number of males)

# In[ ]:


df['MaritalDesc'].value_counts().plot(kind='pie')


# In[ ]:


df['CitizenDesc'].unique()


# In[ ]:


df['CitizenDesc'].value_counts().plot(kind='bar')


# In[ ]:


df['Position'].unique()


# In[ ]:


plt.figure(figsize=(16,9))
df['Position'].value_counts().plot(kind='bar')


# **Is there any relationship between pay rate and age?**

# In[ ]:


df['Pay Rate'].describe()


# In[ ]:


df['Age'].describe()


# In[ ]:


df.plot(x='Age',y='Pay Rate',kind='scatter')
# Looks like thery are not related! 


# **How is the performance score related to pay rate? **

# In[ ]:


df['Performance Score'].isna().any()


# In[ ]:


df_perf = pd.get_dummies(df,columns=['Performance Score'])


# In[ ]:


df_perf.head(7)


# In[ ]:


col_plot= [col for col in df_perf if col.startswith('Performance')]
col_plot


# In[ ]:


fig, axes = plt.subplots(3, 3, figsize=(16,9))
for i,j in enumerate(col_plot):
    df_perf.plot(x=j,y='Pay Rate',ax = axes.flat[i],kind='scatter')
    
#Doesn't look like 


# **Which manager has the best performers?**

# In[ ]:


df['Manager Name'].unique()


# In[ ]:


plt.figure(figsize=(20,20))
sns.countplot(y=df['Manager Name'], hue=df['Performance Score'])


# * Davind Stanley and Kelly Spirea have highest number of employees who fully meet the expectation. 
# * Simon and Brannon have a highest number of exceptional employess!
# * Employees working with Michael need to improve their performance.

# **Pay rate analysis**

# In[ ]:


df['Pay Rate'].describe()


# **Which department pays more?**

# In[ ]:


df.groupby('Department')['Pay Rate'].sum().plot(kind='bar')
#Production department pays more!


# **Which position gives away more money?** 
# This doesn't mean that all employees in this position get maximum pay. The number of employees could be more for this dept.
# Note that we are taking sum of pay rate for each department. 

# In[ ]:


plt.figure(figsize=(16,9))
df.groupby('Position')['Pay Rate'].sum().plot(kind='bar')


# **Who gets the highest salary ? ;) **

# In[ ]:


df.loc[df['Pay Rate'].idxmax()]
# The CEO :p 

