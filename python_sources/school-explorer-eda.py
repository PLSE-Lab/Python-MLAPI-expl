#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as matplot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


student = pd.read_csv('../input/2016 School Explorer.csv')
student.head(2)


# In[ ]:


student.shape


# In[ ]:


student.columns = student.columns.str.replace(' ','_')
student.columns = student.columns.str.replace('?','')
student.columns = student.columns.str.replace('%','')
student.head(1)


# In[ ]:


student = student.drop(['Adjusted_Grade','New','Other_Location_Code_in_LCGMS'], axis = 1)


# In[ ]:


student = student.dropna(how = "any")


# In[ ]:


def convert(x):
    return float(x.strip('%'))/100


student['Percent_Asian'] = student['Percent_Asian'].astype(str).apply(convert)
student['Percent_White'] = student['Percent_White'].astype(str).apply(convert)
student['Percent_Black'] = student['Percent_Black'].astype(str).apply(convert)
student['Percent_Hispanic'] = student['Percent_Hispanic'].astype(str).apply(convert)
student['Percent_of_Students_Chronically_Absent'] = student['Percent_of_Students_Chronically_Absent'].astype(str).apply(convert)
student['Trust_'] = student['Trust_'].astype(str).apply(convert)


# In[ ]:


f, axes = plt.subplots(ncols=4, figsize=(20, 6))

sns.distplot(student['Percent_Hispanic'], kde=False, color="b", ax=axes[0], bins=35).set_title('No. of Hispanic Distribution (%)')
sns.distplot(student['Percent_Black'], kde=False, color="g", ax=axes[1], bins=35).set_title('No. of Black Distribution (%)')
sns.distplot(student['Percent_Asian'], kde=False, color="y", ax=axes[2], bins=25).set_title('No. of Asian Distribution (%)')
sns.distplot(student['Percent_White'], kde=False, color="r", ax=axes[3], bins=25).set_title('No. of White Distribution (%)')

plt.show()


# Observation -  
# The amount of hispanic race is maximum followed by black , asian and white.

# In[ ]:


sns.stripplot(y="Effective_School_Leadership_Rating", x="Percent_of_Students_Chronically_Absent", data=student)
plt.show()


# Observation -  
# Not too many absentees for schools which meet or exceed expectation targets.

# In[ ]:


temp = sns.distplot(student[['Economic_Need_Index']].values, kde=False, color = 'c')
temp= plt.title("ENI distribution")
temp = plt.xlabel("ENI")
temp = plt.ylabel("No. of Schools")
plt.show()


# Observation -  
# High number of schools fall in higher ENI value.

# In[ ]:


df = student.groupby('Community_School')['Trust_'].mean()
print(df.head())


# Observation -  
# The mean of a community and a non-community type of school is almost the same.
