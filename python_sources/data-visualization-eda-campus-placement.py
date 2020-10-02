#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")


# In[ ]:


df.head()


# #### Let us see If the data has any NaN values

# In[ ]:


df.info()


# In[ ]:


sns.heatmap(df.isnull(),yticklabels = False,cbar = False,cmap = 'viridis')


# From the above graph we can see that salary column has missing values. We can replace Null value to 0 because those are not placed can have zero salary.

# In[ ]:


df['salary'] = df['salary'].fillna(value = 0)


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


sns.heatmap(df.isnull(),yticklabels = False,cbar = False,cmap = 'viridis')


# Now I think we have replace all the null values with zero. Here we can say that our data cleaning process is completed

# In[ ]:


print(df['gender'].unique())
print(df['status'].unique())
print(df['workex'].unique())
print(df['hsc_b'].unique())
print(df['ssc_b'].unique())


# As we see there are two columns for serial no. We can replace sl_no as index column

# In[ ]:


df.set_index('sl_no',inplace = True)


# In[ ]:


df.head()


# # Data Visualization
# 
# #### Let's Visualize the Categorical Data

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x = 'gender', data = df)
plt.title('Gender Comparison')
df['gender'].value_counts()


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x = 'ssc_b', data = df)
plt.title('College Comparison')
df['ssc_b'].value_counts()


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x = 'degree_t', data = df)
plt.title('Degree Comparison')
df['degree_t'].value_counts()


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x = 'hsc_s', data = df)
plt.title('Stream Comparison')
df['hsc_s'].value_counts()


# Percentage of male and female present in placement

# In[ ]:


plt.pie(df.gender.value_counts(),labels = ['Male','Female'], autopct = '%1.2f%%', shadow = True, startangle = 90)


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x = 'status', data = df)
plt.title('No of students placed')
df['status'].value_counts()


# #### Visualization of placement among the gender

# In[ ]:


sns.set_style('darkgrid')
sns.countplot(x = 'gender', hue = 'status', data = df)
plt.title('No of students placed amongst gender')
print('Out of 139 males, 100 got placed which is {}'.format((100/139)*100))
print('Out of 76 females, 48 got placed which is {}'.format((48/76)*100))


# #### Visualization of placement amongst the board in SSC

# In[ ]:


sns.set_style('darkgrid')
sns.countplot(x = 'ssc_b', hue = 'status', data = df)
plt.title('No of students placed amongst board in SSC')


# Here we can see that candidate belongs to central board placed more as compared to other board however there is not much of a difference.

# #### Visualization of placement amongst the board in HSC

# In[ ]:


sns.set_style('darkgrid')
sns.countplot(x = 'hsc_b', hue = 'status', data = df)
plt.title('No of students placed amongst board in HSC')


# Here more students got placed from other board as compared to central board

# #### Visualization of placement amongst the degree

# In[ ]:


sns.set_style('darkgrid')
sns.countplot(x = 'degree_t', hue = 'status', data = df)
plt.title('No of students placed amongst degree')


# Turns out more than 100 students got placed having Comm&Mgmt degree

# #### Visualize whether work experience had an impact on students who were placed

# In[ ]:


sns.set_style('darkgrid')
sns.countplot(x = 'workex', hue = 'status', data = df)
plt.title('No of students placed amongst workex')


# Seems like not much get affected by work experience

# #### Does percentage matters for one to get placed?

# In[ ]:


sns.scatterplot(x = 'ssc_p', y = 'status', data = df)


# In[ ]:


sns.scatterplot(x = 'hsc_p', y = 'status', data = df)


# In[ ]:


sns.scatterplot(x = 'degree_p', y = 'status', data = df)


# In[ ]:


sns.scatterplot(x = 'mba_p', y = 'status', data = df)


# #### Conclusion:
# 1) In SSC and HSC, Candidates having less than 50% marks are surely not get placed and more than 80% are surely placed. In between we can't say anything.
# 2) In degree, Candidates having less than 55% marks are surely not get placed and more than 80% are surely placed. In between we can't say anything.
# 3) In MBA, we can't say anything on the basis of percentage.
# 
# Overall we can't say whether the student got placed on the basis of percentage they got in different standard.

# #### Which degree specialization is much demanded by corporate?

# In[ ]:


sns.set_style('darkgrid')
sns.countplot(x = 'specialisation', hue = 'status', data = df)
plt.title('No of students placed amongst specialisation')


# Here More number of students get placed from Mkt&Fin as compared to Mkt&HR

# In[ ]:


data = df[['salary','ssc_p','hsc_p','degree_p','etest_p','mba_p']]
corr = data.corr()
corr


# In[ ]:


plt.figure(figsize = (8,6))
sns.heatmap(corr, annot = True)

