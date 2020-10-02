#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')


# In[ ]:


data.head(10)


# In[ ]:


data.shape


# In[ ]:


#replacing space with underscore('_') for convenient accessing of columns
data.columns = data.columns.str.replace(' ','_')


# In[ ]:


data.columns


# In[ ]:


# checking for null values
data.isna().sum()


# ## Plotting with Single Feature

# In[ ]:


sns.set(font_scale=1.4)
plt.figure(figsize=(7,6))
sns.countplot(x='gender',data=data)


# The dataset contains more number of female observations(not much) than male observations. 
# 
# Let's see its quantity and ratio below

# In[ ]:


print(data.gender.value_counts())
print('----------------------------------')
print(data.gender.value_counts(normalize=True))


# In[ ]:


sns.set(font_scale=1.4)
plt.figure(figsize=(7,6))
sns.countplot(x='race/ethnicity',data=data)


# 'group C' contains more no. of observations than any other groups in our dataset

# In[ ]:


sns.set(font_scale=1.4)
plt.figure(figsize=(14,8))
sns.countplot(x='parental_level_of_education',data=data)


# The dataset contains more observations on the parental level of education with "some college" and "associate's degree"

# In[ ]:


sns.set(font_scale=1.4)
plt.figure(figsize=(7,6))
sns.countplot(x='lunch',data=data)


# The count of 'standard' lunch seems to be higher than 'free/reduced' lunch in the dataset

# In[ ]:


sns.set(font_scale=1.4)
plt.figure(figsize=(7,6))
sns.countplot(x='test_preparation_course',data=data)


# Most of the students in the dataset haven't completed their 'test preparation course'

# ## Plotting with Two Features

# In[ ]:


plt.figure(figsize=(18,8))
sns.countplot(x="parental_level_of_education",hue="gender",data=data)
plt.ylabel('Count of Parental Level of Education')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Parents of female students have their master's degree more than the parents of male students.

# In[ ]:


data.groupby('gender').agg('mean').plot(kind='bar',figsize=(7,5.5))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Female students are noted to be performed well in both reading and wrinting but male students performed exceptionally well in maths(Woah! There must be something wrong in the dataset. lol Just Kidding! ;-p )

# In[ ]:


data.groupby('parental_level_of_education').agg('mean').plot(kind='barh',figsize=(10,10))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# On the whole, students with parents having Master's degree scored well in all three tests

# In[ ]:


data.groupby('test_preparation_course').agg('mean').plot(kind='barh',figsize=(7,7))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Students who completed their test preparation course seemed to be score well in all three tests

# In[ ]:


data.groupby('lunch').agg('mean').plot(kind='barh',figsize=(7,7))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Students with standard lunch scored well in all three tests. Interesting fact, right? So, always have a standard lunch with you, if you want to score well! (Again kidding!) 

# In[ ]:


data.groupby('race/ethnicity').agg('mean').plot(kind='barh',figsize=(9,9))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Students whose race/ethnicity belongs to 'group E' seemed to score pretty good in all three tests

# In[ ]:


plt.figure(figsize=(8,5.5))
sns.countplot(x='race/ethnicity',hue='test_preparation_course',data=data)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Students belonging to 'group C' has high observations in both the categories(none and completed). Keep in mind that the count of 'group C' is higher than any other groups in the dataset

# In[ ]:


sns.countplot(x='gender',hue='lunch',data=data)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# There is no big difference in both genders when comes to lunch type

# In[ ]:


sns.countplot(x='gender',hue='test_preparation_course',data=data)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Female students are noted to be slightly higher in both cases(none and completed) when compared with male students.(Again keep in mind that the count of female observation is higher than the male observation)

# ## Plotting with Multiple Features

# In[ ]:


data.groupby(['race/ethnicity','gender']).agg('mean').plot(kind='bar',figsize=(12,8))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# The main point to be noted is that female students who belonging to group E have performed well in reading and writing test and male students who belonging to group E have performed well in math test. Overall, students belonging to group E performed well in all three tests as we saw in earlier plot.

# In[ ]:


data.groupby(['gender','lunch']).agg('mean').plot(kind='bar',figsize=(12,8))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Female students who are having standard lunch seemed to perform well in reading and writing test and male students with standard lunch performed well in math test. Overall, Students with standard lunch performed well in all three tests as we saw in the earlier plot.

# In[ ]:


data.groupby(['gender','parental_level_of_education']).agg('mean').plot(kind='barh',figsize=(12,10))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Male students whose parents having Master's degree are observed to be performed pretty good in Math test and female students whose parents having Master's degree are observed to be performed well in reading and writing test. Predominantly, Students whose parents having Master's degree are seemed to perform well in all three tests as we saw in the earlier plot

# In[ ]:


data.groupby(['gender','test_preparation_course']).agg('mean').plot(kind='bar',figsize=(12,8))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Female students who completed their 'test preparation course' are noted to perform well in reading and writing test and Male students who completed their 'test preparation course' are noted to be performed well in Math test. Taking those facts into consideration, we can conclude that Students who completed their 'test preparation course' seemed to be score well in all three tests as we saw in the earlier plot

# In[ ]:


data.groupby(['race/ethnicity','parental_level_of_education']).agg('mean').plot(kind='barh',figsize=(12,12))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Students who belonging to 'group E' and whose parents having Master's degree are noted to score well in reading and writing test and Students who belonging to 'group E' and whose parents having Bachelor's degree are noted to perform well in Math test

# Thanks for Viewing. Please do upvote, if you like my notebook
