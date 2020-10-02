#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataset = pd.read_csv('../input/StudentsPerformance.csv')


# In[ ]:


dataset.info()


# In[ ]:


dataset.head(3)


# In[ ]:


# Importing the Visualization Library
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# # **Visualizing the dataset**

# In this dataset , we have total 8 columns. The first 5 columns are the deciding factor and last 3 columns are the scores of individual students. Here we try to check the effects on the mean score for every deciding factors.
# 
# First , we try to check whether there is a effect on the mean score due to variation on gender or not.

# In[ ]:


math_score_mean = dataset.groupby('gender')['math score'].mean()
reading_score_mean = dataset.groupby('gender')['reading score'].mean()
writing_score_mean = dataset.groupby('gender')['writing score'].mean()
plt.figure(figsize=(16,5))
plt.subplot(131)
sns.barplot(x= math_score_mean.index , y = math_score_mean.values)
plt.ylabel('Math Mean Score')
plt.subplot(132)
sns.barplot(x= reading_score_mean.index , y = reading_score_mean.values)
plt.ylabel('Reading Mean Score')
plt.subplot(133)
sns.barplot(x= writing_score_mean.index , y = writing_score_mean.values)
plt.ylabel('Writing Mean Score')


# Here we can see that the average score for male in math is much compared to that of female. But for the other two mean scores, female leads.

# In[ ]:


dataset['race/ethnicity'].value_counts()


# In[ ]:


math_score_mean = dataset.groupby('race/ethnicity')['math score'].mean()
reading_score_mean = dataset.groupby('race/ethnicity')['reading score'].mean()
writing_score_mean = dataset.groupby('race/ethnicity')['writing score'].mean()
plt.figure(figsize=(16,5))
plt.subplot(131)
sns.barplot(x= math_score_mean.index , y = math_score_mean.values)
plt.ylabel('Math Mean Score')
plt.subplot(132)
sns.barplot(x= reading_score_mean.index , y = reading_score_mean.values)
plt.ylabel('Reading Mean Score')
plt.subplot(133)
sns.barplot(x= writing_score_mean.index , y = writing_score_mean.values)
plt.ylabel('Writing Mean Score')


# Here we can see a pattern in mean scores. We have noticed that the mean score gradually increases for ethnicity group from A to E

# In[ ]:


dataset['parental level of education'].value_counts()


# In[ ]:


math_score_mean = dataset.groupby('parental level of education')['math score'].mean()
reading_score_mean = dataset.groupby('parental level of education')['reading score'].mean()
writing_score_mean = dataset.groupby('parental level of education')['writing score'].mean()
plt.figure(figsize=(20,8))
plt.subplot(311)
sns.barplot(x= math_score_mean.index , y = math_score_mean.values)
plt.ylabel('Math Mean Score')
plt.subplot(312)
sns.barplot(x= reading_score_mean.index , y = reading_score_mean.values)
plt.ylabel('Reading Mean Score')
plt.subplot(313)
sns.barplot(x= writing_score_mean.index , y = writing_score_mean.values)
plt.ylabel('Writing Mean Score')


# We can infer that parents level of educations does not affect on the mean score of the students so much. So we can exclude this section for our next bivariate tests

# In[ ]:


dataset['lunch'].value_counts()


# In[ ]:


math_score_mean = dataset.groupby('lunch')['math score'].mean()
reading_score_mean = dataset.groupby('lunch')['reading score'].mean()
writing_score_mean = dataset.groupby('lunch')['writing score'].mean()
plt.figure(figsize=(16,5))
plt.subplot(131)
sns.barplot(x= math_score_mean.index , y = math_score_mean.values)
plt.ylabel('Math Mean Score')
plt.subplot(132)
sns.barplot(x= reading_score_mean.index , y = reading_score_mean.values)
plt.ylabel('Reading Mean Score')
plt.subplot(133)
sns.barplot(x= writing_score_mean.index , y = writing_score_mean.values)
plt.ylabel('Writing Mean Score')


# Individuals with statndard lunch has significantly secured much score than other students on average.

# In[ ]:


dataset['test preparation course'].value_counts()


# In[ ]:


math_score_mean = dataset.groupby('test preparation course')['math score'].mean()
reading_score_mean = dataset.groupby('test preparation course')['reading score'].mean()
writing_score_mean = dataset.groupby('test preparation course')['writing score'].mean()
plt.figure(figsize=(16,5))
plt.subplot(131)
sns.barplot(x= math_score_mean.index , y = math_score_mean.values)
plt.ylabel('Math Mean Score')
plt.subplot(132)
sns.barplot(x= reading_score_mean.index , y = reading_score_mean.values)
plt.ylabel('Reading Mean Score')
plt.subplot(133)
sns.barplot(x= writing_score_mean.index , y = writing_score_mean.values)
plt.ylabel('Writing Mean Score')


# The students, who have completed the test preparation course , considerably scores high in all tests

# **Now we make an extra column which is the total of three individual scores. We named it as a total score. **
# 
#  We shall now check whether two deciding factors simulaneously affects the total mean score or not. 

# In[ ]:


dataset['total number'] = dataset['math score'] + dataset['reading score'] +dataset['writing score']
dataset.head()


# At first, we shall take two deciding factors gender and race/ethnicity. Now we shall check whether these two factors has any distinguishable difference or not.

# In[ ]:


total_score = dataset.groupby(['gender' , 'race/ethnicity'])['total number'].mean()
df = pd.DataFrame(total_score)


# In[ ]:


total_score_1 = dataset.groupby(['gender' , 'race/ethnicity'])['total number'].mean()
total_score_df_1= pd.DataFrame(total_score_1)


# In[ ]:


total_score_df_1.reset_index(inplace=True)
total_score_df_1.columns


# In[ ]:


plt.figure(figsize=(16,5))
sns.barplot(x = 'race/ethnicity' , y = 'total number' ,hue = 'gender', data=total_score_df_1)


# Here we can see that, for each ethnicity ,female students considerably scores higher for average in total numbers.

# In[ ]:


total_score_2 = dataset.groupby(['gender' , 'lunch'])['total number'].mean()
total_score_df_2= pd.DataFrame(total_score_2)


# In[ ]:


total_score_df_2.reset_index(inplace=True)
total_score_df_2.columns


# In[ ]:


plt.figure(figsize=(16,5))
sns.barplot(x = 'lunch' , y = 'total number' ,hue = 'gender', data=total_score_df_2)


# In this graph, we can see that students  having free/reduced lunch has no effect due to gender on average /total score. For standard lunch, female scores higher on average.

# In[ ]:


total_score_3 = dataset.groupby(['gender' , 'test preparation course'])['total number'].mean()
total_score_df_3= pd.DataFrame(total_score_3)


# In[ ]:


total_score_df_3.reset_index(inplace=True)
total_score_df_3.columns


# In[ ]:


plt.figure(figsize=(16,5))
sns.barplot(x = 'test preparation course' , y = 'total number' ,hue = 'gender', data=total_score_df_3)


# In this graph, we can see that students who have completed the test preparation course scores higher in total average , but female leads here again  in two cases.

# ## **The Conclusion**

# Considering all the graphs, we can say that male students considerably scores higher in maths than female students. But , overall, the total scores by a **female student on average is higher than male**.
# 
# The main deciding factors on score of the students are as follows:
# 1) Race \ Ethnicity
# 2) Lunch
# 3) Students completed test preparartion course or not.

# In[ ]:




