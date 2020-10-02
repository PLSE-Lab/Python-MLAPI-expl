#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ![](data:image/jpeg;base64,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)

# Dataset Overview

# In[ ]:


udemy= pd.read_csv('/kaggle/input/udemy-courses/udemy_courses.csv')


# In[ ]:


udemy.head()


# In[ ]:


#checking the columns of the data set
for i in udemy:
    print(i)


# Exploratory Data Analysis

# In[ ]:


#summary
udemy.describe()


# In[ ]:


#Checking missing values
udemy.isna().sum()


# In[ ]:


#Paid courses data frame
Paid=udemy[udemy['is_paid']==True]
Paid.head()


# In[ ]:


Paid.shape


# In[ ]:


Paid_total_num=Paid.shape[0]
Paid_total_num


# In[ ]:


Total_course_num=udemy.shape[0]
Total_course_num


# In[ ]:


#Percentages of Paid Courses
PercPaidCourses=(Paid_total_num/Total_course_num)*100
PercPaidCourses


# In[ ]:


Paid['num_subscribers'].max()


# In[ ]:


#The maximum subcribed course
Paid[Paid['num_subscribers']==121584]


# In[ ]:


udemy.nlargest(10, 'num_subscribers')


# According to Number of Subscribers,
# 
# Well,Web Developer Bootcamp it is not suprising as I have also taken the course
# 
# Web Developer Bootcamp is top Paid course while **Learn HTML5 Programming From Scratch** top courses.

# In[ ]:


udemy.groupby(['subject']).mean()


# People used to go to udemy for online courses and content in udemy is also more.

# In[ ]:


udemy.groupby(['level']).mean()


# Expert levels are paid and costly than others.
# 
# All levels have more content,subscribers and reviews.
# 

# In[ ]:


#Most reviews
udemy.nlargest(5, 'num_reviews')


# More to continue 
# 
# Thanks for your time!!