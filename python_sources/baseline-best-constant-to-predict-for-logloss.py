#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# 
# In this kernel we will create a quick baseline by simply predicting the optimum constant for each class.
# 
# The metric for this competition is multi class logloss.<br>
# The best constant to predict for this metric is the frequency of each class. 
# 
# This simple and fast solution will serve as a common sense baseline that we can use to see  if using complex and computation intensive machine learning techniques will actually produce real benefits.

# In[ ]:


import pandas as pd
import os


# ### What files are available?

# In[ ]:


os.listdir('../input')


# ### What is the class distribution?

# In[ ]:


# read the train set into a dataframe
df_train = pd.read_csv('../input/train.csv')

df_train.head()


# In[ ]:


df_train['event'].value_counts()


# ### Calculate the frequency of each class

# In[ ]:


total = len(df_train)

freq_A = 2848809/total
freq_B = 130597/total
freq_C = 1652686/total
freq_D = 235329/total


# ### Create a submission csv file

# In[ ]:


# delete df_train to free up memory space
del df_train

# read the test set into a dataframe
df_test = pd.read_csv('../input/test.csv')

# create new columns in the test dataframe
df_test['A'] = freq_A
df_test['B'] = freq_B
df_test['C'] = freq_C
df_test['D'] = freq_D

df_test.head()


# In[ ]:


# select the columns that will be part of the submission 
submission = df_test[['id', 'A', 'B', 'C', 'D']]

# save the submission dataframe as a csv file
submission.to_csv('submission.csv', index=False, 
                  columns=['id', 'A', 'B', 'C', 'D'])


# In[ ]:


submission.head()


# ### Resources
# 
# 1. Course: How to Win a Data Science Competition, Week 3, Metrics Optimization<br>
# https://www.coursera.org/learn/competitive-data-science/home/week/3<br>
# 
# 2. Book: Deep Learning with Python by Francois Chollet, Chapter 6<br>
# https://www.manning.com/books/deep-learning-with-python
# 

# ### Conclusion
# This baseline score is already quite good. Now can we use machine learning to do better?
# 
# ***
# Thank you for reading.

# In[ ]:




