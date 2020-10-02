#!/usr/bin/env python
# coding: utf-8

# **04a_Naive.ipynb**
# 
# **Purpose:** Naive solution for titanic 
# 
# **Author**: Alan Chalk  
# **Modified**: 28  April 2018
# 
# **Contents**:
#  - Start_. Packages, directoties, functions
#  - 1. Read the data
#  - 2. Prepare naive solution
#  - 3. Questions
#  
#  
# **Notes**

# ![](http://)### Start_. Import required packages

# In[ ]:


# Admin
import os

# Data manipulation
import pandas as pd


# Check the working directory.  When using a Kaggle kernel the code should work without any changes. When using the structure suggested in AML, you need to be in the Titanic/PCode subdirectory.

# In[ ]:


os.getcwd()


# ### 1. Read the data

#  - Note: In the code below, ../ goes up one directory level
#  - Note: My personal preference is, that if when naming objects you are going to use two or three letters to describe the type of object (eg df_ for dataframe in the code below), then these two or three letters should be at the beginning of the name, not the end, e.g. df_all, mat_X, bln_ind

# In[ ]:


df_train = pd.read_csv('../input/train.csv')


# In[ ]:


df_train.columns


# In[ ]:


get_ipython().run_cell_magic('html', '', '<style>\ntable {float:left}\n</style>')


# Description of the features
# 
# |name| description|
# | :- | :- |
# |PassengerId|
# |Survived| Did the passenger survive (0 = No, 1 = Yes)|
# |Pclass|Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)|
# |Name||
# |Sex||
# |Age|Age in years|
# |SibSp|# of siblings / spouses aboard the Titanic|
# |Parch|# of parents / children aboard the Titanic|
# |Ticket|Ticket number|
# |Fare| Passenger fare|
# |Cabin|Cabin number|
# |Embarked|Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)|

# In[ ]:


df_train.head(3)


# In[ ]:


df_train['Survived'].value_counts(dropna = False)


# In[ ]:


1 - df_train['Survived'].mean()


# In[ ]:


df_test = pd.read_csv('../input/test.csv')


# In[ ]:


df_test.head(3)


# Concatenate the two datasets

# In[ ]:


# not run
#df_all = pd.concat([df_train, df_test], sort = False)
#df_all['Survived'].value_counts(dropna = False)


# ### 2. Prepare naive solution
# 
# The simplest solution is to predict the majority class

# In[ ]:


pd_sub_1 = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": 0
    })


# In[ ]:


pd_sub_1.head()


# In[ ]:


#pd_sub_1.to_csv('../POutput/04a_pd_sub_1.csv', index=False)
pd_sub_1.to_csv('04a_pd_sub_1.csv', index=False)


# Score was 0.62679

# ### Questions
# 
# 1. What perentage of passengers did not survive in the train data?
# 2. What perentage of passengers did not survive in the test data?
# 3. What is the overall survival percentage (including both train and test data)?
# 4. Given the overall survival percentage, could the difference between train and test survival percentages be due to randomness in the samples?
# 5. When preparing other (non naive) solutions which give probabilties as predictions, is it worthwhile to ensure that the average of the predictions is equal to 0.62679?  If yes, how would you make the adjustment?  If not, why not?

# In[ ]:




