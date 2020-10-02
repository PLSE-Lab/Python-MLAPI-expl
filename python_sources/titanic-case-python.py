#!/usr/bin/env python
# coding: utf-8

# # **Importing Libraries**

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk # Sci-Kit Library
import matplotlib.pyplot as plt # Data Visualization
import seaborn as sb # Data Visualization
#to plot inline instead of an external window
get_ipython().run_line_magic('matplotlib', 'inline')


# # **Uploading Training Dataset**

# In[2]:


TrainingData = pd.read_csv('../input/train.csv')
TrainingData.shape


# TrainingData.shape showed us that we have **891** rows and **12** columns
# 
# Now we will load the first 5 rows of the data to have a look at the data 

# In[3]:


TrainingData.head(5)


# ## Definitions 
# 
# | Feature | Description | Comment |
# |------------| ----------------| --------------|
# | PassengerId | Passanger's ID on the ship | |
# | Survived | either dead or alive | 0 = Dead ,1 = Alive |
# | Pclass | Passanger's Ticket Class | 1= High , 2 = Mid , 3 = Low |
# | Name | Passanger's Name | |
# | Sex | Passanger's Sex | Male , Female |
# | Age | Passanger's Age | age in decimal |
# | SibSp | # of siblings / spouses aboard | |
# | Parch | # of parents / children aboard | |
# | Ticket | Ticket number | |
# | Fare | Passenger's fare | |
# | Cabin | Cabin number | |
# | Embarked | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |
# 
# after we defined all the features , now we will gather some info about the data to know what the next step will be 

# In[4]:


TrainingData.isnull().values.any()


# the upper function showed us that some rows have "Null" values , and we cant analyse null values , so let's find the null values so we can fix them

# In[5]:


TrainingData.info()


# From the information , we can see that **Age , Cabin and Embarked** are missing some data 
# 
# |Feature | Found | Missing | Total |
# |------------|----------|-----------|-------|
# |Age|714| 177 |891|
# |Cabin|204 |687| 891|
# |Embarked| 889 |2| 891 |
# 
# 
# **now we are going to work on the missing data**

# In[ ]:





# In[6]:


TrainingData.corr()


# In[ ]:



from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

