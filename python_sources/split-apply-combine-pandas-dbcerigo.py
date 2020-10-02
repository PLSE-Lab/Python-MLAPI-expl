#!/usr/bin/env python
# coding: utf-8

# # Split-Apply-Combine
# A series of short exercises to flex your Pandas split-apply-combine skills.

# ## The Data - Titanic Passenger Survival
# This a (classic) data set on the passengers of the Titanic. 

# In[35]:


import numpy as np # linear algebra
import pandas as pd # data processing, 

df = pd.read_csv('../input/train.csv')
df = df.drop(columns=['Ticket', 'Cabin', 'PassengerId', 'SibSp', 'Parch'])


# In[36]:


print(len(df))
df.head()


# Each row is a passenger. The columns mean as follows:
# * `Survived`: `1` = Yes (Passenger did survive), `0` = No
# * `Pclass`: Passenger Classs - `1` = 1st, `2` = 2nd, `3` = 3rd
# * `Name`: Passenger Name (not really needed, just for curiosity)
# * `Sex`: Sex the passenger
# * `Age`: Age of the passenger
# * `Fare`: The price of the passengers ticket
# * `Embarked`: Location that passenger embarked from - `C` = Cherbourg, `Q` = Queenstown, `S` = Southampton

# ## How the exercises work
# The following exercises all rely on using all or some the split-apply-combine method.  
# For each exercise, you can uncomment and run the `Q1.hint()` line to get a hint if you're stuck, and uncomment and run `Q1.answer()`, and `Q1.solution()` for the final answer and a solution for getting there.  
# So let's jump right in!

# In[37]:


class Q1:
    @staticmethod
    def hint():
        print('HINT: You can compute the survival rate by taking the average of the `Survived` column')
    @staticmethod
    def answer():
        print('ANSWER: male: 0.188908, female: 0.742038')
    @staticmethod
    def solution():
        print('SOLUTION: df.groupby(\'Sex\').Survived.mean()')


# ### Q1. What was the survival rate of women, compared to that of men? 

# In[38]:


# write your solution here...


# Get a hint or check your answers by uncommenting the lines below!

# In[39]:


#Q1.hint()
#Q1.answer()
#Q1.solution()


# How did you do?

# In[40]:


class Q2:
    @staticmethod
    def hint():
        print('HINT: You can sort a series by using the `sort_values()` function!')
    @staticmethod
    def answer():
        print('''
ANSWER: 
Pclass
1    0.629630
2    0.472826
3    0.242363\n''')
    @staticmethod
    def solution():
        print('SOLUTION: df.groupby(\'Pclass\').Survived.mean().sort_values(ascending=False)')


# ### Q2. Output a ranking of the passenger classes by survival rate from best to worst? 
# There are the 3 passenger classes, specified in the Pclass column. Your output should show the classes, with the highest survival rate first. 

# In[41]:


# write your solution here...


# Get a hint or check your answers by uncommenting the lines below!

# In[42]:


#Q2.hint()
#Q2.answer()
#Q2.solution()


# Did you expect those results?

# In[43]:


class Q3:
    @staticmethod
    def hint():
        print('HINT: Don\'t forget about the `max` and `min`  aggregators!')
    @staticmethod
    def answer():
        print('''
ANSWER: 
Pclass
1    512.3292
2     73.5000
3     69.5500\n''')
    @staticmethod
    def solution():
        print('SOLUTION: df.groupby(\'Pclass\').Fare.max()')


# ### Q3. What was the most expensive fare (ticket price) for each passenger class? 

# In[44]:


# write your solution here...


# Get a hint or check your answers by uncommenting the lines below!

# In[45]:


#Q3.hint()
#Q3.answer()
#Q3.solution()


# In[46]:


class Q4:
    @staticmethod
    def hint():
        print('HINT: `groupby` can also take a list!')
    @staticmethod
    def answer():
        print('''
ANSWER: 
Embarked  Pclass
C         1         0.694118
Q         2         0.666667
S         1         0.582677
C         2         0.529412
Q         1         0.500000
S         2         0.463415
C         3         0.378788
Q         3         0.375000
S         3         0.189802\n''')
    @staticmethod
    def solution():
        print('SOLUTION: df.groupby([\'Embarked\', \'Pclass\']).Survived.mean().sort_values(ascending=False)')


# ### Q4. Which combination of embarkment location and passenger class had the highest survival rate?

# In[47]:


# write your solution here...


# Get a hint or check your answers by uncommenting the lines below!

# In[48]:


#Q4.hint()
#Q4.answer()
#Q4.solution()


# **Nice work! Now we are really getting somewhere with unpicking (and even predicting!) survival rates across the different passengers of the Titanic.**
