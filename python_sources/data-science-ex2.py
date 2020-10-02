#!/usr/bin/env python
# coding: utf-8

# # Ex 2
# *You can hand in the assignment in pairs.* 
# 
# In this exercise we will implement the naive base algorithm you've learned in class.<br>
# We are going to classify to male vs female murderer based on features in the data.
# 
# Don't forget to run the cells.
# ### Just run the following cell no code needed.

# In[9]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

# Ignoring runtime warnings.
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print('Cell Executed ^_1_^')

# Any results you write to the current directory are saved as output.


# # Load and peek at the data
# The data we are going to work with is comprised of ~650000 murders the FBI recorded.
# 
# For each murder they recorded many features such as : Year, Month, Victim Sex, Victim Age, Relationship (Victim - perpetrator) and many more.<br/>
# Let's take a look.

# In[10]:


original_df = pd.read_csv('../input/database.csv')
print("Cell Executed ^_2_^")
original_df.head()


# # Preprocess data for analysis
# To make our life a little bit easier we are going to focus our view on 3 features:
# 1. Relationhsip (The relationship between the victim and the perpetrator.
# 1. Weapon (The weapon used to commit the murder).
# 1. Victim Sex (Speaks for itself).
# 

# In[11]:


original_df = pd.read_csv('../input/database.csv')

relationship_features = ['Stranger', 'Family', 'Neighbor']
relationship_mask = original_df['Relationship'].isin(relationship_features)

weapon_features = ['Handgun', 'Knife', 'Blunt Object', 'Strangulation']
weapon_mask = original_df['Weapon'].isin(weapon_features)

victim_sex_features = ['Male', 'Female']
victim_sex_mask = original_df['Victim Sex'].isin(victim_sex_features)

perpetrator_features = ['Male', 'Female']
perpetrator_mask = original_df['Perpetrator Sex'].isin(perpetrator_features)

df = original_df[relationship_mask & weapon_mask & victim_sex_mask & perpetrator_mask]
df = df[['Relationship', 'Weapon', 'Victim Sex' , 'Perpetrator Sex']]

df.head()


# # Categorize
# To make our processing life easier we are going to turn our string value into numeric values.
# Here are the codes we are going to use:<br/>
# 
# __Relationship__ : (0 = Family)  | (1 = Neighbor) | 2 = Stranger)
# 
# __Weapon__ : (0 = Blunt Object), (1 = Handgun), (2 = Knife), (3 = Strangulation)
# 
# __Victim Sex__ : (0 = Female) | (1 = Male) 
# 
# __Perpetrator_sex__ : (0 =Female) | (1 = Male)

# In[12]:


categorize = ['Relationship', 'Weapon', 'Perpetrator Sex', 'Victim Sex']

for col in categorize : 
    df[col] = df[col].astype('category')
    df[col] = df[col].astype('category').cat.codes

df.head()


# # Split the dataset into female and male.
# Since we want to compute the probabilites under each class (in our case female and male) <br/>
# We are going to split our data set into 2 : one for the male and one for the female

# In[13]:


df_mask = (df['Perpetrator Sex'] == 1)
df_male = df[df_mask]
df_female = df[~df_mask]
df_male.head(), df_female.head()


# # Helper Functions

# In[1]:


def get_values_from_array_sorted(arr):
    '''
    Return an array containing all the values in the given array arr. The returned array is sorted from lowest to highest.
    Example: arr=[6,6, 1,1,2,1,3,3,3] will return [1,2,3,6]
    HINT : Use .sort() to sort your array before returning it. 
    '''
    

def test_get_values_from_array():
    arr = [1,2,3]
    if get_values_from_array_sorted(arr) != [1,2,3]:
        print("Failed on arr {}".format(arr))
        return False
    
    arr = [2,1,3]
    if get_values_from_array_sorted(arr) != [1,2,3]:
        print("Failed on arr {}".format(arr))
        print("Did you forget to sort?")
        return False
    
    arr = [1,2,3,3,3,1,1,4,4]
    if get_values_from_array_sorted(arr) != [1,2,3,4]:
        print("Failed on arr {}".format(arr))
        return False
    
    return True

if test_get_values_from_array():
    print("test_get_values_from_array passed.")


# In[2]:


def count_value_in_array(arr, value):
    '''
    Count how many times the given value appear in the given array arr.
    Example: arr=[1,1,2,2,2,2,6] value=2 will return 4.
    '''


def test_count_value_in_array():
    arr = [1,2,3,2,2,2,2,3]
    if count_value_in_array(arr, 1) != 1:
        print("Failed on arr {}, and value {}".format(arr, 1))
        return False
    
    if count_value_in_array(arr, 2) != 5:
        print("Failed on arr {}, and value {}".format(arr, 2))
        return False
    
    if count_value_in_array(arr, 3) != 2:
        print("Failed on arr {}, and value {}".format(arr, 3))
        return False
    
    if count_value_in_array(arr, 4) != 0:
        print("Failed on arr {}, and value {}".format(arr, 4))
        return False
    
    return True

if test_count_value_in_array():
    print("test_count_value_in_array passed.")


# In[3]:


def compute_probabilities(arr):
    '''
    For each value in the given array arr compute the probability of getting that value if you
    sample a value from the array uniformly.
    Example : arr=[1,1,2,2,3,3,3,3] return =[0.25, 0.25, 0.5]
    '''


def test_compute_probabilities():
    arr = [1,1,1,2,2,2]
    if compute_probabilities(arr) != [0.5, 0.5]:
        print("Failed on arr {}".format(arr))
        
        return False
    
    return True

if test_compute_probabilities():
    print("test_compute_probabilites passed")


# # Computing Probabilites
# We will now use our compute probabilites function to .. compute probabilites for each class. <br/>
# These probabilites will then enable use to preform classification for new data.<br/>
# You should call `compute_probabilites(df_class[col_name)` think for each case which df class you need and what is the column name.
# ### Female class

# In[4]:


p_female_relationship = 0
p_female_weapon = 0
p_female_victimsex = 0
print('Cell executed ^__^')


# ### Male Class

# In[5]:


p_male_relationship = 0
p_male_weapon = 0
p_male_victimsex = 0
print('Cell executed ^__^')


# ## Questions
# Let's now ask questions to see how we can use our data.<br/>
# Here is our mapping between numeric value and string value:
# 
# __Relationship__ : (0 = Family)  | (1 = Neighbor) | (2 = Stranger)
# 
# __Weapon__ : (0 = Blunt Object), (1 = Handgun), (2 = Knife), (3 = Strangulation)
# 
# __Victim Sex__ : (0 = Female) | (1 = Male) 
# 
# __Perpetrator_sex__ : (0 =Female) | (1 = Male)
# 
# ### Question 1:
# If you know your perpetrator is a woman, what is the most likely relatioship bewteen her and the victim?

# In[14]:


# Display the probabilites which help you support your question.


# ### Solution 1:
# Write your solution in this cell.

# ### Question 2:
# If you know your perpetrator is a man, what is the most likely weapon they used?

# In[6]:


# Display the probailites which help you support your question


# ### Solution 2:
# Write your solution in this cell.

# ### Question 3:
# What is more likely that a woman perpetrator had a family relation to her victim
# or that a male perpetartor had a family relation to her victim?

# In[7]:


# Display the probabilites which help you support your question:


# ### Solution 3:
# Write your solution in this cell.

# ### Question 4:
# What is more likely that a woman perpetrator had a male victim
# or that a male perpetartor had a male victim?

# ### Question 5.a
# You are investegating a crime and you know the following things about your case:
# 1. The victim was a female.
# 1. The perpetrator had a family relation to their victim
# 1. The perpetrator used a knife. <br/>
# Compute the likelihood under each class (male, female). <br/>
# Recall the likelihood is :<br/>
# $$P(x_1, x_2, x_3|C) = p(x_1|C) \cdot p(x_2|C) \cdot p(x_3|C)$$

# In[ ]:


# Display the probabilites which help you support your question:


# ### Solution 5.a
# Female likelihood for the events = 
# 
# male likelihood for the events = 

# ### Question 5.b
# If all went well you find that the liklihood of the female is much higher that that of a male.<br/>
# But still in your case it is more likely the your perpetrator is a male. </br>
# Explain why this is?
# HINT : Compute the prior for each class

# In[ ]:


# Write code to support your answer here:


# ### Solution 5.b
# Write your solution here.

# In[ ]:





# # End of Exercise
# To submit the exercise, click *File* -> *Download Notebook*.<br>
# After your donwloaded change the name of the file to : 
# ID1_ID2.ipynb if you are submitting in pairs or
# ID1.ipynbg if you are submitting alone (and ofcourse replace ID1 with your personal ID)
# Submit the downloaded file in the moodle under ex2.

# In[ ]:




