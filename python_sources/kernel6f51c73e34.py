#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ** INTRODUCTION:**
#                Description of data:	
#             This Credit Risk Data contains data on 11 variables and 1000 loan applicants and their classification        whether an applicant is considered a Good or a Bad credit risk. In this dataset, each entry represents a            person who takes a credit by a bank. Each person is classified as good or bad credit risks according to the        set of attributes.
#             
#    **Note = It is worse to class a customer as good when they are bad, than it is to class a customer as bad             when they are good.**
#      
#    **i.e. Misclassification rate (False Positive Rate) = FP/N**
# 
#  
# **CONTENT:**
# 
# Age (numeric)
# 
# Sex (text: male, female)
# 
# Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
# 
# Housing (text: own, rent, or free)
# 
# Saving accounts (text - little, moderate, quite rich, rich)
# 
# Checking account (numeric, in DM - Deutsch Mark)
# 
# Credit amount (numeric, in DM)
# 
# Duration (numeric, in month)
# 
# Purpose(text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)
# 
# Risk (Value target - Good or Bad Risk)
# 

# In[ ]:





# In[ ]:


#import required packages for further program



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#to read our datafile using package pandas

data = pd.read_csv("/kaggle/input/german-credit-data-with-risk/german_credit_data.csv")
print (data.columns)
data.head(10)


# In[ ]:


#Let`s clean our data 
#Encoding the all categorical data variables and missing values   

data['Sex'] = data['Sex'].map({'male':0,'female':1})
data['Housing'] = data['Housing'].map({'own':0, 'rent':2,  'free':1})
data['Saving accounts'] = data['Saving accounts'].map({'little':0,  'moderate':1,   'quite rich':2,  'rich':3,  'NaN':4})
data['Checking account'] = data['Checking account'].map({'little':0, 'moderate':1, 'rich':2,'NaN':3})
data['Purpose'] = data['Purpose'].map({'car':0, 'furniture/equipment':1, 'radio/TV':2, 'domestic appliances':3,'repairs':4, 'education':5, 'business':6, 'vacation/others':7})
data['Risk'] = data['Risk'].map({'good':0,'bad':1})
data["Saving accounts"].fillna(4,inplace=True)
data["Checking account"].fillna(3,inplace=True)
data.head(10)


# In[ ]:


#you can save/export  your cleaned file anyweare on your PC  
#enter your path/location with new file name in following command 
#data.to_csv("C:/Users/stat 123/Desktop/data coding in python.csv",index=False)


# **Heatmap:**
# 
#    It is used to investigate the dependence between multiple variables at the same time. The result is a table  containing the correlation coefficients between each variable and the others.
#       
#    **Aim: To check variables in datasets have dependency with each other or not.**
# 

# In[ ]:



hmap = data.corr()
plt.subplots(figsize=(10, 9))
sns.heatmap(hmap, vmax=.8,annot=True,cmap="coolwarm", square=True)


# :from above figure we can see that, the values of correlation coefficients are very less for each variable. There is no dependency in variables. 
