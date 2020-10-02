#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


#Read data
data = pd.read_csv('../input/Mall_Customers.csv')
data.head()


# In[4]:


#Create Arrays
age = data['Age']
income = data['Annual Income (k$)']
spending_score = data['Spending Score (1-100)']
gender = data['Gender']


# In[5]:


#Income-Score
plt.bar(spending_score,income,color="green")
plt.xlabel("Spending Score (1-100)")
plt.ylabel('Income K($)')
plt.title("Spending Score - Income Relations")
plt.show()


# In[6]:


#Age-Spending Score

plt.bar(age,spending_score,color="black")
plt.xlabel("Age")
plt.ylabel('spending_score')
plt.title("Age - Spending Score Relations")
plt.show()


# In[7]:


#Age-Income

plt.bar(age,income,color="blue")
plt.xlabel("Age")
plt.ylabel('Income K($)')
plt.title("Age - Income Relations")
plt.show()

