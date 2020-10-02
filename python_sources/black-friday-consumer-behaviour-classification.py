#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


blackFriday = pd.read_csv("../input/BlackFriday.csv")


# In[ ]:


blackFriday.head(5)


# 
# Black Friday Sales Data Analysis
# * Spending based on Gender
# * Spending based on Age
# * Spending based on Marital status
# * Spending based on Occupation 
# * Spending based on City Category

# In[ ]:


# Setting the plot size and style 


# In[ ]:


plt.figure(figsize=(24,20))
sns.set_style("darkgrid")


# In[ ]:


# Number of Females and Males who participated in the Black Friday Sales


# In[ ]:


sns.countplot(blackFriday.Gender)


# In[ ]:


# Get Percentage of Male and Female who participated in the sale


# In[ ]:


explode = (0.1,0)  # Slices out the first slice in the pie
ax1 = plt.subplot()
ax1.pie(blackFriday['Gender'].value_counts(), explode=explode,labels=['Male','Female'], autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()


# In[ ]:


# Now that we understand that number of Men who participated in the Black Friday sales is much higher
# It may be safe to assume that Men spent more.
# However lets prove it with a visualisation.


# In[ ]:


def spendByGroup(group, column, plot):
    blackFriday.groupby(group)[column].sum().sort_values().unstack().plot(kind=plot, stacked = True)
    plt.ylabel(column)


# In[ ]:


group = ('Age', 'Gender')
spendByGroup(group, 'Purchase', 'bar')


# Men and Women both in the age group of 26-35 spent the most during the Black Friday sales. This could be attributed to number of factors such as stability in job, maturity etc.  One  of the major reasons Men in this age bracket have spent more can be attibuted to the fact that after marriage it could be the men, who are predominantly the primary earner and items could have been bought for by them. 

# In[ ]:


# We should also check how marriage affected Men and Women in making a decision about their purchases.


# In[ ]:


group = ('Marital_Status', 'Gender')
spendByGroup(group, 'Purchase', 'bar')


# A plot of Purchase vs the Marital Status grouped by Gender indicates that unmarried men made most of the purchases during the Black Friday sales, followd by married me. Women in general spent less. The observation challenges the stereotype of Women spending more. However, from a general understanding, hushold and electronic goods are sold during Black Friday sales, which intrest men more than women.  

# In[ ]:


# We should also check how marriage and age have influenced people in making a decision about their purchases.


# In[ ]:


group = ('Age', 'Marital_Status')
spendByGroup(group, 'Purchase', 'bar')


# A plot of Purchase vs Age grouped by Matrial Status indicates that unmarried people tend to spend higher and the maximum spending is in the age bracket of 26-35. 

# In[ ]:


# Lets check which category of cities spent the most


# In[ ]:


blackFriday.City_Category.unique()


# In[ ]:


sns.countplot(blackFriday['City_Category'], hue=blackFriday['Gender'])


# In[ ]:


def sliceOfGroup(column, explode, labels):
    ax1 = plt.subplot()
    ax1.pie(blackFriday[column].value_counts(), explode=explode,labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.tight_layout()
    plt.legend()


# In[ ]:


explode = (0.1, 0, 0)
labels = blackFriday.City_Category.unique()


# In[ ]:


sliceOfGroup('City_Category', explode, labels)


# In[ ]:


# Spend grouped by Age for the City Category
group = ('Age', 'City_Category')
spendByGroup(group, 'Purchase', 'bar')


# In[ ]:


# Spend grouped by Marital_Status for the City Category
group = ('City_Category', 'Marital_Status')
spendByGroup(group, 'Purchase', 'bar')


# In[ ]:


# Spending by product Category grouped by Age, Gender, Marital Status and City Category


# In[ ]:


# Spending for Product_Category_1 grouped by Age
group = ('Product_Category_1', 'Age')
spendByGroup(group, 'Purchase', 'bar')


# In[ ]:


# Spending for Product_Category_1 grouped by Gender
group = ('Product_Category_1', 'Gender')
spendByGroup(group, 'Purchase', 'bar')


# In[ ]:


# Spending for Product_Category_1 grouped by Marital_Status
group = ('Product_Category_1', 'Marital_Status')
spendByGroup(group, 'Purchase', 'bar')


# In[ ]:


# Spending for Product_Category_1 grouped by City_Category
group = ('Product_Category_1', 'City_Category')
spendByGroup(group, 'Purchase', 'bar')


# In[ ]:


# Correlation Matrix


# In[ ]:




