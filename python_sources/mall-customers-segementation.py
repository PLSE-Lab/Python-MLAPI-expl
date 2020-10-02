#!/usr/bin/env python
# coding: utf-8

# Hello Everyone,
# 
# Thank you for viewing this kernel, I want to build some visualizations using seaborn to uncover some insights in the mall customer segmentation dataset. Kindly go through it and share your thoughts.
# 
# Happy Viewing!!!
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  #visualization
import matplotlib.pyplot as plt #build plot as well as styling plot

pd.set_option('display.max_rows', 10)

import sklearn as sk

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


mallData = pd.read_csv("../input/Mall_Customers.csv")


# Let's look at the first five rows on the dataset

# In[ ]:


mallData.head()


# Let's look at the dataset. Inspect the columns and their respectively datatype

# In[ ]:


mallData.info()


# From the dialog above, we call see that all the columns are non-nullable. We can confirm if there is any null value in our dataset

# In[ ]:


mallData.isnull().values.any()


# The result above shows there is no null values in the dataset

# **Let's provide some visualizations that will help us to understand the dataset better. Remember a picture is worth a thousand words**

# First, we will count the number of males and females who carried out a transaction in the given dataset

# In[ ]:


sns.set(style="darkgrid")       #style the plot background to become a grid
genderCount  = sns.countplot(x="Gender", data =mallData).set_title("Gender_Count")  


# More females used the mall that males in this dataset. One who love to know the average age of persons used the mall. This is shown below:

# In[ ]:


genderCountAge = sns.boxplot(x="Gender", y = "Age", data =mallData).set_title("Gender_Count")  


# With the boxplot above, we can see that the average age of females in this dataset is quite lower than that of the males. However one would think among the customers, what's the age group that visited the mall most in this given time?  The visual below will make an attempt to show that

# In[ ]:


genderCountAge = sns.swarmplot(x="Gender", y = "Age", data =mallData).set_title("Gender_Count")  


# By inspection of the figure above, we can see that persons under the age of 20 visited the mall most in the male category while persons in the 30 year mark made most of the purchase in the female category.

# I would love to introduce the annual income data into the mix but before we do that, let's inspect the data points

# In[ ]:


mallData["Annual Income (k$)"].describe()


# In the view above, one can tell that the average annual income is $60,560 with majority of the group earning above this value. There are over 25 income classes in this dataset as shown below

# In[ ]:


mallData["Annual Income (k$)"].unique()


# Now let's invite the annual income data point to the party. Let's see the income distribution between the gender classes.

# In[ ]:


genderCountIncome = sns.boxplot(y="Gender", x = "Annual Income (k$)", data =mallData).set_title("Gender Count by Annual Income")  


# We can intrepret the box plot above by saying the average male annual income value is slightly higher than their female counterpart. We would like to predict the working population state by their ages and annual income value. This is demonstrated in the visual below 

# In[ ]:


g = sns.violinplot(y ="Annual Income (k$)", x= "Age", data =mallData).set_title("Annual Income Distribution by Age")
plt.xticks(rotation='vertical')


# We can see that persons between age of 26 and 45 have the highest annual income values than others. We can say it's a fairly aging working population in this set.  *can anyone help me with the better way of representing the age axis?
# *
# 
# There is a notion that **Women Spend More Than Men**. How True is this? We can find that out when we introduce the "Spending Score (1-100)" into the mix. Let's get to know these data points better 

# In[ ]:


mallData["Spending Score (1-100)"].describe()


# In[ ]:


mallData["Spending Score (1-100)"].unique()


# From the figures above, one can tell that the average spending score is 50.2 with majority of the group earning above this value. There are over 80 spending classes in this dataset. 
# 
# Before we respond to the notion, let's see the age distribution among the spending classes

# In[ ]:


ageDisSpend = sns.relplot(x="Age", y = "Spending Score (1-100)", data =mallData)


# We can see from the relplot above, Customers under the age of 40 have a higher spending score than those who are old than 40. Let's respond to the notion

# In[ ]:


ageDisGender = sns.boxplot(x="Gender", y = "Spending Score (1-100)", data =mallData)


# **Drums Rolling!!!!**
# 
# Given the average spending score of both classes is close (really close .....less than 2 points). I would say women and men do have the same spending power based on this given data set.  

# **Does earning more guarantee a higher spending score?** Hmmm. Let's find out by generate a plot between "spending score" and "annual income"

# In[ ]:


with sns.axes_style("white"):
    sns.jointplot(x="Annual Income (k$)", y = "Spending Score (1-100)", data =mallData)


# The notion that earning more guarantees a higher spending score is not supported with the dataset Since most persons with high spending score are not the richest in the dataset. This is also demonstrated in the figure below

# In[ ]:


with sns.axes_style("white"):
    sns.jointplot(x="Annual Income (k$)", y = "Spending Score (1-100)", kind = "kde", data =mallData)

