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


# This notebook is a tutorial on exploratory data analysis. 
# 
# Sections
# 1. Identifing the Question
# 2. Importing Data
# 3. Previewing Dataset
# 4. Exploring Class
# 5. Exploring Gender
# 6. Exploring Age
# 7. Visualization
# 8. Conclusion
# 

# ### 1. What question are we trying to solve?: *What factors influence whether or not someone survives?*

# ### 2. Importing Data:
# Read the train data set. Name the data frame *titanic*. We will only be working with the training dataset for this tutorial.

# In[ ]:


titanic = pd.read_csv("/kaggle/input/titanic-cleaned-data/train_clean.csv")


# ### 3. Previewing the dataset
# (a) Display the first few rows of the dataset using *.head()*

# In[ ]:


titanic.head()


# (b)How many rows and columns does *titanic* have? Use it's *shape* attribute

# In[ ]:


titanic.shape


# (c) What are the column names?
# 
# **Significance**: The column names are let's us knows what **factors** we can analyze to see what affects survival. 

# In[ ]:


titanic.columns


# 
# ** (d) How many people survived? How many didn't survive? Use *groupby().size()*
# **
# 
#  

# In[ ]:


titanic.groupby("Survived").size()
# can see that the sum of people who survived and didn't survive equal the total number of rows (891)


# ** (e) What percentage of people did and didn't survive?** This number is good to keep in mind for comparision when we explore factors that influence survival rates. 

# In[ ]:


titanic.groupby("Survived").size()/titanic.shape[0]*100


# (f) Select the oldest age in the data set. 
# 

# In[ ]:


titanic["Age"].max()


# (g) What is the median fare?

# In[ ]:


titanic["Fare"].median()


# (h) Select the information of the people who paid the top 10 highest fares.
# Use *.sort_values()* and *.iloc* to select the top 10

# In[ ]:


titanic.sort_values("Fare",ascending=False).iloc[0:10]


# **Observation**: It seems that people who paid the highest fees had better chances of surviving.

# ### 4. Exploring Class
# **Motivation**: Higher fare seems to increase survival rate. Higher fare may be associated with class. Let's take a look at how class affects survival rate. 
# 
# **concepts**: grouby(), groupby().size(), .sum()  

# ** (a) How many people are in each class? Use *.groupby().size()* and store the results in a variable named *total***
# 

# In[ ]:


total = titanic.groupby("Pclass").size()
total


# ** (b) How many people in each class survived? Use *.groupby()* and *.sum()* and store the results in a variable called *survived*
# **
# 

# In[ ]:


survived = titanic.groupby("Pclass").sum()["Survived"]
survived


# ** (c) What is the percentage of people that survived for each class? Use *total* and *survived* from above.**

# In[ ]:


survived/total*100


# **Observation**: Here, we can see that the class someone is in affects the chance they survive or not. 63% of people from the first class survived while only 24% of people in the 3rd class survived. 

# ### 5. Exploring Gender

# (a) What percentage of women survived? What percentage of men survived?

# In[ ]:


titanic.groupby("Sex").sum()["Survived"]/titanic.groupby("Sex").size()*100


# **Observation**: It seems that gender affects wheter someone will survive or not. 74% of females survived while only 19% of males survived. 

# **Looking at gender and class:** 
# It seems that sex and class strongly influences whether someone will survive. People in first class had higher chance of surviving and females had a higher chance of surviving than males. 
# 
# ** (b) What percentage of females from the 1st class survived?** 
# 
# Use *.query()* and boolean indexing

# In[ ]:


# number of female in 1st class that survived
fsurvived = titanic[titanic["Sex"] == "female"].query("Pclass == 1").sum()["Survived"]
# number of femaes in 1st class
ftotal = titanic[titanic["Sex"] == "female"].query("Pclass == 1").shape[0]
fsurvived/ftotal*100


# **Interpretation**: 97% of females in the first class 

# ### 6. Exploring Age

# ### What is the median age of people who survived and didn't survive? Use *.groupby()* and *.mean()*

# In[ ]:


titanic.groupby("Survived").mean()["Age"]


# ### 7. Visualization 
# Plot a histogram of age 

# In[ ]:


titanic.Age.plot.hist()


# # 8. Conclusion:
# - Gender: Females had an advantage for survival.
# - Class: Higher class people had an advantage for survival
# - Age: There is no signicant difference age in the 
