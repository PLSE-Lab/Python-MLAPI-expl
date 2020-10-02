#!/usr/bin/env python
# coding: utf-8

# This is my first project, and I try to code what I have learned so far. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #add the library to plot
import seaborn as sns #visulization tools

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/StudentsPerformance.csv") # adding the csv file


# In[ ]:


data.info() # as seen there is no missing value


# In[ ]:


data.shape # the number of rows, and columns


# In[ ]:


data.dtypes # data types in the csv file, the column datatypes are True, I don't need to change


# In[ ]:


data.describe() # some values of numeric type of columns


# In[ ]:


data.columns # data.columns.values


# In[ ]:


data.head() # first 5 rows


# In[ ]:


data.tail() # last 5 rows


# In[ ]:


data.columns = data.columns.str.replace(" ", "_") # its better when working with columns
data.columns


# In[ ]:


# how many males and females are there, I want to see this as pie chart
labels = data.gender.value_counts().index
colors = ["pink","blue"] 
explode = [0,0]
sizes = data.gender.value_counts().values

plt.figure(figsize=(7,7))
plt.pie(sizes, explode=explode, labels=labels, colors= colors, autopct="%1.1f%%")
plt.title("the ratio of females, and males")
plt.show()


# In[ ]:


# which gender is more succesful
data.groupby("gender").mean() # females are more succesful in reading, and writing, males are in maths


# In[ ]:


math_mean_of_gender = data.groupby("gender").math_score.mean()
# sort by math_score
math_mean_of_gender.sort_values(inplace=True)
x = math_mean_of_gender.index.tolist()
y = math_mean_of_gender.values.tolist()

# set axis labels
plt.xlabel("gender")
plt.ylabel("math score mean")
# set title
plt.title("mean scores for each gender")
plt.bar(x,y)
plt.show()


# In[ ]:


# math success of each race
math_mean_of_race = data.groupby("race/ethnicity").math_score.mean()
math_mean_of_race.sort_values(inplace=True)
x = math_mean_of_race.index.tolist()
y = math_mean_of_race.values.tolist()

plt.xlabel("race")
plt.ylabel("math score mean")
# set title
plt.title("mean scores for each race")
plt.bar(x,y)
plt.show()


# In[ ]:


# test_preparation course vs math_score, whats the relation?
data.groupby("test_preparation_course").mean() # finishing the course makes students get higher score


# In[ ]:


# reading success based on finishing the test preparation course
testpreparation_vs_reading = data.groupby("test_preparation_course").reading_score.mean()
testpreparation_vs_reading.sort_values(inplace=True)
x = testpreparation_vs_reading.index.tolist()
y = testpreparation_vs_reading.values.tolist()

plt.xlabel("test preparation")
plt.ylabel("reading score mean")
# set title
plt.title("reading scores based on finishing the course")
plt.bar(x,y)
plt.show()


# In[ ]:


# showing writing score based on parents education level, and students's gender
plt.figure(figsize=(10,7))
sns.boxplot(x="parental_level_of_education",y="writing_score",hue="gender",data=data,palette="PRGn") 
plt.show()


# In[ ]:


# info based on parents education level and gender with swarmplot
plt.figure(figsize=(10,6))
sns.swarmplot(x="parental_level_of_education",y="reading_score",hue="gender",data=data)
plt.show()


# In[ ]:


# relation of each lesson scores
sns.pairplot(data)
plt.show()


# In[ ]:




