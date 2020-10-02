#!/usr/bin/env python
# coding: utf-8

# <h3><b>Data Science For Student Exam Performance </b></h3>
# <h3><b>Content</b></h3>
# <ul>
#     <a href='#1'><li>Import Library</li></a>
#     <a href='#2'><li>Data Exploratory Analysis</li></a>
#     <a href='#3'><li>Column Operations</li></a>
#     <a href='#4'><li>Matplotlib</li></a>
#         <ul>
#             <a href='#5'><li>Line Plot</li></a>
#             <a href='#6'><li>Scatter Plot</li></a>
#             <a href='#7'><li>Histogram</li></a>

# <p id='1'><h3><b>Import Library</b></h3></p>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <p id='2'><h3><b>Data Exploratory Analysis</b></h3></p>

# In[ ]:


# Read the data from the csv file  
data = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")


# In[ ]:


# First 5 rows
# data.iloc[0:3] Can also be written as
data.head()


# In[ ]:


# Last 5 rows
data.tail()


# In[ ]:


# Random rows
data.sample(3)


# In[ ]:


# Learn about the size of the .csv file
data.shape


# In[ ]:


# Datacontrol
# Look for missing values
data.isnull().values.any()


# In[ ]:


# List for missing values
data.isnull().sum()


# In[ ]:


# To get a short summary of the dataframe
data.info()


# In[ ]:


# datatype of the dataframes
data.dtypes


# In[ ]:


# datatype of the dataframes
data.iloc[:,0:5].dtypes


# In[ ]:


# Learning the datatype with iloc
data.iloc[:,5:8].dtypes


# In[ ]:


# Used to view some basic statistical details
data.describe()


# In[ ]:


# Used to find the pairwise correlation of all columns in the dataframe
data.corr()


# In[ ]:


#correlation map
sns.heatmap(data.corr(), annot=True, fmt= '.1f')
plt.show()


# <p id='3'><h3><b>Column Operations</b></h3></p>

# In[ ]:


# Learning columns
data.columns


# In[ ]:


# Merge with columns _
data.columns = [each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in data.columns]


# In[ ]:


# Convert strings in the Series/Index to be capitalized.
data.columns = [each.title() for each in data.columns]


# In[ ]:


# Show columns
for i,col in enumerate(data.columns):
    print(i+1,". columns ",col)


# In[ ]:


# Show unique Gender
data['Gender'].unique()


# In[ ]:


# Show count Gender
data['Gender'].value_counts()


# In[ ]:


# Show count Lunch
data['Lunch'].value_counts()


# In[ ]:


# Show count Race/Ethnicity
data['Race/Ethnicity'].value_counts()


# In[ ]:


b=(data['Writing_Score']<50).value_counts()
b


# In[ ]:


x=data['Race/Ethnicity'].value_counts().values
x


# <p id='4'><h3><b>Matplotlib</b></h3></p>
# <p id='5'><h3><b>Line Plot</b></h3></p>

# In[ ]:


data.Math_Score.plot(kind = 'line', color = 'lime',label = 'Math_Score',linewidth=1,alpha = 1,grid = True,linestyle = ':')
#data.Reading_Score.plot(color = 'c',label = 'Reading_Score',linewidth=1, alpha = 1,grid = True,linestyle = '-.')
data.Writing_Score.plot(color = 'blue',label = 'Writing_Score',linewidth=1, alpha = 0.5,grid = True,linestyle = '--')
plt.legend(loc='lower center')
plt.xlabel('rows')
plt.ylabel('score')
plt.title('Math_Score & Writing_Score')
plt.show()


# <p id='6'><h3><b>Scatter Plot</b></h3></p>

# In[ ]:


data.plot(kind='scatter',x='Reading_Score',y='Writing_Score', alpha=.7, color='magenta', label = 'kasjsk')
# plt.xlabel("Reading_Score") not writting because x belive 
plt.title('scatter plot')
plt.legend(loc='right')
plt.show()


# <p id='7'><h3><b>Histogram</b></h3></p>

# In[ ]:


data.Math_Score.plot(kind='hist', bins=50, facecolor = "blue", alpha=.5, grid = True)
plt.show()

