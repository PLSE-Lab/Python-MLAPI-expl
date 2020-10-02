#!/usr/bin/env python
# coding: utf-8

# # Quiz 4 Question 1 EDA & Pandas Tutorial
# #Darian Tjiandra
# #dtjiandr@usc.edu

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


# The csv file tells about the information of each individual in Nigeria that is suspected or confirmed to have a disease. It tells about their details too, including gender (male or female), settlement (rural or urban), status (Dead or alive), confirmed or not confirmed sickness, and many more. 

# a. Load the csv file as a DataFrame named diseases and display the first 5 rows.

# In[ ]:


diseases = pd.read_csv('/kaggle/input/disease-outbreaks-in-nigeria-datasets/meningitis_dataset.csv')
diseases.head(5)


# b. How many rows and columns does the DataFrame have?

# In[ ]:


diseases.shape


# c. How many people from both gender are alive?

# In[ ]:


diseases.query("health_status =='Alive'")     .gender     .value_counts()     .head()


# There are 73461 Female and 68733 Male who is alive. Based on this data, we can obtain the number of Male and Female who are stil fighting their diseases. And based on this data, we know that there are not much difference in the number, meaning that the virus attacks anyone disregarding their gender.

# d. Display the top 10 states with the highest number of people confirmed to have a disease. 

# In[ ]:


diseases.groupby('state')     ['confirmed']     .sum()     .sort_values(ascending=False)     .iloc[:10]


# This data shows the state with highest number of civilian with confirmed disease. Based on the data, we should deliver more health support to Kwara, Kebbi, Imo since they are the state with highest number of people with confirmed disease. Civilian should stay away from this area to avoid contingency 

# e. Create a histogram of the age distribution

# In[ ]:


diseases.age.plot.hist()


# The histogram showed the number of people confirmed or unconfirmed with a disease based on their age. The histogram showed a declining pattern meaning that the older the age is, the less people who are suspected with disease. It means that childrens should be careful because they are more prone to getting sick.

# f. What is the age of the oldest person in the data?

# In[ ]:


diseases.age.max()


# It shows the oldest person who is suspected to have a disease.
